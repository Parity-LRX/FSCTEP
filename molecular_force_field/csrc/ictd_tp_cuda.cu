// Group-level CUDA kernels for HarmonicChannelWiseTensorProduct (SO3, broadcast_rhs).
//
// All kernels assume:
//   - broadcast_rhs channel mode (mul_in2 == 1, so `b_rhs` has shape (B, m2))
//   - float32 compute dtype
//   - inputs and outputs are contiguous
//   - `seg_starts[p]` / `seg_ends[p]` describe segment offsets into the group's
//     concatenated k axis, k_total = sum over segments of (2*l3+1).
//
// The forward produces a zero-padded (B, P, O, k_total) tensor; each path p's
// valid region is [seg_starts[p], seg_ends[p]), the rest is explicitly zeroed.
// Callers slice per segment to accumulate into per-l3 outputs. The backward
// kernels treat grad_y outside a segment as zero regardless of what the caller
// writes there.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>

using torch::Tensor;

namespace {

// ---------------------------------------------------------------------------
// Forward
// y[b,p,o,k] = sum_{c,m,n} a[b,c,m] * b_rhs[b,n] * U[m*m2+n,k] * W[p,o,c]
//              for k in [seg_starts[p], seg_ends[p]), else 0.
// One CUDA block per batch index b. Shared memory caches a[b,:,:], b_rhs[b,:],
// U (group-shared), and Z[c,k] = sum_{m,n} a[b,c,m]*b_rhs[b,n]*U[m*m2+n,k].
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void channelwise_group_forward_kernel(
    const scalar_t* __restrict__ a,           // (B, C, m1)
    const scalar_t* __restrict__ b_rhs,       // (B, m2)
    const scalar_t* __restrict__ U,           // (m1*m2, k_total)
    const scalar_t* __restrict__ W,           // (P, O, C)
    const int64_t* __restrict__ seg_starts,   // (P,)
    const int64_t* __restrict__ seg_ends,     // (P,)
    scalar_t* __restrict__ y,                 // (B, P, O, k_total)
    const int C,
    const int m1,
    const int m2,
    const int k_total,
    const int P,
    const int O) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  scalar_t* smem = reinterpret_cast<scalar_t*>(smem_raw);
  scalar_t* a_s = smem;                               // C*m1
  scalar_t* b_s = a_s + C * m1;                       // m2
  scalar_t* U_s = b_s + m2;                           // m1*m2*k_total
  scalar_t* Z_s = U_s + m1 * m2 * k_total;            // C*k_total

  const int b = blockIdx.x;
  const int tid = threadIdx.x;
  const int blk = blockDim.x;

  // Load a[b, :, :]
  const int a_count = C * m1;
  for (int i = tid; i < a_count; i += blk) {
    a_s[i] = a[b * a_count + i];
  }
  // Load b_rhs[b, :]
  for (int i = tid; i < m2; i += blk) {
    b_s[i] = b_rhs[b * m2 + i];
  }
  // Load U
  const int u_count = m1 * m2 * k_total;
  for (int i = tid; i < u_count; i += blk) {
    U_s[i] = U[i];
  }
  __syncthreads();

  // Z[c, k] = sum_{m, n} a[b,c,m] * b_rhs[b,n] * U[m*m2+n, k]
  const int z_total = C * k_total;
  for (int idx = tid; idx < z_total; idx += blk) {
    const int k = idx % k_total;
    const int c = idx / k_total;
    scalar_t acc = scalar_t(0);
    const scalar_t* a_c = a_s + c * m1;
    for (int m = 0; m < m1; ++m) {
      const scalar_t av = a_c[m];
      const scalar_t* U_row = U_s + (m * m2) * k_total + k;
      for (int n = 0; n < m2; ++n) {
        acc += av * b_s[n] * U_row[n * k_total];
      }
    }
    Z_s[c * k_total + k] = acc;
  }
  __syncthreads();

  // y[b, p, o, k] = sum_c Z[c, k] * W[p, o, c]   (masked outside segment)
  const int y_total = P * O * k_total;
  for (int idx = tid; idx < y_total; idx += blk) {
    const int k = idx % k_total;
    int rem = idx / k_total;
    const int o = rem % O;
    const int p = rem / O;
    const int s = static_cast<int>(seg_starts[p]);
    const int e = static_cast<int>(seg_ends[p]);
    scalar_t out = scalar_t(0);
    if (k >= s && k < e) {
      const scalar_t* W_po = W + (p * O + o) * C;
      for (int c = 0; c < C; ++c) {
        out += Z_s[c * k_total + k] * W_po[c];
      }
    }
    y[((b * P + p) * O + o) * k_total + k] = out;
  }
}

// ---------------------------------------------------------------------------
// Backward w.r.t. a
// grad_a[b,c,m] = sum_{p,o,n,k_in_seg} G[b,p,o,k] * b_rhs[b,n] * U[mn,k] * W[p,o,c]
//              = sum_k Q[b,c,k] * T[b,m,k]
// where
//   Q[b,c,k]  = sum_{p,o} G[b,p,o,k] * W[p,o,c]      (effective G is masked by segments)
//   T[b,m,k]  = sum_n b_rhs[b,n] * U[mn,k]
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void channelwise_group_transpose_a_kernel(
    const scalar_t* __restrict__ grad_y,      // (B, P, O, k_total)
    const scalar_t* __restrict__ b_rhs,       // (B, m2)
    const scalar_t* __restrict__ U,           // (m1*m2, k_total)
    const scalar_t* __restrict__ W,           // (P, O, C)
    const int64_t* __restrict__ seg_starts,   // (P,)
    const int64_t* __restrict__ seg_ends,     // (P,)
    scalar_t* __restrict__ grad_a,            // (B, C, m1)
    const int C,
    const int m1,
    const int m2,
    const int k_total,
    const int P,
    const int O) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  scalar_t* smem = reinterpret_cast<scalar_t*>(smem_raw);
  scalar_t* G_s = smem;                               // P*O*k_total
  scalar_t* b_s = G_s + P * O * k_total;              // m2
  scalar_t* U_s = b_s + m2;                           // m1*m2*k_total
  scalar_t* Q_s = U_s + m1 * m2 * k_total;            // C*k_total
  scalar_t* T_s = Q_s + C * k_total;                  // m1*k_total

  const int b = blockIdx.x;
  const int tid = threadIdx.x;
  const int blk = blockDim.x;

  // Load grad_y[b] with segment masking
  const int g_total = P * O * k_total;
  for (int idx = tid; idx < g_total; idx += blk) {
    const int k = idx % k_total;
    int rem = idx / k_total;
    const int o = rem % O;
    const int p = rem / O;
    const int s = static_cast<int>(seg_starts[p]);
    const int e = static_cast<int>(seg_ends[p]);
    scalar_t g = (k >= s && k < e)
                     ? grad_y[((b * P + p) * O + o) * k_total + k]
                     : scalar_t(0);
    G_s[idx] = g;
  }
  for (int i = tid; i < m2; i += blk) {
    b_s[i] = b_rhs[b * m2 + i];
  }
  const int u_count = m1 * m2 * k_total;
  for (int i = tid; i < u_count; i += blk) {
    U_s[i] = U[i];
  }
  __syncthreads();

  // Q[c, k] = sum_{p, o} G[p, o, k] * W[p, o, c]
  const int q_total = C * k_total;
  for (int idx = tid; idx < q_total; idx += blk) {
    const int k = idx % k_total;
    const int c = idx / k_total;
    scalar_t acc = scalar_t(0);
    for (int p = 0; p < P; ++p) {
      for (int o = 0; o < O; ++o) {
        acc += G_s[(p * O + o) * k_total + k] * W[(p * O + o) * C + c];
      }
    }
    Q_s[c * k_total + k] = acc;
  }

  // T[m, k] = sum_n b_rhs[n] * U[m*m2+n, k]
  const int t_total = m1 * k_total;
  for (int idx = tid; idx < t_total; idx += blk) {
    const int k = idx % k_total;
    const int m = idx / k_total;
    scalar_t acc = scalar_t(0);
    for (int n = 0; n < m2; ++n) {
      acc += b_s[n] * U_s[(m * m2 + n) * k_total + k];
    }
    T_s[m * k_total + k] = acc;
  }
  __syncthreads();

  // grad_a[b, c, m] = sum_k Q[c, k] * T[m, k]
  const int a_total = C * m1;
  for (int idx = tid; idx < a_total; idx += blk) {
    const int m = idx % m1;
    const int c = idx / m1;
    scalar_t acc = scalar_t(0);
    const scalar_t* Q_row = Q_s + c * k_total;
    const scalar_t* T_row = T_s + m * k_total;
    for (int k = 0; k < k_total; ++k) {
      acc += Q_row[k] * T_row[k];
    }
    grad_a[b * a_total + idx] = acc;
  }
}

// ---------------------------------------------------------------------------
// Backward w.r.t. b_rhs
// grad_b[b,n] = sum_{c,m,k} a[b,c,m] * U[mn,k] * Q[b,c,k]
//             = sum_{m,k} S[b,m,k] * U[mn,k]
// where S[b,m,k] = sum_c a[b,c,m] * Q[b,c,k], and Q as in grad_a kernel.
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void channelwise_group_transpose_b_kernel(
    const scalar_t* __restrict__ grad_y,      // (B, P, O, k_total)
    const scalar_t* __restrict__ a,           // (B, C, m1)
    const scalar_t* __restrict__ U,           // (m1*m2, k_total)
    const scalar_t* __restrict__ W,           // (P, O, C)
    const int64_t* __restrict__ seg_starts,   // (P,)
    const int64_t* __restrict__ seg_ends,     // (P,)
    scalar_t* __restrict__ grad_b,            // (B, m2)
    const int C,
    const int m1,
    const int m2,
    const int k_total,
    const int P,
    const int O) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  scalar_t* smem = reinterpret_cast<scalar_t*>(smem_raw);
  scalar_t* G_s = smem;                               // P*O*k_total
  scalar_t* a_s = G_s + P * O * k_total;              // C*m1
  scalar_t* U_s = a_s + C * m1;                       // m1*m2*k_total
  scalar_t* Q_s = U_s + m1 * m2 * k_total;            // C*k_total
  scalar_t* S_s = Q_s + C * k_total;                  // m1*k_total

  const int b = blockIdx.x;
  const int tid = threadIdx.x;
  const int blk = blockDim.x;

  // Load grad_y[b] masked
  const int g_total = P * O * k_total;
  for (int idx = tid; idx < g_total; idx += blk) {
    const int k = idx % k_total;
    int rem = idx / k_total;
    const int o = rem % O;
    const int p = rem / O;
    const int s = static_cast<int>(seg_starts[p]);
    const int e = static_cast<int>(seg_ends[p]);
    scalar_t g = (k >= s && k < e)
                     ? grad_y[((b * P + p) * O + o) * k_total + k]
                     : scalar_t(0);
    G_s[idx] = g;
  }
  const int a_count = C * m1;
  for (int i = tid; i < a_count; i += blk) {
    a_s[i] = a[b * a_count + i];
  }
  const int u_count = m1 * m2 * k_total;
  for (int i = tid; i < u_count; i += blk) {
    U_s[i] = U[i];
  }
  __syncthreads();

  // Q[c, k]
  const int q_total = C * k_total;
  for (int idx = tid; idx < q_total; idx += blk) {
    const int k = idx % k_total;
    const int c = idx / k_total;
    scalar_t acc = scalar_t(0);
    for (int p = 0; p < P; ++p) {
      for (int o = 0; o < O; ++o) {
        acc += G_s[(p * O + o) * k_total + k] * W[(p * O + o) * C + c];
      }
    }
    Q_s[c * k_total + k] = acc;
  }
  __syncthreads();

  // S[m, k] = sum_c a[c, m] * Q[c, k]
  const int s_total = m1 * k_total;
  for (int idx = tid; idx < s_total; idx += blk) {
    const int k = idx % k_total;
    const int m = idx / k_total;
    scalar_t acc = scalar_t(0);
    for (int c = 0; c < C; ++c) {
      acc += a_s[c * m1 + m] * Q_s[c * k_total + k];
    }
    S_s[m * k_total + k] = acc;
  }
  __syncthreads();

  // grad_b[b, n] = sum_{m, k} S[m, k] * U[m*m2+n, k]
  for (int n = tid; n < m2; n += blk) {
    scalar_t acc = scalar_t(0);
    for (int m = 0; m < m1; ++m) {
      const scalar_t* U_row = U_s + (m * m2 + n) * k_total;
      const scalar_t* S_row = S_s + m * k_total;
      for (int k = 0; k < k_total; ++k) {
        acc += S_row[k] * U_row[k];
      }
    }
    grad_b[b * m2 + n] = acc;
  }
}

// ---------------------------------------------------------------------------
// Backward w.r.t. U
// grad_U[mn,k] = sum_{b,c} a[b,c,m] * b_rhs[b,n] * X[b,c,k]
// where X[b,c,k] = sum_{p,o} G[b,p,o,k] * W[p,o,c] (effective G masked).
//
// One block per batch index b. Each block accumulates its local contribution
// to grad_U in shared memory, then atomicAdds that contribution into the
// global grad_U tensor.
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void channelwise_group_transpose_u_kernel(
    const scalar_t* __restrict__ grad_y,      // (B, P, O, k_total)
    const scalar_t* __restrict__ a,           // (B, C, m1)
    const scalar_t* __restrict__ b_rhs,       // (B, m2)
    const scalar_t* __restrict__ W,           // (P, O, C)
    const int64_t* __restrict__ seg_starts,   // (P,)
    const int64_t* __restrict__ seg_ends,     // (P,)
    scalar_t* __restrict__ grad_U,            // (m1*m2, k_total)
    const int C,
    const int m1,
    const int m2,
    const int k_total,
    const int P,
    const int O) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  scalar_t* smem = reinterpret_cast<scalar_t*>(smem_raw);
  scalar_t* G_s = smem;                               // P*O*k_total
  scalar_t* a_s = G_s + P * O * k_total;              // C*m1
  scalar_t* b_s = a_s + C * m1;                       // m2
  scalar_t* X_s = b_s + m2;                           // C*k_total
  scalar_t* gU_s = X_s + C * k_total;                 // m1*m2*k_total

  const int b = blockIdx.x;
  const int tid = threadIdx.x;
  const int blk = blockDim.x;

  // Load grad_y[b] masked
  const int g_total = P * O * k_total;
  for (int idx = tid; idx < g_total; idx += blk) {
    const int k = idx % k_total;
    int rem = idx / k_total;
    const int o = rem % O;
    const int p = rem / O;
    const int s = static_cast<int>(seg_starts[p]);
    const int e = static_cast<int>(seg_ends[p]);
    scalar_t g = (k >= s && k < e)
                     ? grad_y[((b * P + p) * O + o) * k_total + k]
                     : scalar_t(0);
    G_s[idx] = g;
  }
  const int a_count = C * m1;
  for (int i = tid; i < a_count; i += blk) {
    a_s[i] = a[b * a_count + i];
  }
  for (int i = tid; i < m2; i += blk) {
    b_s[i] = b_rhs[b * m2 + i];
  }
  __syncthreads();

  // X[c, k] = sum_{p, o} G[p, o, k] * W[p, o, c]
  const int x_total = C * k_total;
  for (int idx = tid; idx < x_total; idx += blk) {
    const int k = idx % k_total;
    const int c = idx / k_total;
    scalar_t acc = scalar_t(0);
    for (int p = 0; p < P; ++p) {
      for (int o = 0; o < O; ++o) {
        acc += G_s[(p * O + o) * k_total + k] * W[(p * O + o) * C + c];
      }
    }
    X_s[c * k_total + k] = acc;
  }
  __syncthreads();

  // Local grad_U contribution: gU_s[m*m2+n, k] = sum_c a[c,m] * b_rhs[n] * X[c, k]
  const int gu_total = m1 * m2 * k_total;
  for (int idx = tid; idx < gu_total; idx += blk) {
    const int k = idx % k_total;
    int rem = idx / k_total;
    const int n = rem % m2;
    const int m = rem / m2;
    const scalar_t bn = b_s[n];
    scalar_t acc = scalar_t(0);
    for (int c = 0; c < C; ++c) {
      acc += a_s[c * m1 + m] * bn * X_s[c * k_total + k];
    }
    gU_s[idx] = acc;
  }
  __syncthreads();

  // AtomicAdd local gU to global grad_U
  for (int idx = tid; idx < gu_total; idx += blk) {
    atomicAdd(grad_U + idx, gU_s[idx]);
  }
}

// ---------------------------------------------------------------------------
// Backward w.r.t. W
// grad_W[p,o,c] = sum_{b, k in [s_p, e_p)} G[b,p,o,k] * Z[b,c,k]
// where Z[b,c,k] = sum_{m,n} a[b,c,m] * b_rhs[b,n] * U[mn,k]  (same as forward).
//
// One block per batch b. Accumulate local contribution in smem, atomicAdd to
// global grad_W at the end.
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void channelwise_group_transpose_w_kernel(
    const scalar_t* __restrict__ grad_y,      // (B, P, O, k_total)
    const scalar_t* __restrict__ a,           // (B, C, m1)
    const scalar_t* __restrict__ b_rhs,       // (B, m2)
    const scalar_t* __restrict__ U,           // (m1*m2, k_total)
    const int64_t* __restrict__ seg_starts,   // (P,)
    const int64_t* __restrict__ seg_ends,     // (P,)
    scalar_t* __restrict__ grad_W,            // (P, O, C)
    const int C,
    const int m1,
    const int m2,
    const int k_total,
    const int P,
    const int O) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  scalar_t* smem = reinterpret_cast<scalar_t*>(smem_raw);
  scalar_t* G_s = smem;                               // P*O*k_total
  scalar_t* a_s = G_s + P * O * k_total;              // C*m1
  scalar_t* b_s = a_s + C * m1;                       // m2
  scalar_t* U_s = b_s + m2;                           // m1*m2*k_total
  scalar_t* Z_s = U_s + m1 * m2 * k_total;            // C*k_total

  const int b = blockIdx.x;
  const int tid = threadIdx.x;
  const int blk = blockDim.x;

  // Load grad_y[b] masked
  const int g_total = P * O * k_total;
  for (int idx = tid; idx < g_total; idx += blk) {
    const int k = idx % k_total;
    int rem = idx / k_total;
    const int o = rem % O;
    const int p = rem / O;
    const int s = static_cast<int>(seg_starts[p]);
    const int e = static_cast<int>(seg_ends[p]);
    scalar_t g = (k >= s && k < e)
                     ? grad_y[((b * P + p) * O + o) * k_total + k]
                     : scalar_t(0);
    G_s[idx] = g;
  }
  const int a_count = C * m1;
  for (int i = tid; i < a_count; i += blk) {
    a_s[i] = a[b * a_count + i];
  }
  for (int i = tid; i < m2; i += blk) {
    b_s[i] = b_rhs[b * m2 + i];
  }
  const int u_count = m1 * m2 * k_total;
  for (int i = tid; i < u_count; i += blk) {
    U_s[i] = U[i];
  }
  __syncthreads();

  // Z[c, k] = sum_{m, n} a[c, m] * b_rhs[n] * U[m*m2+n, k]
  const int z_total = C * k_total;
  for (int idx = tid; idx < z_total; idx += blk) {
    const int k = idx % k_total;
    const int c = idx / k_total;
    scalar_t acc = scalar_t(0);
    const scalar_t* a_c = a_s + c * m1;
    for (int m = 0; m < m1; ++m) {
      const scalar_t av = a_c[m];
      for (int n = 0; n < m2; ++n) {
        acc += av * b_s[n] * U_s[(m * m2 + n) * k_total + k];
      }
    }
    Z_s[c * k_total + k] = acc;
  }
  __syncthreads();

  // For each (p, o, c): contribution = sum_{k in seg_p} G[p,o,k] * Z[c, k]
  // AtomicAdd into global grad_W[p, o, c].
  const int poc_total = P * O * C;
  for (int idx = tid; idx < poc_total; idx += blk) {
    const int c = idx % C;
    int rem = idx / C;
    const int o = rem % O;
    const int p = rem / O;
    scalar_t acc = scalar_t(0);
    const scalar_t* G_po = G_s + (p * O + o) * k_total;
    const scalar_t* Z_c = Z_s + c * k_total;
    for (int k = 0; k < k_total; ++k) {
      acc += G_po[k] * Z_c[k];
    }
    atomicAdd(grad_W + idx, acc);
  }
}

// ---------------------------------------------------------------------------
// Launchers
// ---------------------------------------------------------------------------
template <typename scalar_t>
void launch_group_forward(
    const Tensor& a,
    const Tensor& b_rhs,
    const Tensor& U,
    const Tensor& W,
    const Tensor& seg_starts,
    const Tensor& seg_ends,
    Tensor& y) {
  const int B = static_cast<int>(a.size(0));
  const int C = static_cast<int>(a.size(1));
  const int m1 = static_cast<int>(a.size(2));
  const int m2 = static_cast<int>(b_rhs.size(1));
  const int k_total = static_cast<int>(U.size(1));
  const int P = static_cast<int>(W.size(0));
  const int O = static_cast<int>(W.size(1));
  const int threads = 256;
  const size_t smem =
      (C * m1 + m2 + m1 * m2 * k_total + C * k_total) * sizeof(scalar_t);
  channelwise_group_forward_kernel<scalar_t>
      <<<B, threads, smem, at::cuda::getCurrentCUDAStream()>>>(
          a.data_ptr<scalar_t>(),
          b_rhs.data_ptr<scalar_t>(),
          U.data_ptr<scalar_t>(),
          W.data_ptr<scalar_t>(),
          seg_starts.data_ptr<int64_t>(),
          seg_ends.data_ptr<int64_t>(),
          y.data_ptr<scalar_t>(),
          C, m1, m2, k_total, P, O);
}

template <typename scalar_t>
void launch_group_transpose_a(
    const Tensor& grad_y,
    const Tensor& b_rhs,
    const Tensor& U,
    const Tensor& W,
    const Tensor& seg_starts,
    const Tensor& seg_ends,
    Tensor& grad_a) {
  const int B = static_cast<int>(grad_y.size(0));
  const int P = static_cast<int>(grad_y.size(1));
  const int O = static_cast<int>(grad_y.size(2));
  const int k_total = static_cast<int>(grad_y.size(3));
  const int C = static_cast<int>(grad_a.size(1));
  const int m1 = static_cast<int>(grad_a.size(2));
  const int m2 = static_cast<int>(b_rhs.size(1));
  const int threads = 256;
  const size_t smem =
      (P * O * k_total + m2 + m1 * m2 * k_total + C * k_total + m1 * k_total) *
      sizeof(scalar_t);
  channelwise_group_transpose_a_kernel<scalar_t>
      <<<B, threads, smem, at::cuda::getCurrentCUDAStream()>>>(
          grad_y.data_ptr<scalar_t>(),
          b_rhs.data_ptr<scalar_t>(),
          U.data_ptr<scalar_t>(),
          W.data_ptr<scalar_t>(),
          seg_starts.data_ptr<int64_t>(),
          seg_ends.data_ptr<int64_t>(),
          grad_a.data_ptr<scalar_t>(),
          C, m1, m2, k_total, P, O);
}

template <typename scalar_t>
void launch_group_transpose_b(
    const Tensor& grad_y,
    const Tensor& a,
    const Tensor& U,
    const Tensor& W,
    const Tensor& seg_starts,
    const Tensor& seg_ends,
    Tensor& grad_b) {
  const int B = static_cast<int>(grad_y.size(0));
  const int P = static_cast<int>(grad_y.size(1));
  const int O = static_cast<int>(grad_y.size(2));
  const int k_total = static_cast<int>(grad_y.size(3));
  const int C = static_cast<int>(a.size(1));
  const int m1 = static_cast<int>(a.size(2));
  const int m2 = static_cast<int>(grad_b.size(1));
  const int threads = 256;
  const size_t smem =
      (P * O * k_total + C * m1 + m1 * m2 * k_total + C * k_total + m1 * k_total) *
      sizeof(scalar_t);
  channelwise_group_transpose_b_kernel<scalar_t>
      <<<B, threads, smem, at::cuda::getCurrentCUDAStream()>>>(
          grad_y.data_ptr<scalar_t>(),
          a.data_ptr<scalar_t>(),
          U.data_ptr<scalar_t>(),
          W.data_ptr<scalar_t>(),
          seg_starts.data_ptr<int64_t>(),
          seg_ends.data_ptr<int64_t>(),
          grad_b.data_ptr<scalar_t>(),
          C, m1, m2, k_total, P, O);
}

template <typename scalar_t>
void launch_group_transpose_u(
    const Tensor& grad_y,
    const Tensor& a,
    const Tensor& b_rhs,
    const Tensor& W,
    const Tensor& seg_starts,
    const Tensor& seg_ends,
    Tensor& grad_U) {
  const int B = static_cast<int>(grad_y.size(0));
  const int P = static_cast<int>(grad_y.size(1));
  const int O = static_cast<int>(grad_y.size(2));
  const int k_total = static_cast<int>(grad_y.size(3));
  const int C = static_cast<int>(a.size(1));
  const int m1 = static_cast<int>(a.size(2));
  const int m2 = static_cast<int>(b_rhs.size(1));
  const int threads = 256;
  const size_t smem =
      (P * O * k_total + C * m1 + m2 + C * k_total + m1 * m2 * k_total) *
      sizeof(scalar_t);
  channelwise_group_transpose_u_kernel<scalar_t>
      <<<B, threads, smem, at::cuda::getCurrentCUDAStream()>>>(
          grad_y.data_ptr<scalar_t>(),
          a.data_ptr<scalar_t>(),
          b_rhs.data_ptr<scalar_t>(),
          W.data_ptr<scalar_t>(),
          seg_starts.data_ptr<int64_t>(),
          seg_ends.data_ptr<int64_t>(),
          grad_U.data_ptr<scalar_t>(),
          C, m1, m2, k_total, P, O);
}

template <typename scalar_t>
void launch_group_transpose_w(
    const Tensor& grad_y,
    const Tensor& a,
    const Tensor& b_rhs,
    const Tensor& U,
    const Tensor& seg_starts,
    const Tensor& seg_ends,
    Tensor& grad_W) {
  const int B = static_cast<int>(grad_y.size(0));
  const int P = static_cast<int>(grad_y.size(1));
  const int O = static_cast<int>(grad_y.size(2));
  const int k_total = static_cast<int>(grad_y.size(3));
  const int C = static_cast<int>(a.size(1));
  const int m1 = static_cast<int>(a.size(2));
  const int m2 = static_cast<int>(b_rhs.size(1));
  const int threads = 256;
  const size_t smem =
      (P * O * k_total + C * m1 + m2 + m1 * m2 * k_total + C * k_total) *
      sizeof(scalar_t);
  channelwise_group_transpose_w_kernel<scalar_t>
      <<<B, threads, smem, at::cuda::getCurrentCUDAStream()>>>(
          grad_y.data_ptr<scalar_t>(),
          a.data_ptr<scalar_t>(),
          b_rhs.data_ptr<scalar_t>(),
          U.data_ptr<scalar_t>(),
          seg_starts.data_ptr<int64_t>(),
          seg_ends.data_ptr<int64_t>(),
          grad_W.data_ptr<scalar_t>(),
          C, m1, m2, k_total, P, O);
}

}  // namespace

// ---------------------------------------------------------------------------
// Tensor-facing top-level entry points
// ---------------------------------------------------------------------------
Tensor channelwise_group_forward_cuda(
    const Tensor& a,
    const Tensor& b_rhs,
    const Tensor& U,
    const Tensor& W,
    const Tensor& seg_starts,
    const Tensor& seg_ends) {
  TORCH_CHECK(a.is_cuda() && b_rhs.is_cuda() && U.is_cuda() && W.is_cuda(),
              "channelwise_group_forward_cuda expects CUDA tensors");
  TORCH_CHECK(seg_starts.is_cuda() && seg_ends.is_cuda(),
              "seg_starts/seg_ends must be on CUDA");
  TORCH_CHECK(a.scalar_type() == torch::kFloat32, "float32 only for now");
  TORCH_CHECK(a.dim() == 3, "a must be (B, C, m1)");
  TORCH_CHECK(b_rhs.dim() == 2, "b_rhs must be (B, m2)");
  TORCH_CHECK(U.dim() == 2, "U must be (m1*m2, k_total)");
  TORCH_CHECK(W.dim() == 3, "W must be (P, O, C)");
  TORCH_CHECK(seg_starts.dim() == 1 && seg_ends.dim() == 1, "segs must be 1D");
  TORCH_CHECK(seg_starts.scalar_type() == torch::kInt64, "seg_starts must be int64");
  TORCH_CHECK(seg_ends.scalar_type() == torch::kInt64, "seg_ends must be int64");
  const auto B = a.size(0);
  const auto P = W.size(0);
  const auto O = W.size(1);
  const auto k_total = U.size(1);
  TORCH_CHECK(b_rhs.size(0) == B, "b_rhs batch mismatch");
  TORCH_CHECK(W.size(2) == a.size(1), "W last dim must equal a channel dim");
  TORCH_CHECK(seg_starts.size(0) == P, "seg_starts must have P entries");
  TORCH_CHECK(seg_ends.size(0) == P, "seg_ends must have P entries");
  TORCH_CHECK(U.size(0) == a.size(2) * b_rhs.size(1),
              "U first dim must equal m1*m2");

  auto a_c = a.contiguous();
  auto b_c = b_rhs.contiguous();
  auto U_c = U.contiguous();
  auto W_c = W.contiguous();
  auto s_c = seg_starts.contiguous();
  auto e_c = seg_ends.contiguous();

  auto y = torch::empty({B, P, O, k_total}, a.options());
  const c10::cuda::CUDAGuard device_guard(a.device());
  launch_group_forward<float>(a_c, b_c, U_c, W_c, s_c, e_c, y);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

Tensor channelwise_group_transpose_a_cuda(
    const Tensor& grad_y,
    const Tensor& b_rhs,
    const Tensor& U,
    const Tensor& W,
    const Tensor& seg_starts,
    const Tensor& seg_ends,
    int64_t channel_mul,
    int64_t m1_size) {
  TORCH_CHECK(grad_y.is_cuda(), "grad_y must be CUDA");
  TORCH_CHECK(grad_y.scalar_type() == torch::kFloat32, "float32 only");
  const auto B = grad_y.size(0);
  auto grad_a = torch::empty({B, channel_mul, m1_size}, grad_y.options());
  auto g_c = grad_y.contiguous();
  auto b_c = b_rhs.contiguous();
  auto U_c = U.contiguous();
  auto W_c = W.contiguous();
  auto s_c = seg_starts.contiguous();
  auto e_c = seg_ends.contiguous();
  const c10::cuda::CUDAGuard device_guard(grad_y.device());
  launch_group_transpose_a<float>(g_c, b_c, U_c, W_c, s_c, e_c, grad_a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad_a;
}

Tensor channelwise_group_transpose_b_cuda(
    const Tensor& grad_y,
    const Tensor& a,
    const Tensor& U,
    const Tensor& W,
    const Tensor& seg_starts,
    const Tensor& seg_ends,
    int64_t m2_size) {
  TORCH_CHECK(grad_y.is_cuda(), "grad_y must be CUDA");
  TORCH_CHECK(grad_y.scalar_type() == torch::kFloat32, "float32 only");
  const auto B = grad_y.size(0);
  auto grad_b = torch::empty({B, m2_size}, grad_y.options());
  auto g_c = grad_y.contiguous();
  auto a_c = a.contiguous();
  auto U_c = U.contiguous();
  auto W_c = W.contiguous();
  auto s_c = seg_starts.contiguous();
  auto e_c = seg_ends.contiguous();
  const c10::cuda::CUDAGuard device_guard(grad_y.device());
  launch_group_transpose_b<float>(g_c, a_c, U_c, W_c, s_c, e_c, grad_b);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad_b;
}

Tensor channelwise_group_transpose_u_cuda(
    const Tensor& grad_y,
    const Tensor& a,
    const Tensor& b_rhs,
    const Tensor& W,
    const Tensor& seg_starts,
    const Tensor& seg_ends) {
  TORCH_CHECK(grad_y.is_cuda(), "grad_y must be CUDA");
  TORCH_CHECK(grad_y.scalar_type() == torch::kFloat32, "float32 only");
  const auto m1 = a.size(2);
  const auto m2 = b_rhs.size(1);
  const auto k_total = grad_y.size(3);
  auto grad_U = torch::zeros({m1 * m2, k_total}, grad_y.options());
  auto g_c = grad_y.contiguous();
  auto a_c = a.contiguous();
  auto b_c = b_rhs.contiguous();
  auto W_c = W.contiguous();
  auto s_c = seg_starts.contiguous();
  auto e_c = seg_ends.contiguous();
  const c10::cuda::CUDAGuard device_guard(grad_y.device());
  launch_group_transpose_u<float>(g_c, a_c, b_c, W_c, s_c, e_c, grad_U);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad_U;
}

Tensor channelwise_group_transpose_w_cuda(
    const Tensor& grad_y,
    const Tensor& a,
    const Tensor& b_rhs,
    const Tensor& U,
    const Tensor& seg_starts,
    const Tensor& seg_ends) {
  TORCH_CHECK(grad_y.is_cuda(), "grad_y must be CUDA");
  TORCH_CHECK(grad_y.scalar_type() == torch::kFloat32, "float32 only");
  const auto P = grad_y.size(1);
  const auto O = grad_y.size(2);
  const auto C = a.size(1);
  auto grad_W = torch::zeros({P, O, C}, grad_y.options());
  auto g_c = grad_y.contiguous();
  auto a_c = a.contiguous();
  auto b_c = b_rhs.contiguous();
  auto U_c = U.contiguous();
  auto s_c = seg_starts.contiguous();
  auto e_c = seg_ends.contiguous();
  const c10::cuda::CUDAGuard device_guard(grad_y.device());
  launch_group_transpose_w<float>(g_c, a_c, b_c, U_c, s_c, e_c, grad_W);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad_W;
}
