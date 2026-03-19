#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>

using torch::Tensor;

namespace {

template <typename scalar_t>
__global__ void project_forward_mul_in2eq1_kernel(
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    const scalar_t* __restrict__ u,
    scalar_t* __restrict__ y,
    const int64_t batch,
    const int64_t mul_in1,
    const int64_t m1,
    const int64_t m2,
    const int64_t pk) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = batch * pk * mul_in1;
  if (linear >= total) {
    return;
  }

  const int64_t i = linear % mul_in1;
  const int64_t tmp = linear / mul_in1;
  const int64_t pk_idx = tmp % pk;
  const int64_t b_idx = tmp / pk;

  scalar_t acc = scalar_t(0);
  for (int64_t mi = 0; mi < m1; ++mi) {
    const scalar_t a_val = a[(b_idx * mul_in1 + i) * m1 + mi];
    const int64_t u_base = (mi * m2) * pk + pk_idx;
    for (int64_t mj = 0; mj < m2; ++mj) {
      acc += a_val * b[b_idx * m2 + mj] * u[u_base + mj * pk];
    }
  }
  y[(b_idx * pk + pk_idx) * mul_in1 + i] = acc;
}

template <typename scalar_t>
__global__ void project_transpose_a_mul_in2eq1_kernel(
    const scalar_t* __restrict__ grad_y,
    const scalar_t* __restrict__ b,
    const scalar_t* __restrict__ u,
    scalar_t* __restrict__ grad_a,
    const int64_t batch,
    const int64_t mul_in1,
    const int64_t m1,
    const int64_t m2,
    const int64_t pk) {
  extern __shared__ char smem_raw_a[];
  scalar_t* u_smem = reinterpret_cast<scalar_t*>(smem_raw_a);

  const int64_t u_total = m1 * m2 * pk;
  for (int64_t idx = threadIdx.x; idx < u_total; idx += blockDim.x) {
    u_smem[idx] = u[idx];
  }
  __syncthreads();

  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = batch * mul_in1 * m1;
  if (linear >= total) {
    return;
  }

  const int64_t mi = linear % m1;
  const int64_t tmp = linear / m1;
  const int64_t i = tmp % mul_in1;
  const int64_t b_idx = tmp / mul_in1;

  scalar_t acc = scalar_t(0);
  for (int64_t pk_idx = 0; pk_idx < pk; ++pk_idx) {
    scalar_t inner = scalar_t(0);
    const int64_t u_base = (mi * m2) * pk + pk_idx;
    for (int64_t mj = 0; mj < m2; ++mj) {
      inner += b[b_idx * m2 + mj] * u_smem[u_base + mj * pk];
    }
    acc += grad_y[(b_idx * pk + pk_idx) * mul_in1 + i] * inner;
  }
  grad_a[(b_idx * mul_in1 + i) * m1 + mi] = acc;
}

template <typename scalar_t>
__global__ void project_transpose_b_mul_in2eq1_kernel(
    const scalar_t* __restrict__ grad_y,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ u,
    scalar_t* __restrict__ grad_b,
    const int64_t batch,
    const int64_t mul_in1,
    const int64_t m1,
    const int64_t m2,
    const int64_t pk) {
  extern __shared__ char smem_raw_b[];
  scalar_t* u_smem = reinterpret_cast<scalar_t*>(smem_raw_b);

  const int64_t u_total = m1 * m2 * pk;
  for (int64_t idx = threadIdx.x; idx < u_total; idx += blockDim.x) {
    u_smem[idx] = u[idx];
  }
  __syncthreads();

  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = batch * m2;
  if (linear >= total) {
    return;
  }

  const int64_t mj = linear % m2;
  const int64_t b_idx = linear / m2;

  scalar_t acc = scalar_t(0);
  for (int64_t pk_idx = 0; pk_idx < pk; ++pk_idx) {
    for (int64_t i = 0; i < mul_in1; ++i) {
      scalar_t inner = scalar_t(0);
      for (int64_t mi = 0; mi < m1; ++mi) {
        inner += a[(b_idx * mul_in1 + i) * m1 + mi] * u_smem[(mi * m2 + mj) * pk + pk_idx];
      }
      acc += grad_y[(b_idx * pk + pk_idx) * mul_in1 + i] * inner;
    }
  }
  grad_b[b_idx * m2 + mj] = acc;
}

template <typename scalar_t>
__global__ void project_transpose_u_mul_in2eq1_kernel(
    const scalar_t* __restrict__ grad_y,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ grad_u,
    const int64_t batch,
    const int64_t mul_in1,
    const int64_t m1,
    const int64_t m2,
    const int64_t pk) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = m1 * m2 * pk;
  if (linear >= total) return;

  const int64_t pk_idx = linear % pk;
  const int64_t tmp = linear / pk;
  const int64_t mj = tmp % m2;
  const int64_t mi = tmp / m2;

  scalar_t acc = scalar_t(0);
  for (int64_t b_idx = 0; b_idx < batch; ++b_idx) {
    const scalar_t b_val = b[b_idx * m2 + mj];
    for (int64_t i = 0; i < mul_in1; ++i) {
      acc += grad_y[(b_idx * pk + pk_idx) * mul_in1 + i] *
             a[(b_idx * mul_in1 + i) * m1 + mi] *
             b_val;
    }
  }
  grad_u[(mi * m2 + mj) * pk + pk_idx] = acc;
}

template <typename scalar_t>
void launch_project_forward_mul_in2eq1(
    const Tensor& a,
    const Tensor& b,
    const Tensor& u,
    Tensor& y) {
  const auto batch = a.size(0);
  const auto mul_in1 = a.size(1);
  const auto m1 = a.size(2);
  const auto m2 = b.size(1);
  const auto pk = u.size(1);
  const int threads = 256;
  const int64_t total = batch * pk * mul_in1;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  project_forward_mul_in2eq1_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
      a.data_ptr<scalar_t>(),
      b.data_ptr<scalar_t>(),
      u.data_ptr<scalar_t>(),
      y.data_ptr<scalar_t>(),
      batch,
      mul_in1,
      m1,
      m2,
      pk);
}

template <typename scalar_t>
void launch_project_transpose_a_mul_in2eq1(
    const Tensor& grad_y,
    const Tensor& b,
    const Tensor& u,
    Tensor& grad_a) {
  const auto batch = b.size(0);
  const auto mul_in1 = grad_a.size(1);
  const auto m1 = grad_a.size(2);
  const auto m2 = b.size(1);
  const auto pk = grad_y.size(1) * grad_y.size(2);
  const int threads = 256;
  const int64_t total = batch * mul_in1 * m1;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  const size_t smem_bytes = m1 * m2 * pk * sizeof(scalar_t);
  project_transpose_a_mul_in2eq1_kernel<scalar_t><<<blocks, threads, smem_bytes, at::cuda::getDefaultCUDAStream()>>>(
      grad_y.data_ptr<scalar_t>(),
      b.data_ptr<scalar_t>(),
      u.data_ptr<scalar_t>(),
      grad_a.data_ptr<scalar_t>(),
      batch,
      mul_in1,
      m1,
      m2,
      pk);
}

template <typename scalar_t>
void launch_project_transpose_b_mul_in2eq1(
    const Tensor& grad_y,
    const Tensor& a,
    const Tensor& u,
    Tensor& grad_b) {
  const auto batch = a.size(0);
  const auto mul_in1 = a.size(1);
  const auto m1 = a.size(2);
  const auto m2 = grad_b.size(2);
  const auto pk = grad_y.size(1) * grad_y.size(2);
  const int threads = 256;
  const int64_t total = batch * m2;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  const size_t smem_bytes = m1 * m2 * pk * sizeof(scalar_t);
  project_transpose_b_mul_in2eq1_kernel<scalar_t><<<blocks, threads, smem_bytes, at::cuda::getDefaultCUDAStream()>>>(
      grad_y.data_ptr<scalar_t>(),
      a.data_ptr<scalar_t>(),
      u.data_ptr<scalar_t>(),
      grad_b.data_ptr<scalar_t>(),
      batch,
      mul_in1,
      m1,
      m2,
      pk);
}

template <typename scalar_t>
void launch_project_transpose_u_mul_in2eq1(
    const Tensor& grad_y,
    const Tensor& a,
    const Tensor& b,
    Tensor& grad_u) {
  const auto batch = a.size(0);
  const auto mul_in1 = a.size(1);
  const auto m1 = a.size(2);
  const auto m2 = b.size(1);
  const auto pk = grad_y.size(1) * grad_y.size(2);
  const int threads = 256;
  const int64_t total = m1 * m2 * pk;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  project_transpose_u_mul_in2eq1_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
      grad_y.data_ptr<scalar_t>(),
      a.data_ptr<scalar_t>(),
      b.data_ptr<scalar_t>(),
      grad_u.data_ptr<scalar_t>(),
      batch,
      mul_in1,
      m1,
      m2,
      pk);
}

template <typename scalar_t>
__global__ void mix_forward_kernel(
    const scalar_t* __restrict__ y,
    const scalar_t* __restrict__ w,
    const scalar_t* __restrict__ gates,
    scalar_t* __restrict__ out,
    const int64_t batch,
    const int64_t num_paths,
    const int64_t kdim,
    const int64_t mul_out,
    const int64_t ij) {
  extern __shared__ char smem_raw_mix[];
  scalar_t* w_tile = reinterpret_cast<scalar_t*>(smem_raw_mix);

  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = batch * mul_out * kdim;

  int64_t k = 0, o = 0, b_idx = 0;
  if (linear < total) {
    k = linear % kdim;
    const int64_t tmp = linear / kdim;
    o = tmp % mul_out;
    b_idx = tmp / mul_out;
  }

  scalar_t acc = scalar_t(0);

  for (int64_t p = 0; p < num_paths; ++p) {
    const int64_t w_offset = p * mul_out * ij;
    for (int64_t idx = threadIdx.x; idx < mul_out * ij; idx += blockDim.x) {
      w_tile[idx] = w[w_offset + idx];
    }
    __syncthreads();

    if (linear < total) {
      const scalar_t gate = gates[b_idx * num_paths + p];
      const int64_t y_base = ((b_idx * num_paths + p) * kdim + k) * ij;
      scalar_t inner = scalar_t(0);
      for (int64_t q = 0; q < ij; ++q) {
        inner += y[y_base + q] * w_tile[o * ij + q];
      }
      acc += gate * inner;
    }
    __syncthreads();
  }

  if (linear < total) {
    out[(b_idx * mul_out + o) * kdim + k] = acc;
  }
}

template <typename scalar_t>
__global__ void mix_transpose_y_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ w,
    const scalar_t* __restrict__ gates,
    scalar_t* __restrict__ grad_y,
    const int64_t batch,
    const int64_t num_paths,
    const int64_t kdim,
    const int64_t mul_out,
    const int64_t ij) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = batch * num_paths * kdim * ij;
  if (linear >= total) {
    return;
  }

  const int64_t q = linear % ij;
  const int64_t tmp0 = linear / ij;
  const int64_t k = tmp0 % kdim;
  const int64_t tmp1 = tmp0 / kdim;
  const int64_t p = tmp1 % num_paths;
  const int64_t b_idx = tmp1 / num_paths;

  const scalar_t gate = gates[b_idx * num_paths + p];
  scalar_t acc = scalar_t(0);
  for (int64_t o = 0; o < mul_out; ++o) {
    acc += grad_out[(b_idx * mul_out + o) * kdim + k] * w[(p * mul_out + o) * ij + q];
  }
  grad_y[linear] = gate * acc;
}

template <typename scalar_t>
__global__ void mix_transpose_w_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ y,
    const scalar_t* __restrict__ gates,
    scalar_t* __restrict__ grad_w,
    const int64_t batch,
    const int64_t num_paths,
    const int64_t kdim,
    const int64_t mul_out,
    const int64_t ij) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = num_paths * mul_out * ij;
  if (linear >= total) {
    return;
  }

  const int64_t q = linear % ij;
  const int64_t tmp = linear / ij;
  const int64_t o = tmp % mul_out;
  const int64_t p = tmp / mul_out;

  scalar_t acc = scalar_t(0);
  for (int64_t b_idx = 0; b_idx < batch; ++b_idx) {
    const scalar_t gate = gates[b_idx * num_paths + p];
    for (int64_t k = 0; k < kdim; ++k) {
      acc += grad_out[(b_idx * mul_out + o) * kdim + k] *
             gate *
             y[((b_idx * num_paths + p) * kdim + k) * ij + q];
    }
  }
  grad_w[(p * mul_out + o) * ij + q] = acc;
}

template <typename scalar_t>
__global__ void mix_transpose_g_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ y,
    const scalar_t* __restrict__ w,
    scalar_t* __restrict__ grad_g,
    const int64_t batch,
    const int64_t num_paths,
    const int64_t kdim,
    const int64_t mul_out,
    const int64_t ij) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = batch * num_paths;
  if (linear >= total) {
    return;
  }

  const int64_t p = linear % num_paths;
  const int64_t b_idx = linear / num_paths;

  scalar_t acc = scalar_t(0);
  for (int64_t o = 0; o < mul_out; ++o) {
    for (int64_t k = 0; k < kdim; ++k) {
      const scalar_t go = grad_out[(b_idx * mul_out + o) * kdim + k];
      const int64_t y_base = ((b_idx * num_paths + p) * kdim + k) * ij;
      const int64_t w_base = (p * mul_out + o) * ij;
      scalar_t inner = scalar_t(0);
      for (int64_t q = 0; q < ij; ++q) {
        inner += y[y_base + q] * w[w_base + q];
      }
      acc += go * inner;
    }
  }
  grad_g[linear] = acc;
}

template <typename scalar_t>
void launch_mix_forward(
    const Tensor& y,
    const Tensor& w,
    const Tensor& gates,
    Tensor& out) {
  const auto batch = y.size(0);
  const auto num_paths = y.size(1);
  const auto kdim = y.size(2);
  const auto ij = y.size(3);
  const auto mul_out = w.size(1);
  const int threads = 256;
  const int64_t total = batch * mul_out * kdim;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  const size_t smem_bytes = mul_out * ij * sizeof(scalar_t);
  mix_forward_kernel<scalar_t><<<blocks, threads, smem_bytes, at::cuda::getDefaultCUDAStream()>>>(
      y.data_ptr<scalar_t>(),
      w.data_ptr<scalar_t>(),
      gates.data_ptr<scalar_t>(),
      out.data_ptr<scalar_t>(),
      batch,
      num_paths,
      kdim,
      mul_out,
      ij);
}

template <typename scalar_t>
void launch_mix_transpose_y(
    const Tensor& grad_out,
    const Tensor& w,
    const Tensor& gates,
    Tensor& grad_y) {
  const auto batch = grad_y.size(0);
  const auto num_paths = grad_y.size(1);
  const auto kdim = grad_y.size(2);
  const auto ij = grad_y.size(3);
  const auto mul_out = grad_out.size(1);
  const int threads = 256;
  const int64_t total = batch * num_paths * kdim * ij;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  mix_transpose_y_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
      grad_out.data_ptr<scalar_t>(),
      w.data_ptr<scalar_t>(),
      gates.data_ptr<scalar_t>(),
      grad_y.data_ptr<scalar_t>(),
      batch,
      num_paths,
      kdim,
      mul_out,
      ij);
}

template <typename scalar_t>
void launch_mix_transpose_w(
    const Tensor& grad_out,
    const Tensor& y,
    const Tensor& gates,
    Tensor& grad_w) {
  const auto batch = y.size(0);
  const auto num_paths = y.size(1);
  const auto kdim = y.size(2);
  const auto ij = y.size(3);
  const auto mul_out = grad_out.size(1);
  const int threads = 256;
  const int64_t total = num_paths * mul_out * ij;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  mix_transpose_w_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
      grad_out.data_ptr<scalar_t>(),
      y.data_ptr<scalar_t>(),
      gates.data_ptr<scalar_t>(),
      grad_w.data_ptr<scalar_t>(),
      batch,
      num_paths,
      kdim,
      mul_out,
      ij);
}

template <typename scalar_t>
void launch_mix_transpose_g(
    const Tensor& grad_out,
    const Tensor& y,
    const Tensor& w,
    Tensor& grad_g) {
  const auto batch = y.size(0);
  const auto num_paths = y.size(1);
  const auto kdim = y.size(2);
  const auto ij = y.size(3);
  const auto mul_out = grad_out.size(1);
  const int threads = 256;
  const int64_t total = batch * num_paths;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  mix_transpose_g_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
      grad_out.data_ptr<scalar_t>(),
      y.data_ptr<scalar_t>(),
      w.data_ptr<scalar_t>(),
      grad_g.data_ptr<scalar_t>(),
      batch,
      num_paths,
      kdim,
      mul_out,
      ij);
}

// ---------------------------------------------------------------------------
// Project double-backward fused kernels
// grad_H  = PF(GGA,b,U) + PF(a,GGB,U) + PF(a,b,GGU)   [3-term, PF shape]
// grad_a2 = PAT(H,GGB,U) + PAT(H,b,GGU)                [2-term, PAT shape]
// grad_b2 = PBT(H,GGA,U) + PBT(H,a,GGU)                [2-term, PBT shape]
// grad_U2 = PUT(H,GGA,b) + PUT(H,a,GGB)                 [2-term, PUT shape]
// ---------------------------------------------------------------------------

template <typename scalar_t>
__global__ void project_dbl_bwd_grad_h_kernel(
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ gga,
    const scalar_t* __restrict__ b,
    const scalar_t* __restrict__ ggb,
    const scalar_t* __restrict__ u,
    const scalar_t* __restrict__ ggu,
    scalar_t* __restrict__ grad_h,
    const int64_t batch,
    const int64_t mul_in1,
    const int64_t m1,
    const int64_t m2,
    const int64_t pk) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = batch * pk * mul_in1;
  if (linear >= total) return;

  const int64_t i = linear % mul_in1;
  const int64_t tmp = linear / mul_in1;
  const int64_t pk_idx = tmp % pk;
  const int64_t b_idx = tmp / pk;

  scalar_t acc = scalar_t(0);
  for (int64_t mi = 0; mi < m1; ++mi) {
    const scalar_t a_val = a[(b_idx * mul_in1 + i) * m1 + mi];
    const scalar_t gga_val = gga[(b_idx * mul_in1 + i) * m1 + mi];
    const int64_t u_base = (mi * m2) * pk + pk_idx;
    for (int64_t mj = 0; mj < m2; ++mj) {
      const scalar_t b_val = b[b_idx * m2 + mj];
      const scalar_t ggb_val = ggb[b_idx * m2 + mj];
      const scalar_t u_val = u[u_base + mj * pk];
      const scalar_t ggu_val = ggu[u_base + mj * pk];
      acc += gga_val * b_val * u_val
           + a_val * ggb_val * u_val
           + a_val * b_val * ggu_val;
    }
  }
  grad_h[(b_idx * pk + pk_idx) * mul_in1 + i] = acc;
}

template <typename scalar_t>
__global__ void project_dbl_bwd_grad_a_kernel(
    const scalar_t* __restrict__ h,
    const scalar_t* __restrict__ b,
    const scalar_t* __restrict__ ggb,
    const scalar_t* __restrict__ u,
    const scalar_t* __restrict__ ggu,
    scalar_t* __restrict__ grad_a,
    const int64_t batch,
    const int64_t mul_in1,
    const int64_t m1,
    const int64_t m2,
    const int64_t pk) {
  extern __shared__ char smem_dbl_a[];
  scalar_t* u_smem = reinterpret_cast<scalar_t*>(smem_dbl_a);
  scalar_t* ggu_smem = u_smem + m1 * m2 * pk;

  const int64_t u_total = m1 * m2 * pk;
  for (int64_t idx = threadIdx.x; idx < u_total; idx += blockDim.x) {
    u_smem[idx] = u[idx];
    ggu_smem[idx] = ggu[idx];
  }
  __syncthreads();

  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = batch * mul_in1 * m1;
  if (linear >= total) return;

  const int64_t mi = linear % m1;
  const int64_t tmp = linear / m1;
  const int64_t i = tmp % mul_in1;
  const int64_t b_idx = tmp / mul_in1;

  scalar_t acc = scalar_t(0);
  for (int64_t pk_idx = 0; pk_idx < pk; ++pk_idx) {
    scalar_t inner = scalar_t(0);
    const int64_t u_base = (mi * m2) * pk + pk_idx;
    for (int64_t mj = 0; mj < m2; ++mj) {
      inner += ggb[b_idx * m2 + mj] * u_smem[u_base + mj * pk]
             + b[b_idx * m2 + mj] * ggu_smem[u_base + mj * pk];
    }
    acc += h[(b_idx * pk + pk_idx) * mul_in1 + i] * inner;
  }
  grad_a[(b_idx * mul_in1 + i) * m1 + mi] = acc;
}

template <typename scalar_t>
__global__ void project_dbl_bwd_grad_b_kernel(
    const scalar_t* __restrict__ h,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ gga,
    const scalar_t* __restrict__ u,
    const scalar_t* __restrict__ ggu,
    scalar_t* __restrict__ grad_b,
    const int64_t batch,
    const int64_t mul_in1,
    const int64_t m1,
    const int64_t m2,
    const int64_t pk) {
  extern __shared__ char smem_dbl_b[];
  scalar_t* u_smem = reinterpret_cast<scalar_t*>(smem_dbl_b);
  scalar_t* ggu_smem = u_smem + m1 * m2 * pk;

  const int64_t u_total = m1 * m2 * pk;
  for (int64_t idx = threadIdx.x; idx < u_total; idx += blockDim.x) {
    u_smem[idx] = u[idx];
    ggu_smem[idx] = ggu[idx];
  }
  __syncthreads();

  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = batch * m2;
  if (linear >= total) return;

  const int64_t mj = linear % m2;
  const int64_t b_idx = linear / m2;

  scalar_t acc = scalar_t(0);
  for (int64_t pk_idx = 0; pk_idx < pk; ++pk_idx) {
    for (int64_t i = 0; i < mul_in1; ++i) {
      scalar_t inner = scalar_t(0);
      for (int64_t mi = 0; mi < m1; ++mi) {
        inner += gga[(b_idx * mul_in1 + i) * m1 + mi] * u_smem[(mi * m2 + mj) * pk + pk_idx]
               + a[(b_idx * mul_in1 + i) * m1 + mi] * ggu_smem[(mi * m2 + mj) * pk + pk_idx];
      }
      acc += h[(b_idx * pk + pk_idx) * mul_in1 + i] * inner;
    }
  }
  grad_b[b_idx * m2 + mj] = acc;
}

template <typename scalar_t>
__global__ void project_dbl_bwd_grad_u_kernel(
    const scalar_t* __restrict__ h,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ gga,
    const scalar_t* __restrict__ b,
    const scalar_t* __restrict__ ggb,
    scalar_t* __restrict__ grad_u,
    const int64_t batch,
    const int64_t mul_in1,
    const int64_t m1,
    const int64_t m2,
    const int64_t pk) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = m1 * m2 * pk;
  if (linear >= total) return;

  const int64_t pk_idx = linear % pk;
  const int64_t tmp = linear / pk;
  const int64_t mj = tmp % m2;
  const int64_t mi = tmp / m2;

  scalar_t acc = scalar_t(0);
  for (int64_t b_idx = 0; b_idx < batch; ++b_idx) {
    const scalar_t b_val = b[b_idx * m2 + mj];
    const scalar_t ggb_val = ggb[b_idx * m2 + mj];
    for (int64_t i = 0; i < mul_in1; ++i) {
      const scalar_t gy_val = h[(b_idx * pk + pk_idx) * mul_in1 + i];
      acc += gy_val * (gga[(b_idx * mul_in1 + i) * m1 + mi] * b_val
                     + a[(b_idx * mul_in1 + i) * m1 + mi] * ggb_val);
    }
  }

  if (linear < total) {
    grad_u[(mi * m2 + mj) * pk + pk_idx] = acc;
  }
}

template <typename scalar_t>
void launch_project_dbl_bwd_grad_h(
    const Tensor& a, const Tensor& gga,
    const Tensor& b, const Tensor& ggb,
    const Tensor& u, const Tensor& ggu,
    Tensor& out) {
  const auto batch = a.size(0);
  const auto mul_in1 = a.size(1);
  const auto m1 = a.size(2);
  const auto m2 = b.size(1);
  const auto pk = u.size(1);
  const int threads = 256;
  const int64_t total = batch * pk * mul_in1;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  project_dbl_bwd_grad_h_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
      a.data_ptr<scalar_t>(), gga.data_ptr<scalar_t>(),
      b.data_ptr<scalar_t>(), ggb.data_ptr<scalar_t>(),
      u.data_ptr<scalar_t>(), ggu.data_ptr<scalar_t>(),
      out.data_ptr<scalar_t>(),
      batch, mul_in1, m1, m2, pk);
}

template <typename scalar_t>
void launch_project_dbl_bwd_grad_a(
    const Tensor& h,
    const Tensor& b, const Tensor& ggb,
    const Tensor& u, const Tensor& ggu,
    Tensor& out) {
  const auto batch = b.size(0);
  const auto mul_in1 = out.size(1);
  const auto m1 = out.size(2);
  const auto m2 = b.size(1);
  const auto pk = h.size(1) * h.size(2);
  const int threads = 256;
  const int64_t total = batch * mul_in1 * m1;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  const size_t smem_bytes = 2 * m1 * m2 * pk * sizeof(scalar_t);
  project_dbl_bwd_grad_a_kernel<scalar_t><<<blocks, threads, smem_bytes, at::cuda::getDefaultCUDAStream()>>>(
      h.data_ptr<scalar_t>(),
      b.data_ptr<scalar_t>(), ggb.data_ptr<scalar_t>(),
      u.data_ptr<scalar_t>(), ggu.data_ptr<scalar_t>(),
      out.data_ptr<scalar_t>(),
      batch, mul_in1, m1, m2, pk);
}

template <typename scalar_t>
void launch_project_dbl_bwd_grad_b(
    const Tensor& h,
    const Tensor& a, const Tensor& gga,
    const Tensor& u, const Tensor& ggu,
    Tensor& out) {
  const auto batch = a.size(0);
  const auto mul_in1 = a.size(1);
  const auto m1 = a.size(2);
  const auto m2 = out.size(1);
  const auto pk = h.size(1) * h.size(2);
  const int threads = 256;
  const int64_t total = batch * m2;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  const size_t smem_bytes = 2 * m1 * m2 * pk * sizeof(scalar_t);
  project_dbl_bwd_grad_b_kernel<scalar_t><<<blocks, threads, smem_bytes, at::cuda::getDefaultCUDAStream()>>>(
      h.data_ptr<scalar_t>(),
      a.data_ptr<scalar_t>(), gga.data_ptr<scalar_t>(),
      u.data_ptr<scalar_t>(), ggu.data_ptr<scalar_t>(),
      out.data_ptr<scalar_t>(),
      batch, mul_in1, m1, m2, pk);
}

template <typename scalar_t>
void launch_project_dbl_bwd_grad_u(
    const Tensor& h,
    const Tensor& a, const Tensor& gga,
    const Tensor& b, const Tensor& ggb,
    Tensor& out) {
  const auto batch = a.size(0);
  const auto mul_in1 = a.size(1);
  const auto m1 = a.size(2);
  const auto m2 = b.size(1);
  const auto pk = h.size(1) * h.size(2);
  const int threads = 256;
  const int64_t total = m1 * m2 * pk;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  project_dbl_bwd_grad_u_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
      h.data_ptr<scalar_t>(),
      a.data_ptr<scalar_t>(), gga.data_ptr<scalar_t>(),
      b.data_ptr<scalar_t>(), ggb.data_ptr<scalar_t>(),
      out.data_ptr<scalar_t>(),
      batch, mul_in1, m1, m2, pk);
}

// ---------------------------------------------------------------------------
// Mix double-backward fused kernels
// grad_G  = MF(GGY,W,g) + MF(Y,GGW,g) + MF(Y,W,GGg)   [3-term, MF shape]
// grad_Y2 = MYT(G,GGW,g) + MYT(G,W,GGg)                [2-term, MYT shape]
// grad_W2 = MWT(G,GGY,g) + MWT(G,Y,GGg)                 [2-term, MWT shape]
// grad_g2 = MGT(G,GGY,W) + MGT(G,Y,GGW)                 [2-term, MGT shape]
// ---------------------------------------------------------------------------

template <typename scalar_t>
__global__ void mix_dbl_bwd_grad_g_out_kernel(
    const scalar_t* __restrict__ y,
    const scalar_t* __restrict__ ggy,
    const scalar_t* __restrict__ w,
    const scalar_t* __restrict__ ggw,
    const scalar_t* __restrict__ gates,
    const scalar_t* __restrict__ ggg,
    scalar_t* __restrict__ grad_g_out,
    const int64_t batch,
    const int64_t num_paths,
    const int64_t kdim,
    const int64_t mul_out,
    const int64_t ij) {
  extern __shared__ char smem_dbl_mg[];
  scalar_t* w_tile = reinterpret_cast<scalar_t*>(smem_dbl_mg);
  scalar_t* ggw_tile = w_tile + mul_out * ij;

  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = batch * mul_out * kdim;

  int64_t k = 0, o = 0, b_idx = 0;
  if (linear < total) {
    k = linear % kdim;
    const int64_t tmp = linear / kdim;
    o = tmp % mul_out;
    b_idx = tmp / mul_out;
  }

  scalar_t acc = scalar_t(0);

  for (int64_t p = 0; p < num_paths; ++p) {
    const int64_t w_offset = p * mul_out * ij;
    for (int64_t idx = threadIdx.x; idx < mul_out * ij; idx += blockDim.x) {
      w_tile[idx] = w[w_offset + idx];
      ggw_tile[idx] = ggw[w_offset + idx];
    }
    __syncthreads();

    if (linear < total) {
      const scalar_t gate = gates[b_idx * num_paths + p];
      const scalar_t ggate = ggg[b_idx * num_paths + p];
      const int64_t y_base = ((b_idx * num_paths + p) * kdim + k) * ij;
      scalar_t inner = scalar_t(0);
      for (int64_t q = 0; q < ij; ++q) {
        const scalar_t y_val = y[y_base + q];
        const scalar_t ggy_val = ggy[y_base + q];
        const scalar_t w_val = w_tile[o * ij + q];
        const scalar_t ggw_val = ggw_tile[o * ij + q];
        inner += gate * (ggy_val * w_val + y_val * ggw_val)
               + ggate * y_val * w_val;
      }
      acc += inner;
    }
    __syncthreads();
  }

  if (linear < total) {
    grad_g_out[(b_idx * mul_out + o) * kdim + k] = acc;
  }
}

template <typename scalar_t>
__global__ void mix_dbl_bwd_grad_y_kernel(
    const scalar_t* __restrict__ g_out,
    const scalar_t* __restrict__ w,
    const scalar_t* __restrict__ ggw,
    const scalar_t* __restrict__ gates,
    const scalar_t* __restrict__ ggg,
    scalar_t* __restrict__ grad_y,
    const int64_t batch,
    const int64_t num_paths,
    const int64_t kdim,
    const int64_t mul_out,
    const int64_t ij) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = batch * num_paths * kdim * ij;
  if (linear >= total) return;

  const int64_t q = linear % ij;
  const int64_t tmp0 = linear / ij;
  const int64_t k = tmp0 % kdim;
  const int64_t tmp1 = tmp0 / kdim;
  const int64_t p = tmp1 % num_paths;
  const int64_t b_idx = tmp1 / num_paths;

  const scalar_t gate = gates[b_idx * num_paths + p];
  const scalar_t ggate = ggg[b_idx * num_paths + p];
  scalar_t acc = scalar_t(0);
  for (int64_t o = 0; o < mul_out; ++o) {
    const scalar_t go = g_out[(b_idx * mul_out + o) * kdim + k];
    acc += go * (gate * ggw[(p * mul_out + o) * ij + q]
               + ggate * w[(p * mul_out + o) * ij + q]);
  }
  grad_y[linear] = acc;
}

template <typename scalar_t>
__global__ void mix_dbl_bwd_grad_w_kernel(
    const scalar_t* __restrict__ g_out,
    const scalar_t* __restrict__ y,
    const scalar_t* __restrict__ ggy,
    const scalar_t* __restrict__ gates,
    const scalar_t* __restrict__ ggg,
    scalar_t* __restrict__ grad_w,
    const int64_t batch,
    const int64_t num_paths,
    const int64_t kdim,
    const int64_t mul_out,
    const int64_t ij) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = num_paths * mul_out * ij;
  if (linear >= total) return;

  const int64_t q = linear % ij;
  const int64_t tmp = linear / ij;
  const int64_t o = tmp % mul_out;
  const int64_t p = tmp / mul_out;

  scalar_t acc = scalar_t(0);
  for (int64_t b_idx = 0; b_idx < batch; ++b_idx) {
    const scalar_t gate = gates[b_idx * num_paths + p];
    const scalar_t ggate = ggg[b_idx * num_paths + p];
    for (int64_t k = 0; k < kdim; ++k) {
      const scalar_t go = g_out[(b_idx * mul_out + o) * kdim + k];
      acc += go * (gate * ggy[((b_idx * num_paths + p) * kdim + k) * ij + q]
                 + ggate * y[((b_idx * num_paths + p) * kdim + k) * ij + q]);
    }
  }
  grad_w[(p * mul_out + o) * ij + q] = acc;
}

template <typename scalar_t>
__global__ void mix_dbl_bwd_grad_g_kernel(
    const scalar_t* __restrict__ g_out,
    const scalar_t* __restrict__ y,
    const scalar_t* __restrict__ ggy,
    const scalar_t* __restrict__ w,
    const scalar_t* __restrict__ ggw,
    scalar_t* __restrict__ grad_g,
    const int64_t batch,
    const int64_t num_paths,
    const int64_t kdim,
    const int64_t mul_out,
    const int64_t ij) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = batch * num_paths;
  if (linear >= total) return;

  const int64_t p = linear % num_paths;
  const int64_t b_idx = linear / num_paths;

  scalar_t acc = scalar_t(0);
  for (int64_t o = 0; o < mul_out; ++o) {
    for (int64_t k = 0; k < kdim; ++k) {
      const scalar_t go = g_out[(b_idx * mul_out + o) * kdim + k];
      const int64_t y_base = ((b_idx * num_paths + p) * kdim + k) * ij;
      const int64_t w_base = (p * mul_out + o) * ij;
      scalar_t inner = scalar_t(0);
      for (int64_t q = 0; q < ij; ++q) {
        inner += ggy[y_base + q] * w[w_base + q]
               + y[y_base + q] * ggw[w_base + q];
      }
      acc += go * inner;
    }
  }
  grad_g[linear] = acc;
}

template <typename scalar_t>
void launch_mix_dbl_bwd_grad_g_out(
    const Tensor& y, const Tensor& ggy,
    const Tensor& w, const Tensor& ggw,
    const Tensor& gates, const Tensor& ggg,
    Tensor& out) {
  const auto batch = y.size(0);
  const auto num_paths = y.size(1);
  const auto kdim = y.size(2);
  const auto ij = y.size(3);
  const auto mul_out = w.size(1);
  const int threads = 256;
  const int64_t total = batch * mul_out * kdim;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  const size_t smem_bytes = 2 * mul_out * ij * sizeof(scalar_t);
  mix_dbl_bwd_grad_g_out_kernel<scalar_t><<<blocks, threads, smem_bytes, at::cuda::getDefaultCUDAStream()>>>(
      y.data_ptr<scalar_t>(), ggy.data_ptr<scalar_t>(),
      w.data_ptr<scalar_t>(), ggw.data_ptr<scalar_t>(),
      gates.data_ptr<scalar_t>(), ggg.data_ptr<scalar_t>(),
      out.data_ptr<scalar_t>(),
      batch, num_paths, kdim, mul_out, ij);
}

template <typename scalar_t>
void launch_mix_dbl_bwd_grad_y(
    const Tensor& g_out,
    const Tensor& w, const Tensor& ggw,
    const Tensor& gates, const Tensor& ggg,
    Tensor& out) {
  const auto batch = out.size(0);
  const auto num_paths = out.size(1);
  const auto kdim = out.size(2);
  const auto ij = out.size(3);
  const auto mul_out = g_out.size(1);
  const int threads = 256;
  const int64_t total = batch * num_paths * kdim * ij;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  mix_dbl_bwd_grad_y_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
      g_out.data_ptr<scalar_t>(),
      w.data_ptr<scalar_t>(), ggw.data_ptr<scalar_t>(),
      gates.data_ptr<scalar_t>(), ggg.data_ptr<scalar_t>(),
      out.data_ptr<scalar_t>(),
      batch, num_paths, kdim, mul_out, ij);
}

template <typename scalar_t>
void launch_mix_dbl_bwd_grad_w(
    const Tensor& g_out,
    const Tensor& y, const Tensor& ggy,
    const Tensor& gates, const Tensor& ggg,
    Tensor& out) {
  const auto batch = y.size(0);
  const auto num_paths = y.size(1);
  const auto kdim = y.size(2);
  const auto ij = y.size(3);
  const auto mul_out = g_out.size(1);
  const int threads = 256;
  const int64_t total = num_paths * mul_out * ij;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  mix_dbl_bwd_grad_w_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
      g_out.data_ptr<scalar_t>(),
      y.data_ptr<scalar_t>(), ggy.data_ptr<scalar_t>(),
      gates.data_ptr<scalar_t>(), ggg.data_ptr<scalar_t>(),
      out.data_ptr<scalar_t>(),
      batch, num_paths, kdim, mul_out, ij);
}

template <typename scalar_t>
void launch_mix_dbl_bwd_grad_g(
    const Tensor& g_out,
    const Tensor& y, const Tensor& ggy,
    const Tensor& w, const Tensor& ggw,
    Tensor& out) {
  const auto batch = y.size(0);
  const auto num_paths = y.size(1);
  const auto kdim = y.size(2);
  const auto ij = y.size(3);
  const auto mul_out = g_out.size(1);
  const int threads = 256;
  const int64_t total = batch * num_paths;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  mix_dbl_bwd_grad_g_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
      g_out.data_ptr<scalar_t>(),
      y.data_ptr<scalar_t>(), ggy.data_ptr<scalar_t>(),
      w.data_ptr<scalar_t>(), ggw.data_ptr<scalar_t>(),
      out.data_ptr<scalar_t>(),
      batch, num_paths, kdim, mul_out, ij);
}

}  // namespace

// ---------------------------------------------------------------------------
// Project double-backward C++ entry points
// ---------------------------------------------------------------------------

Tensor project_dbl_bwd_grad_h_cuda(
    const Tensor& a, const Tensor& gga,
    const Tensor& b, const Tensor& ggb,
    const Tensor& u, const Tensor& ggu,
    int64_t num_paths) {
  const auto kdim = u.size(1) / num_paths;
  auto out = torch::zeros({a.size(0), num_paths, kdim, a.size(1)}, a.options());
  const c10::cuda::CUDAGuard device_guard(a.device());
  auto a_c = a.contiguous(); auto gga_c = gga.contiguous();
  auto b_c = b.contiguous().view({b.size(0), b.size(2)});
  auto ggb_c = ggb.contiguous().view({ggb.size(0), ggb.size(2)});
  auto u_c = u.contiguous(); auto ggu_c = ggu.contiguous();
  if (a.scalar_type() == torch::kFloat32) {
    launch_project_dbl_bwd_grad_h<float>(a_c, gga_c, b_c, ggb_c, u_c, ggu_c, out);
  } else {
    launch_project_dbl_bwd_grad_h<double>(a_c, gga_c, b_c, ggb_c, u_c, ggu_c, out);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

Tensor project_dbl_bwd_grad_a_cuda(
    const Tensor& h,
    const Tensor& b, const Tensor& ggb,
    const Tensor& u, const Tensor& ggu) {
  auto grad_a = torch::zeros({b.size(0), h.size(3), u.size(0) / b.size(2)}, h.options());
  const c10::cuda::CUDAGuard device_guard(h.device());
  auto h_c = h.contiguous();
  auto b_c = b.contiguous().view({b.size(0), b.size(2)});
  auto ggb_c = ggb.contiguous().view({ggb.size(0), ggb.size(2)});
  auto u_c = u.contiguous(); auto ggu_c = ggu.contiguous();
  if (h.scalar_type() == torch::kFloat32) {
    launch_project_dbl_bwd_grad_a<float>(h_c, b_c, ggb_c, u_c, ggu_c, grad_a);
  } else {
    launch_project_dbl_bwd_grad_a<double>(h_c, b_c, ggb_c, u_c, ggu_c, grad_a);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad_a;
}

Tensor project_dbl_bwd_grad_b_cuda(
    const Tensor& h,
    const Tensor& a, const Tensor& gga,
    const Tensor& u, const Tensor& ggu) {
  auto grad_b = torch::zeros({a.size(0), 1, u.size(0) / a.size(2)}, h.options());
  const c10::cuda::CUDAGuard device_guard(h.device());
  auto h_c = h.contiguous(); auto a_c = a.contiguous(); auto gga_c = gga.contiguous();
  auto u_c = u.contiguous(); auto ggu_c = ggu.contiguous();
  if (h.scalar_type() == torch::kFloat32) {
    launch_project_dbl_bwd_grad_b<float>(h_c, a_c, gga_c, u_c, ggu_c, grad_b);
  } else {
    launch_project_dbl_bwd_grad_b<double>(h_c, a_c, gga_c, u_c, ggu_c, grad_b);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad_b;
}

Tensor project_dbl_bwd_grad_u_cuda(
    const Tensor& h,
    const Tensor& a, const Tensor& gga,
    const Tensor& b, const Tensor& ggb) {
  auto grad_u = torch::zeros({a.size(2) * b.size(2), h.size(1) * h.size(2)}, h.options());
  const c10::cuda::CUDAGuard device_guard(h.device());
  auto h_c = h.contiguous(); auto a_c = a.contiguous(); auto gga_c = gga.contiguous();
  auto b_c = b.contiguous().view({b.size(0), b.size(2)});
  auto ggb_c = ggb.contiguous().view({ggb.size(0), ggb.size(2)});
  if (h.scalar_type() == torch::kFloat32) {
    launch_project_dbl_bwd_grad_u<float>(h_c, a_c, gga_c, b_c, ggb_c, grad_u);
  } else {
    launch_project_dbl_bwd_grad_u<double>(h_c, a_c, gga_c, b_c, ggb_c, grad_u);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad_u;
}

// ---------------------------------------------------------------------------
// Mix double-backward C++ entry points
// ---------------------------------------------------------------------------

Tensor mix_dbl_bwd_grad_g_out_cuda(
    const Tensor& y, const Tensor& ggy,
    const Tensor& w, const Tensor& ggw,
    const Tensor& gates, const Tensor& ggg) {
  auto out = torch::zeros({y.size(0), w.size(1), y.size(2)}, y.options());
  const c10::cuda::CUDAGuard device_guard(y.device());
  auto y_c = y.contiguous(); auto ggy_c = ggy.contiguous();
  auto w_c = w.contiguous(); auto ggw_c = ggw.contiguous();
  auto g_c = gates.contiguous(); auto ggg_c = ggg.contiguous();
  if (y.scalar_type() == torch::kFloat32) {
    launch_mix_dbl_bwd_grad_g_out<float>(y_c, ggy_c, w_c, ggw_c, g_c, ggg_c, out);
  } else {
    launch_mix_dbl_bwd_grad_g_out<double>(y_c, ggy_c, w_c, ggw_c, g_c, ggg_c, out);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

Tensor mix_dbl_bwd_grad_y_cuda(
    const Tensor& g_out,
    const Tensor& w, const Tensor& ggw,
    const Tensor& gates, const Tensor& ggg) {
  auto out = torch::zeros({g_out.size(0), w.size(0), g_out.size(2), w.size(2)}, g_out.options());
  const c10::cuda::CUDAGuard device_guard(g_out.device());
  auto go_c = g_out.contiguous();
  auto w_c = w.contiguous(); auto ggw_c = ggw.contiguous();
  auto g_c = gates.contiguous(); auto ggg_c = ggg.contiguous();
  if (g_out.scalar_type() == torch::kFloat32) {
    launch_mix_dbl_bwd_grad_y<float>(go_c, w_c, ggw_c, g_c, ggg_c, out);
  } else {
    launch_mix_dbl_bwd_grad_y<double>(go_c, w_c, ggw_c, g_c, ggg_c, out);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

Tensor mix_dbl_bwd_grad_w_cuda(
    const Tensor& g_out,
    const Tensor& y, const Tensor& ggy,
    const Tensor& gates, const Tensor& ggg) {
  auto out = torch::zeros({y.size(1), g_out.size(1), y.size(3)}, y.options());
  const c10::cuda::CUDAGuard device_guard(g_out.device());
  auto go_c = g_out.contiguous();
  auto y_c = y.contiguous(); auto ggy_c = ggy.contiguous();
  auto g_c = gates.contiguous(); auto ggg_c = ggg.contiguous();
  if (g_out.scalar_type() == torch::kFloat32) {
    launch_mix_dbl_bwd_grad_w<float>(go_c, y_c, ggy_c, g_c, ggg_c, out);
  } else {
    launch_mix_dbl_bwd_grad_w<double>(go_c, y_c, ggy_c, g_c, ggg_c, out);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

Tensor mix_dbl_bwd_grad_g_cuda(
    const Tensor& g_out,
    const Tensor& y, const Tensor& ggy,
    const Tensor& w, const Tensor& ggw) {
  auto out = torch::zeros({y.size(0), y.size(1)}, y.options());
  const c10::cuda::CUDAGuard device_guard(g_out.device());
  auto go_c = g_out.contiguous();
  auto y_c = y.contiguous(); auto ggy_c = ggy.contiguous();
  auto w_c = w.contiguous(); auto ggw_c = ggw.contiguous();
  if (g_out.scalar_type() == torch::kFloat32) {
    launch_mix_dbl_bwd_grad_g<float>(go_c, y_c, ggy_c, w_c, ggw_c, out);
  } else {
    launch_mix_dbl_bwd_grad_g<double>(go_c, y_c, ggy_c, w_c, ggw_c, out);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

Tensor project_bucket_forward_cuda(
    const Tensor& a,
    const Tensor& b,
    const Tensor& u_bucket,
    int64_t num_paths) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda() && u_bucket.is_cuda(), "project_bucket_forward_cuda expects CUDA tensors");
  TORCH_CHECK(b.size(1) == 1, "project_bucket_forward_cuda currently supports mul_in2 == 1");
  TORCH_CHECK(u_bucket.size(1) % num_paths == 0, "u_bucket second dim must be divisible by num_paths");
  const auto kdim = u_bucket.size(1) / num_paths;
  auto y = torch::zeros({a.size(0), num_paths, kdim, a.size(1)}, a.options());
  const c10::cuda::CUDAGuard device_guard(a.device());
  if (a.scalar_type() == torch::kFloat32) {
    launch_project_forward_mul_in2eq1<float>(a.contiguous(), b.contiguous().view({b.size(0), b.size(2)}), u_bucket.contiguous(), y);
  } else if (a.scalar_type() == torch::kFloat64) {
    launch_project_forward_mul_in2eq1<double>(a.contiguous(), b.contiguous().view({b.size(0), b.size(2)}), u_bucket.contiguous(), y);
  } else {
    TORCH_CHECK(false, "project_bucket_forward_cuda supports only float32/float64");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

Tensor project_bucket_transpose_a_cuda(
    const Tensor& grad_y,
    const Tensor& b,
    const Tensor& u_bucket) {
  TORCH_CHECK(grad_y.is_cuda() && b.is_cuda() && u_bucket.is_cuda(), "project_bucket_transpose_a_cuda expects CUDA tensors");
  TORCH_CHECK(b.size(1) == 1, "project_bucket_transpose_a_cuda currently supports mul_in2 == 1");
  auto grad_a = torch::zeros({b.size(0), grad_y.size(3), u_bucket.size(0) / b.size(2)}, grad_y.options());
  const c10::cuda::CUDAGuard device_guard(grad_y.device());
  if (grad_y.scalar_type() == torch::kFloat32) {
    launch_project_transpose_a_mul_in2eq1<float>(grad_y.contiguous(), b.contiguous().view({b.size(0), b.size(2)}), u_bucket.contiguous(), grad_a);
  } else if (grad_y.scalar_type() == torch::kFloat64) {
    launch_project_transpose_a_mul_in2eq1<double>(grad_y.contiguous(), b.contiguous().view({b.size(0), b.size(2)}), u_bucket.contiguous(), grad_a);
  } else {
    TORCH_CHECK(false, "project_bucket_transpose_a_cuda supports only float32/float64");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad_a;
}

Tensor project_bucket_transpose_b_cuda(
    const Tensor& grad_y,
    const Tensor& a,
    const Tensor& u_bucket) {
  TORCH_CHECK(grad_y.is_cuda() && a.is_cuda() && u_bucket.is_cuda(), "project_bucket_transpose_b_cuda expects CUDA tensors");
  TORCH_CHECK(grad_y.size(3) % a.size(1) == 0, "grad_y IJ dim must be divisible by mul_in1");
  TORCH_CHECK(grad_y.size(3) / a.size(1) == 1, "project_bucket_transpose_b_cuda currently supports mul_in2 == 1");
  auto grad_b = torch::zeros({a.size(0), 1, u_bucket.size(0) / a.size(2)}, grad_y.options());
  const c10::cuda::CUDAGuard device_guard(grad_y.device());
  if (grad_y.scalar_type() == torch::kFloat32) {
    launch_project_transpose_b_mul_in2eq1<float>(grad_y.contiguous(), a.contiguous(), u_bucket.contiguous(), grad_b);
  } else if (grad_y.scalar_type() == torch::kFloat64) {
    launch_project_transpose_b_mul_in2eq1<double>(grad_y.contiguous(), a.contiguous(), u_bucket.contiguous(), grad_b);
  } else {
    TORCH_CHECK(false, "project_bucket_transpose_b_cuda supports only float32/float64");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad_b;
}

Tensor project_bucket_transpose_u_cuda(
    const Tensor& grad_y,
    const Tensor& a,
    const Tensor& b) {
  TORCH_CHECK(grad_y.is_cuda() && a.is_cuda() && b.is_cuda(), "project_bucket_transpose_u_cuda expects CUDA tensors");
  TORCH_CHECK(b.size(1) == 1, "project_bucket_transpose_u_cuda currently supports mul_in2 == 1");
  auto grad_u = torch::zeros({a.size(2) * b.size(2), grad_y.size(1) * grad_y.size(2)}, grad_y.options());
  const c10::cuda::CUDAGuard device_guard(grad_y.device());
  if (grad_y.scalar_type() == torch::kFloat32) {
    launch_project_transpose_u_mul_in2eq1<float>(grad_y.contiguous(), a.contiguous(), b.contiguous().view({b.size(0), b.size(2)}), grad_u);
  } else if (grad_y.scalar_type() == torch::kFloat64) {
    launch_project_transpose_u_mul_in2eq1<double>(grad_y.contiguous(), a.contiguous(), b.contiguous().view({b.size(0), b.size(2)}), grad_u);
  } else {
    TORCH_CHECK(false, "project_bucket_transpose_u_cuda supports only float32/float64");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad_u;
}

Tensor mix_bucket_forward_cuda(
    const Tensor& y,
    const Tensor& w,
    const Tensor& gates) {
  TORCH_CHECK(y.is_cuda() && w.is_cuda() && gates.is_cuda(), "mix_bucket_forward_cuda expects CUDA tensors");
  auto out = torch::zeros({y.size(0), w.size(1), y.size(2)}, y.options());
  const c10::cuda::CUDAGuard device_guard(y.device());
  auto y_c = y.contiguous();
  auto w_c = w.contiguous();
  auto g_c = gates.contiguous();
  if (y.scalar_type() == torch::kFloat32) {
    launch_mix_forward<float>(y_c, w_c, g_c, out);
  } else if (y.scalar_type() == torch::kFloat64) {
    launch_mix_forward<double>(y_c, w_c, g_c, out);
  } else {
    TORCH_CHECK(false, "mix_bucket_forward_cuda supports only float32/float64");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

Tensor mix_bucket_transpose_y_cuda(
    const Tensor& grad_out,
    const Tensor& w,
    const Tensor& gates) {
  TORCH_CHECK(grad_out.is_cuda() && w.is_cuda() && gates.is_cuda(), "mix_bucket_transpose_y_cuda expects CUDA tensors");
  auto grad_y = torch::zeros({grad_out.size(0), w.size(0), grad_out.size(2), w.size(2)}, grad_out.options());
  const c10::cuda::CUDAGuard device_guard(grad_out.device());
  auto go_c = grad_out.contiguous();
  auto w_c = w.contiguous();
  auto g_c = gates.contiguous();
  if (grad_out.scalar_type() == torch::kFloat32) {
    launch_mix_transpose_y<float>(go_c, w_c, g_c, grad_y);
  } else if (grad_out.scalar_type() == torch::kFloat64) {
    launch_mix_transpose_y<double>(go_c, w_c, g_c, grad_y);
  } else {
    TORCH_CHECK(false, "mix_bucket_transpose_y_cuda supports only float32/float64");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad_y;
}

Tensor mix_bucket_transpose_w_cuda(
    const Tensor& grad_out,
    const Tensor& y,
    const Tensor& gates) {
  TORCH_CHECK(grad_out.is_cuda() && y.is_cuda() && gates.is_cuda(), "mix_bucket_transpose_w_cuda expects CUDA tensors");
  auto grad_w = torch::zeros({y.size(1), grad_out.size(1), y.size(3)}, y.options());
  const c10::cuda::CUDAGuard device_guard(grad_out.device());
  auto go_c = grad_out.contiguous();
  auto y_c = y.contiguous();
  auto g_c = gates.contiguous();
  if (grad_out.scalar_type() == torch::kFloat32) {
    launch_mix_transpose_w<float>(go_c, y_c, g_c, grad_w);
  } else if (grad_out.scalar_type() == torch::kFloat64) {
    launch_mix_transpose_w<double>(go_c, y_c, g_c, grad_w);
  } else {
    TORCH_CHECK(false, "mix_bucket_transpose_w_cuda supports only float32/float64");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad_w;
}

Tensor mix_bucket_transpose_g_cuda(
    const Tensor& grad_out,
    const Tensor& y,
    const Tensor& w) {
  TORCH_CHECK(grad_out.is_cuda() && y.is_cuda() && w.is_cuda(), "mix_bucket_transpose_g_cuda expects CUDA tensors");
  auto grad_g = torch::zeros({y.size(0), y.size(1)}, y.options());
  const c10::cuda::CUDAGuard device_guard(grad_out.device());
  auto go_c = grad_out.contiguous();
  auto y_c = y.contiguous();
  auto w_c = w.contiguous();
  if (grad_out.scalar_type() == torch::kFloat32) {
    launch_mix_transpose_g<float>(go_c, y_c, w_c, grad_g);
  } else if (grad_out.scalar_type() == torch::kFloat64) {
    launch_mix_transpose_g<double>(go_c, y_c, w_c, grad_g);
  } else {
    TORCH_CHECK(false, "mix_bucket_transpose_g_cuda supports only float32/float64");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad_g;
}
