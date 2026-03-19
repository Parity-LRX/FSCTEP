#include <torch/extension.h>

#include <vector>

using torch::Tensor;

#ifdef WITH_CUDA
Tensor project_bucket_forward_cuda(
    const Tensor& a,
    const Tensor& b,
    const Tensor& u_bucket,
    int64_t num_paths);
Tensor project_bucket_transpose_a_cuda(
    const Tensor& grad_y,
    const Tensor& b,
    const Tensor& u_bucket);
Tensor project_bucket_transpose_b_cuda(
    const Tensor& grad_y,
    const Tensor& a,
    const Tensor& u_bucket);
Tensor project_bucket_transpose_u_cuda(
    const Tensor& grad_y,
    const Tensor& a,
    const Tensor& b);
Tensor mix_bucket_forward_cuda(
    const Tensor& y,
    const Tensor& w,
    const Tensor& gates);
Tensor mix_bucket_transpose_y_cuda(
    const Tensor& grad_out,
    const Tensor& w,
    const Tensor& gates);
Tensor mix_bucket_transpose_w_cuda(
    const Tensor& grad_out,
    const Tensor& y,
    const Tensor& gates);
Tensor mix_bucket_transpose_g_cuda(
    const Tensor& grad_out,
    const Tensor& y,
    const Tensor& w);
Tensor project_dbl_bwd_grad_h_cuda(
    const Tensor& a, const Tensor& gga,
    const Tensor& b, const Tensor& ggb,
    const Tensor& u, const Tensor& ggu,
    int64_t num_paths);
Tensor project_dbl_bwd_grad_a_cuda(
    const Tensor& h,
    const Tensor& b, const Tensor& ggb,
    const Tensor& u, const Tensor& ggu);
Tensor project_dbl_bwd_grad_b_cuda(
    const Tensor& h,
    const Tensor& a, const Tensor& gga,
    const Tensor& u, const Tensor& ggu);
Tensor project_dbl_bwd_grad_u_cuda(
    const Tensor& h,
    const Tensor& a, const Tensor& gga,
    const Tensor& b, const Tensor& ggb);
Tensor mix_dbl_bwd_grad_g_out_cuda(
    const Tensor& y, const Tensor& ggy,
    const Tensor& w, const Tensor& ggw,
    const Tensor& gates, const Tensor& ggg);
Tensor mix_dbl_bwd_grad_y_cuda(
    const Tensor& g_out,
    const Tensor& w, const Tensor& ggw,
    const Tensor& gates, const Tensor& ggg);
Tensor mix_dbl_bwd_grad_w_cuda(
    const Tensor& g_out,
    const Tensor& y, const Tensor& ggy,
    const Tensor& gates, const Tensor& ggg);
Tensor mix_dbl_bwd_grad_g_cuda(
    const Tensor& g_out,
    const Tensor& y, const Tensor& ggy,
    const Tensor& w, const Tensor& ggw);
#endif

namespace {

void check_projection_args(
    const Tensor& a,
    const Tensor& b,
    const Tensor& u_bucket) {
  TORCH_CHECK(a.dim() == 3, "a must have shape (B, mul_in1, m1)");
  TORCH_CHECK(b.dim() == 3, "b must have shape (B, mul_in2, m2)");
  TORCH_CHECK(u_bucket.dim() == 2, "u_bucket must have shape (m1*m2, P*kdim)");
  TORCH_CHECK(a.device() == b.device(), "a and b must be on the same device");
  TORCH_CHECK(a.device() == u_bucket.device(), "a and u_bucket must be on the same device");
  TORCH_CHECK(a.scalar_type() == b.scalar_type(), "a and b must have the same dtype");
  TORCH_CHECK(a.scalar_type() == u_bucket.scalar_type(), "a and u_bucket must have the same dtype");
  TORCH_CHECK(b.size(0) == a.size(0), "a and b batch dimensions must match");
  TORCH_CHECK(u_bucket.size(0) == a.size(2) * b.size(2), "u_bucket first dim must equal m1*m2");
}

void check_mix_args(
    const Tensor& y,
    const Tensor& w,
    const Tensor& gates) {
  TORCH_CHECK(y.dim() == 4, "y must have shape (B, P, kdim, IJ)");
  TORCH_CHECK(w.dim() == 3, "w must have shape (P, mul_out, IJ)");
  TORCH_CHECK(gates.dim() == 2, "gates must have shape (B, P)");
  TORCH_CHECK(y.device() == w.device(), "y and w must be on the same device");
  TORCH_CHECK(y.device() == gates.device(), "y and gates must be on the same device");
  TORCH_CHECK(y.scalar_type() == w.scalar_type(), "y and w must have the same dtype");
  TORCH_CHECK(y.scalar_type() == gates.scalar_type(), "y and gates must have the same dtype");
  TORCH_CHECK(y.size(0) == gates.size(0), "gates batch dim must match y");
  TORCH_CHECK(y.size(1) == w.size(0), "w path dim must match y");
  TORCH_CHECK(y.size(1) == gates.size(1), "gates path dim must match y");
  TORCH_CHECK(y.size(3) == w.size(2), "w IJ dim must match y");
}

Tensor pack_u_stage1(
    const Tensor& u_bucket,
    const int64_t m1,
    const int64_t m2) {
  const auto pk = u_bucket.size(1);
  return u_bucket.contiguous().view({m1, m2, pk}).permute({0, 2, 1}).contiguous().view({m1, pk * m2});
}

Tensor unpack_stage1_grad_to_u_bucket(
    const Tensor& grad_u_pack,
    const int64_t m1,
    const int64_t m2,
    const int64_t pk) {
  return grad_u_pack.view({m1, pk, m2}).permute({0, 2, 1}).contiguous().view({m1 * m2, pk});
}

Tensor grad_y_to_stage2_rows(
    const Tensor& grad_y,
    const int64_t mul_in1,
    const int64_t mul_in2) {
  TORCH_CHECK(grad_y.dim() == 4, "grad_y must have shape (B, P, kdim, IJ)");
  const auto batch = grad_y.size(0);
  const auto num_paths = grad_y.size(1);
  const auto kdim = grad_y.size(2);
  const auto ij = grad_y.size(3);
  TORCH_CHECK(ij == mul_in1 * mul_in2, "grad_y last dim must equal mul_in1*mul_in2");
  auto grad_y_5d = grad_y.contiguous().view({batch, num_paths, kdim, mul_in1, mul_in2});
  return grad_y_5d.permute({0, 3, 1, 2, 4}).contiguous().view({batch, mul_in1 * num_paths * kdim, mul_in2});
}

Tensor project_bucket_forward(
    const Tensor& a,
    const Tensor& b,
    const Tensor& u_bucket,
    const int64_t num_paths) {
  check_projection_args(a, b, u_bucket);
  TORCH_CHECK(num_paths > 0, "num_paths must be positive");
  TORCH_CHECK(u_bucket.size(1) % num_paths == 0, "u_bucket second dim must be divisible by num_paths");

  const auto batch = a.size(0);
  const auto mul_in1 = a.size(1);
  const auto mul_in2 = b.size(1);
  const auto kdim = u_bucket.size(1) / num_paths;
  const auto pk = num_paths * kdim;
  const auto ij = mul_in1 * mul_in2;
  const auto m1 = a.size(2);
  const auto m2 = b.size(2);

#ifdef WITH_CUDA
  if (a.is_cuda() && mul_in2 == 1) {
    return project_bucket_forward_cuda(a, b, u_bucket, num_paths);
  }
#endif

  auto a_c = a.contiguous();
  auto b_c = b.contiguous();
  auto u_stage1 = pack_u_stage1(u_bucket, m1, m2);                          // (m1, pk*m2)
  auto tmp_flat = at::matmul(a_c.view({batch * mul_in1, m1}), u_stage1);    // (B*I, pk*m2)
  auto tmp = tmp_flat.view({batch, mul_in1, pk, m2});                        // (B, I, pk, m2)
  auto tmp_rows = tmp.view({batch, mul_in1 * pk, m2});                       // (B, I*pk, m2)
  auto b_t = b_c.transpose(1, 2).contiguous();                               // (B, m2, J)
  auto y_rows = at::bmm(tmp_rows, b_t);                                      // (B, I*pk, J)
  return y_rows.view({batch, mul_in1, num_paths, kdim, mul_in2})
      .permute({0, 2, 3, 1, 4})
      .contiguous()
      .view({batch, num_paths, kdim, ij});
}

Tensor project_bucket_transpose_a(
    const Tensor& grad_y,
    const Tensor& b,
    const Tensor& u_bucket) {
  TORCH_CHECK(b.dim() == 3, "b must have shape (B, mul_in2, m2)");
  TORCH_CHECK(u_bucket.dim() == 2, "u_bucket must have shape (m1*m2, P*kdim)");
  const auto batch = b.size(0);
  const auto mul_in2 = b.size(1);
  const auto m2 = b.size(2);
  TORCH_CHECK(grad_y.dim() == 4, "grad_y must have shape (B, P, kdim, IJ)");
  TORCH_CHECK(grad_y.size(0) == batch, "grad_y batch must match b");
  TORCH_CHECK(grad_y.size(3) % mul_in2 == 0, "grad_y IJ dim must be divisible by mul_in2");
  const auto mul_in1 = grad_y.size(3) / mul_in2;
  const auto m1 = u_bucket.size(0) / m2;
  const auto num_paths = grad_y.size(1);
  const auto kdim = grad_y.size(2);
  const auto pk = num_paths * kdim;
  TORCH_CHECK(m1 * m2 == u_bucket.size(0), "u_bucket first dim must factor into m1*m2");

#ifdef WITH_CUDA
  if (grad_y.is_cuda() && mul_in2 == 1) {
    return project_bucket_transpose_a_cuda(grad_y, b, u_bucket);
  }
#endif

  auto grad_rows = grad_y_to_stage2_rows(grad_y, mul_in1, mul_in2);        // (B, I*pk, J)
  auto grad_tmp = at::bmm(grad_rows, b.contiguous());                       // (B, I*pk, m2)
  auto grad_tmp_flat = grad_tmp.view({batch * mul_in1, pk * m2});          // (B*I, pk*m2)
  auto u_stage1_t = pack_u_stage1(u_bucket, m1, m2).transpose(0, 1).contiguous();  // (pk*m2, m1)
  return at::matmul(grad_tmp_flat, u_stage1_t).view({batch, mul_in1, m1}).contiguous();
}

Tensor project_bucket_transpose_b(
    const Tensor& grad_y,
    const Tensor& a,
    const Tensor& u_bucket) {
  TORCH_CHECK(a.dim() == 3, "a must have shape (B, mul_in1, m1)");
  TORCH_CHECK(u_bucket.dim() == 2, "u_bucket must have shape (m1*m2, P*kdim)");
  const auto batch = a.size(0);
  const auto mul_in1 = a.size(1);
  const auto m1 = a.size(2);
  TORCH_CHECK(grad_y.dim() == 4, "grad_y must have shape (B, P, kdim, IJ)");
  TORCH_CHECK(grad_y.size(0) == batch, "grad_y batch must match a");
  TORCH_CHECK(grad_y.size(3) % mul_in1 == 0, "grad_y IJ dim must be divisible by mul_in1");
  const auto mul_in2 = grad_y.size(3) / mul_in1;
  const auto m2 = u_bucket.size(0) / m1;
  const auto num_paths = grad_y.size(1);
  const auto kdim = grad_y.size(2);
  const auto pk = num_paths * kdim;
  TORCH_CHECK(m1 * m2 == u_bucket.size(0), "u_bucket first dim must factor into m1*m2");

#ifdef WITH_CUDA
  if (grad_y.is_cuda() && mul_in2 == 1) {
    return project_bucket_transpose_b_cuda(grad_y, a, u_bucket);
  }
#endif

  auto grad_rows = grad_y_to_stage2_rows(grad_y, mul_in1, mul_in2);        // (B, I*pk, J)
  auto u_stage1 = pack_u_stage1(u_bucket, m1, m2);                          // (m1, pk*m2)
  auto tmp_flat = at::matmul(a.contiguous().view({batch * mul_in1, m1}), u_stage1);  // (B*I, pk*m2)
  auto tmp_rows = tmp_flat.view({batch, mul_in1 * pk, m2});                 // (B, I*pk, m2)
  auto grad_b_t = at::bmm(tmp_rows.transpose(1, 2).contiguous(), grad_rows);  // (B, m2, J)
  return grad_b_t.transpose(1, 2).contiguous();
}

Tensor project_bucket_transpose_u(
    const Tensor& grad_y,
    const Tensor& a,
    const Tensor& b) {
  TORCH_CHECK(a.dim() == 3, "a must have shape (B, mul_in1, m1)");
  TORCH_CHECK(b.dim() == 3, "b must have shape (B, mul_in2, m2)");
  TORCH_CHECK(a.device() == b.device(), "a and b must be on the same device");
  TORCH_CHECK(a.scalar_type() == b.scalar_type(), "a and b must have the same dtype");
  const auto mul_in1 = a.size(1);
  const auto mul_in2 = b.size(1);
  const auto m1 = a.size(2);
  const auto m2 = b.size(2);
  const auto num_paths = grad_y.size(1);
  const auto kdim = grad_y.size(2);
  const auto pk = num_paths * kdim;
#ifdef WITH_CUDA
  if (grad_y.is_cuda() && mul_in2 == 1) {
    return project_bucket_transpose_u_cuda(grad_y, a, b);
  }
#endif
  auto grad_rows = grad_y_to_stage2_rows(grad_y, mul_in1, mul_in2);        // (B, I*pk, J)
  auto grad_tmp = at::bmm(grad_rows, b.contiguous());                       // (B, I*pk, m2)
  auto grad_tmp_flat = grad_tmp.view({grad_tmp.size(0), mul_in1, pk * m2});  // (B, I, pk*m2)
  auto a_t = a.contiguous().transpose(1, 2).contiguous();                  // (B, m1, I)
  auto grad_u_pack = at::bmm(a_t, grad_tmp_flat).sum(0);                   // (m1, pk*m2)
  return unpack_stage1_grad_to_u_bucket(grad_u_pack, m1, m2, pk);
}

Tensor mix_bucket_forward(
    const Tensor& y,
    const Tensor& w,
    const Tensor& gates) {
  check_mix_args(y, w, gates);
  const auto batch = y.size(0);
  const auto num_paths = y.size(1);
  const auto kdim = y.size(2);
  const auto ij = y.size(3);
  const auto mul_out = w.size(1);

#ifdef WITH_CUDA
  if (y.is_cuda()) {
    return mix_bucket_forward_cuda(y, w, gates);
  }
#endif

  auto y_bmm = y.permute({1, 0, 2, 3}).contiguous().view({num_paths, batch * kdim, ij});
  auto w_bmm = w.permute({0, 2, 1}).contiguous();
  auto out_bmm = at::bmm(y_bmm, w_bmm);  // (P, B*k, O)
  auto out_per = out_bmm.view({num_paths, batch, kdim, mul_out}).permute({1, 0, 3, 2}).contiguous();
  auto gated = out_per * gates.contiguous().view({batch, num_paths, 1, 1});
  return gated.sum(1).contiguous();
}

Tensor mix_bucket_transpose_y(
    const Tensor& grad_out,
    const Tensor& w,
    const Tensor& gates) {
  TORCH_CHECK(grad_out.dim() == 3, "grad_out must have shape (B, O, kdim)");
  TORCH_CHECK(w.dim() == 3, "w must have shape (P, O, IJ)");
  TORCH_CHECK(gates.dim() == 2, "gates must have shape (B, P)");
  const auto batch = grad_out.size(0);
  const auto mul_out = grad_out.size(1);
  const auto kdim = grad_out.size(2);
  const auto num_paths = w.size(0);
  const auto ij = w.size(2);
  TORCH_CHECK(w.size(1) == mul_out, "w output dim must match grad_out");
  TORCH_CHECK(gates.size(0) == batch && gates.size(1) == num_paths, "gates must match grad_out and w");

#ifdef WITH_CUDA
  if (grad_out.is_cuda()) {
    return mix_bucket_transpose_y_cuda(grad_out, w, gates);
  }
#endif

  auto grad_out_gated = grad_out.contiguous().unsqueeze(1) * gates.contiguous().view({batch, num_paths, 1, 1});
  auto lhs = grad_out_gated.permute({1, 0, 3, 2}).contiguous().view({num_paths, batch * kdim, mul_out});
  auto rhs = w.contiguous();
  auto grad_y_bmm = at::bmm(lhs, rhs);  // (P, B*k, IJ)
  return grad_y_bmm.view({num_paths, batch, kdim, ij}).permute({1, 0, 2, 3}).contiguous();
}

Tensor mix_bucket_transpose_w(
    const Tensor& grad_out,
    const Tensor& y,
    const Tensor& gates) {
  check_mix_args(y, torch::empty({y.size(1), grad_out.size(1), y.size(3)}, y.options()), gates);
  const auto batch = y.size(0);
  const auto num_paths = y.size(1);
  const auto kdim = y.size(2);
  const auto ij = y.size(3);
  const auto mul_out = grad_out.size(1);

#ifdef WITH_CUDA
  if (grad_out.is_cuda()) {
    return mix_bucket_transpose_w_cuda(grad_out, y, gates);
  }
#endif

  auto grad_out_gated = grad_out.contiguous().unsqueeze(1) * gates.contiguous().view({batch, num_paths, 1, 1});
  auto lhs = grad_out_gated.permute({1, 2, 0, 3}).contiguous().view({num_paths, mul_out, batch * kdim});
  auto rhs = y.permute({1, 0, 2, 3}).contiguous().view({num_paths, batch * kdim, ij});
  return at::bmm(lhs, rhs).contiguous();  // (P, O, IJ)
}

Tensor mix_bucket_transpose_g(
    const Tensor& grad_out,
    const Tensor& y,
    const Tensor& w) {
  TORCH_CHECK(grad_out.dim() == 3, "grad_out must have shape (B, O, kdim)");
  TORCH_CHECK(y.dim() == 4, "y must have shape (B, P, kdim, IJ)");
  TORCH_CHECK(w.dim() == 3, "w must have shape (P, O, IJ)");
  const auto batch = y.size(0);
  const auto num_paths = y.size(1);
  const auto kdim = y.size(2);
  const auto ij = y.size(3);
  const auto mul_out = grad_out.size(1);
  TORCH_CHECK(grad_out.size(0) == batch && grad_out.size(2) == kdim, "grad_out must match y");
  TORCH_CHECK(w.size(0) == num_paths && w.size(1) == mul_out && w.size(2) == ij, "w must match y and grad_out");

#ifdef WITH_CUDA
  if (grad_out.is_cuda()) {
    return mix_bucket_transpose_g_cuda(grad_out, y, w);
  }
#endif

  auto y_bmm = y.permute({1, 0, 2, 3}).contiguous().view({num_paths, batch * kdim, ij});
  auto w_bmm = w.permute({0, 2, 1}).contiguous();
  auto out_bmm = at::bmm(y_bmm, w_bmm);
  auto out_per = out_bmm.view({num_paths, batch, kdim, mul_out}).permute({1, 0, 3, 2}).contiguous();
  return (grad_out.contiguous().unsqueeze(1) * out_per).sum({2, 3}).contiguous();
}

Tensor bucketed_tp_forward_impl(
    const Tensor& a,
    const Tensor& b,
    const Tensor& u_bucket,
    const Tensor& w,
    const Tensor& gates) {
  check_projection_args(a, b, u_bucket);
  TORCH_CHECK(w.dim() == 3, "w must have shape (P, mul_out, IJ)");
  TORCH_CHECK(gates.dim() == 2, "gates must have shape (B, P)");
  const auto num_paths = w.size(0);
  TORCH_CHECK(gates.size(1) == num_paths, "gates path dim must match w");
  TORCH_CHECK(u_bucket.size(1) % num_paths == 0, "u_bucket second dim must be divisible by number of paths");
  auto y = project_bucket_forward(a, b, u_bucket, num_paths);
  return mix_bucket_forward(y, w, gates);
}

Tensor project_dbl_bwd_grad_h(
    const Tensor& a, const Tensor& gga,
    const Tensor& b, const Tensor& ggb,
    const Tensor& u, const Tensor& ggu,
    int64_t num_paths) {
#ifdef WITH_CUDA
  if (a.is_cuda()) {
    return project_dbl_bwd_grad_h_cuda(a, gga, b, ggb, u, ggu, num_paths);
  }
#endif
  TORCH_CHECK(false, "project_dbl_bwd_grad_h requires CUDA tensors");
  return {};
}

Tensor project_dbl_bwd_grad_a(
    const Tensor& h,
    const Tensor& b, const Tensor& ggb,
    const Tensor& u, const Tensor& ggu) {
#ifdef WITH_CUDA
  if (h.is_cuda()) {
    return project_dbl_bwd_grad_a_cuda(h, b, ggb, u, ggu);
  }
#endif
  TORCH_CHECK(false, "project_dbl_bwd_grad_a requires CUDA tensors");
  return {};
}

Tensor project_dbl_bwd_grad_b(
    const Tensor& h,
    const Tensor& a, const Tensor& gga,
    const Tensor& u, const Tensor& ggu) {
#ifdef WITH_CUDA
  if (h.is_cuda()) {
    return project_dbl_bwd_grad_b_cuda(h, a, gga, u, ggu);
  }
#endif
  TORCH_CHECK(false, "project_dbl_bwd_grad_b requires CUDA tensors");
  return {};
}

Tensor project_dbl_bwd_grad_u(
    const Tensor& h,
    const Tensor& a, const Tensor& gga,
    const Tensor& b, const Tensor& ggb) {
#ifdef WITH_CUDA
  if (h.is_cuda()) {
    return project_dbl_bwd_grad_u_cuda(h, a, gga, b, ggb);
  }
#endif
  TORCH_CHECK(false, "project_dbl_bwd_grad_u requires CUDA tensors");
  return {};
}

Tensor mix_dbl_bwd_grad_g_out(
    const Tensor& y, const Tensor& ggy,
    const Tensor& w, const Tensor& ggw,
    const Tensor& gates, const Tensor& ggg) {
#ifdef WITH_CUDA
  if (y.is_cuda()) {
    return mix_dbl_bwd_grad_g_out_cuda(y, ggy, w, ggw, gates, ggg);
  }
#endif
  TORCH_CHECK(false, "mix_dbl_bwd_grad_g_out requires CUDA tensors");
  return {};
}

Tensor mix_dbl_bwd_grad_y(
    const Tensor& g_out,
    const Tensor& w, const Tensor& ggw,
    const Tensor& gates, const Tensor& ggg) {
#ifdef WITH_CUDA
  if (g_out.is_cuda()) {
    return mix_dbl_bwd_grad_y_cuda(g_out, w, ggw, gates, ggg);
  }
#endif
  TORCH_CHECK(false, "mix_dbl_bwd_grad_y requires CUDA tensors");
  return {};
}

Tensor mix_dbl_bwd_grad_w(
    const Tensor& g_out,
    const Tensor& y, const Tensor& ggy,
    const Tensor& gates, const Tensor& ggg) {
#ifdef WITH_CUDA
  if (g_out.is_cuda()) {
    return mix_dbl_bwd_grad_w_cuda(g_out, y, ggy, gates, ggg);
  }
#endif
  TORCH_CHECK(false, "mix_dbl_bwd_grad_w requires CUDA tensors");
  return {};
}

Tensor mix_dbl_bwd_grad_g(
    const Tensor& g_out,
    const Tensor& y, const Tensor& ggy,
    const Tensor& w, const Tensor& ggw) {
#ifdef WITH_CUDA
  if (g_out.is_cuda()) {
    return mix_dbl_bwd_grad_g_cuda(g_out, y, ggy, w, ggw);
  }
#endif
  TORCH_CHECK(false, "mix_dbl_bwd_grad_g requires CUDA tensors");
  return {};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("project_bucket_forward", &project_bucket_forward, "ICTD TP projection forward");
  m.def("project_bucket_transpose_a", &project_bucket_transpose_a, "ICTD TP projection transpose wrt a");
  m.def("project_bucket_transpose_b", &project_bucket_transpose_b, "ICTD TP projection transpose wrt b");
  m.def("project_bucket_transpose_u", &project_bucket_transpose_u, "ICTD TP projection transpose wrt U");
  m.def("mix_bucket_forward", &mix_bucket_forward, "ICTD TP channel mix forward");
  m.def("mix_bucket_transpose_y", &mix_bucket_transpose_y, "ICTD TP channel mix transpose wrt y");
  m.def("mix_bucket_transpose_w", &mix_bucket_transpose_w, "ICTD TP channel mix transpose wrt w");
  m.def("mix_bucket_transpose_g", &mix_bucket_transpose_g, "ICTD TP channel mix transpose wrt gates");
  m.def("bucketed_tp_forward", &bucketed_tp_forward_impl, "Packed bucketed ICTD TP forward");
  m.def("project_dbl_bwd_grad_h", &project_dbl_bwd_grad_h, "Project dbl-bwd: grad wrt H (3-term fused PF)");
  m.def("project_dbl_bwd_grad_a", &project_dbl_bwd_grad_a, "Project dbl-bwd: grad wrt a (2-term fused PAT)");
  m.def("project_dbl_bwd_grad_b", &project_dbl_bwd_grad_b, "Project dbl-bwd: grad wrt b (2-term fused PBT)");
  m.def("project_dbl_bwd_grad_u", &project_dbl_bwd_grad_u, "Project dbl-bwd: grad wrt U (2-term fused PUT)");
  m.def("mix_dbl_bwd_grad_g_out", &mix_dbl_bwd_grad_g_out, "Mix dbl-bwd: grad wrt G (3-term fused MF)");
  m.def("mix_dbl_bwd_grad_y", &mix_dbl_bwd_grad_y, "Mix dbl-bwd: grad wrt Y (2-term fused MYT)");
  m.def("mix_dbl_bwd_grad_w", &mix_dbl_bwd_grad_w, "Mix dbl-bwd: grad wrt W (2-term fused MWT)");
  m.def("mix_dbl_bwd_grad_g", &mix_dbl_bwd_grad_g, "Mix dbl-bwd: grad wrt g (2-term fused MGT)");
  m.def("has_cuda", []() {
#ifdef WITH_CUDA
    return true;
#else
    return false;
#endif
  });
}
