#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

torch::Tensor rms_norm_cuda(torch::Tensor x, torch::Tensor weight, float epsilon);
torch::Tensor rms_norm_backward_cuda(torch::Tensor grad_y, torch::Tensor x, torch::Tensor weight, float epsilon);
torch::Tensor rope_cuda(torch::Tensor x, torch::Tensor cos_table, torch::Tensor sin_table);
torch::Tensor rope_backward_cuda(torch::Tensor grad_y, torch::Tensor cos_table, torch::Tensor sin_table);
std::vector<torch::Tensor> xentropy_forward_cuda(torch::Tensor logits, torch::Tensor labels, float smoothing, float z_loss, int64_t ignore_index);
torch::Tensor xentropy_backward_cuda(torch::Tensor grad_losses, torch::Tensor logits, torch::Tensor logsumexp, torch::Tensor labels, float smoothing, float z_loss, int64_t ignore_index);

void init_thread_manager(py::module &m);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("rms_norm", &rms_norm_cuda, "RMSNorm forward (CUDA)");
    m.def("rms_norm_backward", &rms_norm_backward_cuda, "RMSNorm backward (CUDA)");
    m.def("rope", &rope_cuda, "RoPE forward (CUDA)");
    m.def("rope_backward", &rope_backward_cuda, "RoPE backward (CUDA)");
    m.def("xentropy_forward", &xentropy_forward_cuda, "Softmax cross entropy forward (CUDA)");
    m.def("xentropy_backward", &xentropy_backward_cuda, "Softmax cross entropy backward (CUDA)");
    init_thread_manager(m);
}
