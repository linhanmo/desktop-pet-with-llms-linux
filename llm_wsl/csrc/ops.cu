#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
template <typename T>
__global__ void rms_norm_kernel(
    const T *__restrict__ x,
    const T *__restrict__ weight,
    T *__restrict__ output,
    const float epsilon,
    const int hidden_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = idx / hidden_dim;
    int row = blockIdx.x;
    int tid = threadIdx.x;
    extern __shared__ float sdata[];
    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x)
    {
        float val = static_cast<float>(x[row * hidden_dim + i]);
        sum_sq += val * val;
    }
    sdata[tid] = sum_sq;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    float inv_rms = 0.0f;
    if (tid == 0)
    {
        inv_rms = rsqrtf(sdata[0] / hidden_dim + epsilon);
        sdata[0] = inv_rms;
    }
    __syncthreads();
    inv_rms = sdata[0];
    for (int i = tid; i < hidden_dim; i += blockDim.x)
    {
        float val = static_cast<float>(x[row * hidden_dim + i]);
        output[row * hidden_dim + i] = static_cast<T>(val * inv_rms * static_cast<float>(weight[i]));
    }
}
template <typename T>
__global__ void rms_norm_backward_kernel(
    const T *__restrict__ grad_y,
    const T *__restrict__ x,
    const T *__restrict__ weight,
    T *__restrict__ grad_x,
    const float epsilon,
    const int hidden_dim)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    extern __shared__ float sdata[];
    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x)
    {
        float val = static_cast<float>(x[row * hidden_dim + i]);
        sum_sq += val * val;
    }
    sdata[tid] = sum_sq;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float inv_rms = rsqrtf(sdata[0] / hidden_dim + epsilon);
    float sum_grad_term = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x)
    {
        int idx = row * hidden_dim + i;
        float val_x = static_cast<float>(x[idx]);
        float val_dy = static_cast<float>(grad_y[idx]);
        float val_w = static_cast<float>(weight[i]);
        sum_grad_term += val_dy * val_w * val_x;
    }
    __syncthreads();
    sdata[tid] = sum_grad_term;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float global_sum_term = sdata[0];
    float mean_term = global_sum_term * inv_rms * inv_rms / hidden_dim;
    for (int i = tid; i < hidden_dim; i += blockDim.x)
    {
        int idx = row * hidden_dim + i;
        float val_x = static_cast<float>(x[idx]);
        float val_dy = static_cast<float>(grad_y[idx]);
        float val_w = static_cast<float>(weight[i]);
        float res = inv_rms * (val_dy * val_w - mean_term * val_x);
        grad_x[idx] = static_cast<T>(res);
    }
}
torch::Tensor rms_norm_cuda(torch::Tensor x, torch::Tensor weight, float epsilon)
{
    auto x_c = x.contiguous();
    auto w_c = weight.contiguous();
    auto output = torch::empty_like(x_c);
    int hidden_dim = x.size(-1);
    int num_tokens = x.numel() / hidden_dim;
    int threads = 256;
    int blocks = num_tokens;
    size_t shared_mem = threads * sizeof(float);
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "rms_norm_cuda", ([&]
                                                                                                 { rms_norm_kernel<scalar_t><<<blocks, threads, shared_mem, stream>>>(
                                                                                                       x_c.data_ptr<scalar_t>(),
                                                                                                       w_c.data_ptr<scalar_t>(),
                                                                                                       output.data_ptr<scalar_t>(),
                                                                                                       epsilon,
                                                                                                       hidden_dim); }));
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "rms_norm_cuda launch failed: ", cudaGetErrorString(err));
    return output;
}
torch::Tensor rms_norm_backward_cuda(
    torch::Tensor grad_y,
    torch::Tensor x,
    torch::Tensor weight,
    float epsilon)
{
    auto grad_y_c = grad_y.contiguous();
    auto x_c = x.contiguous();
    auto w_c = weight.contiguous();
    auto grad_x = torch::empty_like(x_c);
    int hidden_dim = x.size(-1);
    int num_tokens = x.numel() / hidden_dim;
    int threads = 256;
    int blocks = num_tokens;
    size_t shared_mem = threads * sizeof(float);
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "rms_norm_backward_cuda", ([&]
                                                                                                          { rms_norm_backward_kernel<scalar_t><<<blocks, threads, shared_mem, stream>>>(
                                                                                                                grad_y_c.data_ptr<scalar_t>(),
                                                                                                                x_c.data_ptr<scalar_t>(),
                                                                                                                w_c.data_ptr<scalar_t>(),
                                                                                                                grad_x.data_ptr<scalar_t>(),
                                                                                                                epsilon,
                                                                                                                hidden_dim); }));
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "rms_norm_backward_cuda launch failed: ", cudaGetErrorString(err));
    return grad_x;
}
template <typename T>
__global__ void rope_kernel(
    const T *__restrict__ x,
    const T *__restrict__ cos_table,
    const T *__restrict__ sin_table,
    T *__restrict__ output,
    int head_dim,
    int num_heads,
    int seq_len,
    int total_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements / 2)
        return;
    int half_dim = head_dim / 2;
    int head_idx = (idx / half_dim) % num_heads;
    int seq_idx = (idx / (half_dim * num_heads)) % seq_len;
    int batch_idx = idx / (half_dim * num_heads * seq_len);
    int dim_idx = idx % half_dim;
    int offset = batch_idx * (seq_len * num_heads * head_dim) +
                 seq_idx * (num_heads * head_dim) +
                 head_idx * head_dim +
                 dim_idx;
    float x1 = static_cast<float>(x[offset]);
    float x2 = static_cast<float>(x[offset + half_dim]);
    int rot_idx = seq_idx * half_dim + dim_idx;
    float c = static_cast<float>(cos_table[rot_idx]);
    float s = static_cast<float>(sin_table[rot_idx]);
    float out_val1 = x1 * c - x2 * s;
    float out_val2 = x1 * s + x2 * c;
    output[offset] = static_cast<T>(out_val1);
    output[offset + half_dim] = static_cast<T>(out_val2);
}
template <typename T>
__global__ void rope_backward_kernel(
    const T *__restrict__ grad_y,
    const T *__restrict__ cos_table,
    const T *__restrict__ sin_table,
    T *__restrict__ grad_x,
    int head_dim,
    int num_heads,
    int seq_len,
    int total_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements / 2)
        return;
    int half_dim = head_dim / 2;
    int head_idx = (idx / half_dim) % num_heads;
    int seq_idx = (idx / (half_dim * num_heads)) % seq_len;
    int batch_idx = idx / (half_dim * num_heads * seq_len);
    int dim_idx = idx % half_dim;
    int offset = batch_idx * (seq_len * num_heads * head_dim) +
                 seq_idx * (num_heads * head_dim) +
                 head_idx * head_dim +
                 dim_idx;
    float dy1 = static_cast<float>(grad_y[offset]);
    float dy2 = static_cast<float>(grad_y[offset + half_dim]);
    int rot_idx = seq_idx * half_dim + dim_idx;
    float c = static_cast<float>(cos_table[rot_idx]);
    float s = static_cast<float>(sin_table[rot_idx]);
    float dx1 = dy1 * c + dy2 * s;
    float dx2 = -dy1 * s + dy2 * c;
    grad_x[offset] = static_cast<T>(dx1);
    grad_x[offset + half_dim] = static_cast<T>(dx2);
}
torch::Tensor rope_cuda(torch::Tensor x, torch::Tensor cos_table, torch::Tensor sin_table)
{
    auto x_c = x.contiguous();
    auto cos_c = cos_table.contiguous();
    auto sin_c = sin_table.contiguous();
    auto output = torch::empty_like(x_c);
    int total_elements = x.numel();
    int head_dim = x.size(-1);
    int num_heads = x.size(-2);
    int seq_len = x.size(-3);
    int threads = 256;
    int blocks = (total_elements / 2 + threads - 1) / threads;
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "rope_cuda", ([&]
                                                                                             { rope_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                                                                                                   x_c.data_ptr<scalar_t>(),
                                                                                                   cos_c.data_ptr<scalar_t>(),
                                                                                                   sin_c.data_ptr<scalar_t>(),
                                                                                                   output.data_ptr<scalar_t>(),
                                                                                                   head_dim,
                                                                                                   num_heads,
                                                                                                   seq_len,
                                                                                                   total_elements); }));
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "rope_cuda launch failed: ", cudaGetErrorString(err));
    return output;
}
torch::Tensor rope_backward_cuda(
    torch::Tensor grad_y,
    torch::Tensor cos_table,
    torch::Tensor sin_table)
{
    auto grad_y_c = grad_y.contiguous();
    auto cos_c = cos_table.contiguous();
    auto sin_c = sin_table.contiguous();
    auto grad_x = torch::empty_like(grad_y_c);
    int total_elements = grad_y.numel();
    int head_dim = grad_y.size(-1);
    int num_heads = grad_y.size(-2);
    int seq_len = grad_y.size(-3);
    int threads = 256;
    int blocks = (total_elements / 2 + threads - 1) / threads;
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, grad_y.scalar_type(), "rope_backward_cuda", ([&]
                                                                                                           { rope_backward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                                                                                                                 grad_y_c.data_ptr<scalar_t>(),
                                                                                                                 cos_c.data_ptr<scalar_t>(),
                                                                                                                 sin_c.data_ptr<scalar_t>(),
                                                                                                                 grad_x.data_ptr<scalar_t>(),
                                                                                                                 head_dim,
                                                                                                                 num_heads,
                                                                                                                 seq_len,
                                                                                                                 total_elements); }));
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "rope_backward_cuda launch failed: ", cudaGetErrorString(err));
    return grad_x;
}
