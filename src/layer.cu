#include "layer.h"

#include <cmath>
#include <cstring>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "util.h"

namespace {

inline size_t flat_rows(Tensor *tensor) {
  CHECK_ERROR(tensor->ndim >= 2, "Tensor must have at least 2 dimensions");
  return tensor->num_elem() / tensor->shape[tensor->ndim - 1];
}

inline size_t last_dim(Tensor *tensor) { return tensor->shape[tensor->ndim - 1]; }

}  // namespace

void EmbeddingPositionAdd(TokenBatch *tokens, Tensor *wte, Tensor *wpe,
                          Tensor *output) {
  CHECK_ERROR(output->shape[0] == tokens->B && output->shape[1] == tokens->T,
              "Embedding output shape mismatch");
  CHECK_ERROR(output->shape[2] == C, "Embedding hidden size mismatch");

#pragma omp parallel for collapse(2)
  for (size_t b = 0; b < tokens->B; ++b) {
    for (size_t t = 0; t < tokens->T; ++t) {
      int32_t token_id = tokens->buf[b * tokens->T + t];
      CHECK_ERROR(token_id >= 0 && token_id < (int32_t)VOCAB_SIZE,
                  "Token id %d out of range", token_id);
      const float *tok = wte->buf + (size_t)token_id * C;
      const float *pos = wpe->buf + t * C;
      float *out = output->buf + (b * tokens->T + t) * C;
      for (size_t c = 0; c < C; ++c) {
        out[c] = tok[c] + pos[c];
      }
    }
  }
}

__global__ void embedding_position_add_kernel(int* input_ids, float* wte, float* wpe, float* output, size_t vocab_size) {
  size_t S = gridDim.y;
  size_t H = blockDim.x;
  size_t b = blockIdx.x;
  size_t s = blockIdx.y;
  size_t h = threadIdx.x;

  size_t idx = input_ids[b * S + s];
  if (idx >= vocab_size)  return;
  output[(b * S * H) + (s * H) + h] = wte[idx * H + h] + wpe[s * H + h];
}

void EmbeddingPositionAdd_gpu(TokenBatch *tokens, Tensor *wte, Tensor *wpe,
                              Tensor *output) {  
  size_t batch_size  = tokens->B;
  size_t sequence    = tokens->T;
  size_t hidden_size = C;
  size_t vocab_size  = wte->shape[0];
  
  dim3 gridDim(batch_size, sequence);
  dim3 blockDim(hidden_size);

  embedding_position_add_kernel<<<gridDim, blockDim>>>(tokens->gpu_buf, wte->gpu_buf, wpe->gpu_buf, output->gpu_buf, vocab_size);

  CHECK_CUDA(cudaDeviceSynchronize());
}

void LayerNorm(Tensor *input, Tensor *weight, Tensor *bias, Tensor *output,
               float eps) {
  size_t rows = flat_rows(input);
  size_t cols = last_dim(input);
  CHECK_ERROR(weight->shape[0] == cols && bias->shape[0] == cols,
              "LayerNorm parameter shape mismatch");

#pragma omp parallel for
  for (size_t row = 0; row < rows; ++row) {
    const float *in = input->buf + row * cols;
    float *out = output->buf + row * cols;

    float mean = 0.0f;
    for (size_t col = 0; col < cols; ++col) {
      mean += in[col];
    }
    mean /= cols;

    float var = 0.0f;
    for (size_t col = 0; col < cols; ++col) {
      float diff = in[col] - mean;
      var += diff * diff;
    }
    var /= cols;

    float inv_std = rsqrtf(var + eps);
    for (size_t col = 0; col < cols; ++col) {
      out[col] = (in[col] - mean) * inv_std * weight->buf[col] + bias->buf[col];
    }
  }
}

__global__ void layer_norm_kernel(float* input, float* weight, float* bias, float* output, float eps) {
  size_t cols = blockDim.x;
  size_t row = blockIdx.x;
  size_t col = threadIdx.x;
  const float *in = input + row * cols;
  float *out = output + row * cols;
  __shared__ float L[1024];
  __shared__ float L2[1024];

  L[col] = in[col];
  L2[col] = in[col]*in[col];

  __syncthreads();
  
  if (col < (cols - 512)) {
    L[col] += L[col+512];
    L2[col] += L2[col+512];
  }
  __syncthreads();
  for (int i = 256; i > 0 ; i /= 2) {
    if (col < i) {
      L[col] += L[col+i];
      L2[col] += L2[col+i];
    }
    __syncthreads();
  }

  float mean = L[0]/cols;
  float var = L2[0]/cols - mean * mean;

  float inv_std = rsqrtf(var + eps);
  out[col] = (in[col] - mean) * inv_std * weight[col] + bias[col];
}

void LayerNorm_gpu(Tensor *input, Tensor *weight, Tensor *bias, Tensor *output,
                   float eps) {
  size_t rows = flat_rows(input);
  size_t cols = last_dim(input);

  dim3 gridDim(rows);
  dim3 blockDim(cols);
  layer_norm_kernel<<<gridDim, blockDim>>>(input->gpu_buf, weight->gpu_buf, bias->gpu_buf, output->gpu_buf, eps);
  CHECK_CUDA(cudaDeviceSynchronize());
}

void Linear(Tensor *input, Tensor *weight, Tensor *bias, Tensor *output) {
  size_t rows = flat_rows(input);
  size_t in_dim = last_dim(input);
  size_t out_dim = weight->shape[1];
  CHECK_ERROR(weight->shape[0] == in_dim, "Linear input dim mismatch");
  CHECK_ERROR(bias->shape[0] == out_dim, "Linear bias dim mismatch");
  CHECK_ERROR(output->num_elem() == rows * out_dim, "Linear output shape mismatch");

#pragma omp parallel for
  for (size_t row = 0; row < rows; ++row) {
    const float *in = input->buf + row * in_dim;
    float *out = output->buf + row * out_dim;
    for (size_t col = 0; col < out_dim; ++col) {
      float sum = bias->buf[col];
      for (size_t k = 0; k < in_dim; ++k) {
        sum += in[k] * weight->buf[k * out_dim + col];
      }
      out[col] = sum;
    }
  }
}

__global__ void linear_kernel(float* input, float* weight, float* bias, float* output, size_t in_dim) {
  size_t out_dim = gridDim.y;
  size_t row = blockIdx.x;
  size_t col = blockIdx.y;
  size_t idx = threadIdx.x;
  size_t block_size = blockDim.x;
  __shared__ float L[1024];
  const float *in = input + row * in_dim;
  float *out = output + row * out_dim;

  float sum = 0.0;
  for (int k = idx ; k < in_dim ; k += block_size) {
    sum += in[k] * weight[k * out_dim + col];
  }
  L[idx] = sum;
  __syncthreads();
  for (int i = 512 ; i > 0 ; i/=2 ) {
    if (idx < i) L[idx] += L[idx+i];
    __syncthreads();
  }
  if (idx == 0) out[col] = L[0] + bias[col];
}

void Linear_gpu(Tensor *input, Tensor *weight, Tensor *bias, Tensor *output) {
  size_t rows = flat_rows(input);
  size_t in_dim = last_dim(input);
  size_t out_dim = weight->shape[1];
  CHECK_ERROR(weight->shape[0] == in_dim, "Linear input dim mismatch");
  CHECK_ERROR(bias->shape[0] == out_dim, "Linear bias dim mismatch");
  CHECK_ERROR(output->num_elem() == rows * out_dim, "Linear output shape mismatch");

  dim3 gridDim(rows, out_dim);
  dim3 blockDim(1024);
  linear_kernel<<<gridDim, blockDim>>>(input->gpu_buf, weight->gpu_buf, bias->gpu_buf, output->gpu_buf, in_dim);

  CHECK_CUDA(cudaDeviceSynchronize());
}

void SplitQKV(Tensor *qkv, Tensor *q, Tensor *k, Tensor *v) {
  const size_t B = qkv->shape[0];
  const size_t T = qkv->shape[1];
  CHECK_ERROR(qkv->shape[2] == 3 * C, "QKV input size mismatch");

#pragma omp parallel for collapse(3)
  for (size_t b = 0; b < B; ++b) {
    for (size_t t = 0; t < T; ++t) {
      for (size_t h = 0; h < N_HEAD; ++h) {
        const size_t src_base = (b * T + t) * (3 * C) + h * HEAD_DIM;
        const size_t dst_base = ((b * N_HEAD + h) * T + t) * HEAD_DIM;
        for (size_t d = 0; d < HEAD_DIM; ++d) {
          q->buf[dst_base + d] = qkv->buf[src_base + d];
          k->buf[dst_base + d] = qkv->buf[src_base + C + d];
          v->buf[dst_base + d] = qkv->buf[src_base + 2 * C + d];
        }
      }
    }
  }
}

__global__ void split_qkv_kernel(float* qkv, float* q, float* k, float* v) {
  size_t T = gridDim.x;
  size_t b = blockIdx.y;
  size_t t = blockIdx.x;
  size_t h = threadIdx.y;
  size_t d = threadIdx.x;
  qkv += (b * T + t) * (3 * C) + h * HEAD_DIM;
  const size_t dst_base = ((b * N_HEAD + h) * T + t) * HEAD_DIM;
  q[dst_base + d] = qkv[d];
  k[dst_base + d] = qkv[C + d];
  v[dst_base + d] = qkv[2 * C + d];
}

void SplitQKV_gpu(Tensor *qkv, Tensor *q, Tensor *k, Tensor *v) {
  const size_t B = qkv->shape[0];
  const size_t T = qkv->shape[1];
  CHECK_ERROR(qkv->shape[2] == 3 * C, "QKV input size mismatch");

  dim3 gridDim(T, B);
  dim3 blockDim(HEAD_DIM, N_HEAD);
  split_qkv_kernel<<<gridDim, blockDim>>>(qkv->gpu_buf, q->gpu_buf, k->gpu_buf, v->gpu_buf);
  CHECK_CUDA(cudaDeviceSynchronize());
}

void AttentionScores(Tensor *q, Tensor *k, Tensor *scores) {
  const size_t B = q->shape[0];
  const size_t H = q->shape[1];
  const size_t T = q->shape[2];
  const size_t D = q->shape[3];

#pragma omp parallel for collapse(3)
  for (size_t b = 0; b < B; ++b) {
    for (size_t h = 0; h < H; ++h) {
      for (size_t tq = 0; tq < T; ++tq) {
        const size_t score_base = ((b * H + h) * T + tq) * T;
        const size_t q_base = ((b * H + h) * T + tq) * D;
        for (size_t tk = 0; tk < T; ++tk) {
          const size_t k_base = ((b * H + h) * T + tk) * D;
          float sum = 0.0f;
          for (size_t d = 0; d < D; ++d) {
            sum += q->buf[q_base + d] * k->buf[k_base + d];
          }
          scores->buf[score_base + tk] = sum;
        }
      }
    }
  }
}

__global__ void attention_scores_kernel(float* q, float* k, float* scores, size_t D) {
  size_t H = gridDim.x;
  size_t T = blockDim.x;
  size_t b = blockIdx.y;
  size_t h = blockIdx.x;
  size_t tk = threadIdx.x;
  size_t tq = threadIdx.y;

  scores += ((b * H + h) * T + tq) * T;
  q += ((b * H + h) * T + tq) * D;
  k += ((b * H + h) * T + tk) * D;
  float sum = 0.0f;
  for (size_t d = 0; d < D; ++d) {
    sum += q[d] * k[d];
  }
  scores[tk] = sum;
}
void AttentionScores_gpu(Tensor *q, Tensor *k, Tensor *scores) {
  const size_t B = q->shape[0];
  const size_t H = q->shape[1];
  const size_t T = q->shape[2];
  const size_t D = q->shape[3];

  dim3 gridDim(H,B);
  dim3 blockDim(T,T);
  attention_scores_kernel<<<gridDim, blockDim>>>(q->gpu_buf, k->gpu_buf, scores->gpu_buf, D);
  CHECK_CUDA(cudaDeviceSynchronize());
}

void ScaleMaskSoftmax(Tensor *scores, Tensor *probs) {
  const size_t B = scores->shape[0];
  const size_t H = scores->shape[1];
  const size_t T = scores->shape[2];
  const float scale = 1.0f / sqrtf((float)HEAD_DIM);
  // B: 1, H: 12, T: 8

#pragma omp parallel for collapse(3)
  for (size_t b = 0; b < B; ++b) {
    for (size_t h = 0; h < H; ++h) {
      for (size_t tq = 0; tq < T; ++tq) {
        const size_t row_base = ((b * H + h) * T + tq) * T;
        float row_max = -1e30f;
        for (size_t tk = 0; tk <= tq; ++tk) {
          float value = scores->buf[row_base + tk] * scale;
          row_max = fmaxf(row_max, value);
        }

        float sum = 0.0f;
        for (size_t tk = 0; tk < T; ++tk) {
          if (tk > tq) {
            probs->buf[row_base + tk] = 0.0f;
            continue;
          }
          float value = scores->buf[row_base + tk] * scale;
          float e = expf(value - row_max);
          probs->buf[row_base + tk] = e;
          sum += e;
        }

        for (size_t tk = 0; tk <= tq; ++tk) {
          probs->buf[row_base + tk] /= sum;
        }
      }
    }
  }
}

__global__ void scale_mask_softmax_kernel(float* scores, float* probs, float scale) {
  size_t H = blockDim.y;
  size_t T = blockDim.x;
  size_t b = blockIdx.x;
  size_t h = threadIdx.y;
  size_t tq = threadIdx.x;
  const size_t row_base = ((b * H + h) * T + tq) * T;
  scores += row_base;
  probs += row_base;
  float row_max = -1e30f;
  for (size_t tk = 0; tk <= tq; ++tk) {
    float value = scores[tk] * scale;
    row_max = fmaxf(row_max, value);
  }

  float sum = 0.0f;
  for (size_t tk = 0; tk < T; ++tk) {
    if (tk > tq) {
      probs[tk] = 0.0f;
      continue;
    }
    float value = scores[tk] * scale;
    float e = expf(value - row_max);
    probs[tk] = e;
    sum += e;
  }

  for (size_t tk = 0; tk <= tq; ++tk) {
    probs[tk] /= sum;
  }
}

void ScaleMaskSoftmax_gpu(Tensor *scores, Tensor *probs) {
  // B: 1, H: 12, T: 8
  const size_t B = scores->shape[0];
  const size_t H = scores->shape[1];
  const size_t T = scores->shape[2];
  const float scale = 1.0f / sqrtf((float)HEAD_DIM);
  dim3 gridDim(B);
  dim3 blockDim(T, H);
  scale_mask_softmax_kernel<<<gridDim, blockDim>>>(scores->gpu_buf, probs->gpu_buf, scale);
  CHECK_CUDA(cudaDeviceSynchronize());
}

void AttentionContext(Tensor *probs, Tensor *v, Tensor *context) {
  const size_t B = probs->shape[0];
  const size_t H = probs->shape[1];
  const size_t T = probs->shape[2];
  const size_t D = v->shape[3];

#pragma omp parallel for collapse(3)
  for (size_t b = 0; b < B; ++b) {
    for (size_t h = 0; h < H; ++h) {
      for (size_t tq = 0; tq < T; ++tq) {
        const size_t prob_base = ((b * H + h) * T + tq) * T;
        const size_t out_base = ((b * H + h) * T + tq) * D;
        for (size_t d = 0; d < D; ++d) {
          float sum = 0.0f;
          for (size_t tk = 0; tk < T; ++tk) {
            const size_t v_base = ((b * H + h) * T + tk) * D;
            sum += probs->buf[prob_base + tk] * v->buf[v_base + d];
          }
          context->buf[out_base + d] = sum;
        }
      }
    }
  }
}

__global__ void attention_context_kernel(float* probs, float* v, float* context) {
  size_t H = gridDim.x;
  size_t T = blockDim.y;
  size_t D = blockDim.x;
  size_t d = threadIdx.x;
  size_t tq = threadIdx.y;
  size_t h = blockIdx.x;
  size_t b = blockIdx.y;
  probs += ((b * H + h) * T + tq) * T;
  float sum = 0.0f;
  for (size_t tk = 0; tk < T; ++tk) {
    sum += probs[tk] * v[((b * H + h) * T + tk) * D + d];
  }
  context[((b * H + h) * T + tq) * D + d] = sum;
}
/*
B: 1
H: 12
T: 8
D: 64
*/
void AttentionContext_gpu(Tensor *probs, Tensor *v, Tensor *context) {
  const size_t B = probs->shape[0];
  const size_t H = probs->shape[1];
  const size_t T = probs->shape[2];
  const size_t D = v->shape[3];

  dim3 gridDim(H,B);
  dim3 blockDim(D,T);
  attention_context_kernel<<<gridDim, blockDim>>>(probs->gpu_buf, v->gpu_buf, context->gpu_buf);
  CHECK_CUDA(cudaDeviceSynchronize());
}

void MergeHeads(Tensor *context, Tensor *merged) {
  const size_t B = context->shape[0];
  const size_t H = context->shape[1];
  const size_t T = context->shape[2];
  const size_t D = context->shape[3];

#pragma omp parallel for collapse(3)
  for (size_t b = 0; b < B; ++b) {
    for (size_t t = 0; t < T; ++t) {
      for (size_t h = 0; h < H; ++h) {
        const size_t src_base = ((b * H + h) * T + t) * D;
        const size_t dst_base = (b * T + t) * C + h * D;
        memcpy(merged->buf + dst_base, context->buf + src_base,
               D * sizeof(float));
      }
    }
  }
}

__global__ void merge_heads_kernel(float* context, float* merged, size_t B, size_t H, size_t T, size_t D) {
  size_t n = blockDim.x * blockIdx.x + threadIdx.x;
  size_t N = B * H * T * D;
  size_t b = n / (H * T * D);
  size_t h = (n / (T * D)) % H;
  size_t t = (n / D) % T;
  size_t d = n % D;
  if (n >= N) return ;
  merged[(b * T * C) + (t * C ) + h * D + d] = context[n];
}
void MergeHeads_gpu(Tensor *context, Tensor *merged) {
  const size_t B = context->shape[0];
  const size_t H = context->shape[1];
  const size_t T = context->shape[2];
  const size_t D = context->shape[3];
  const size_t N = B * H * T * D; 

  dim3 gridDim((N+1023)/1024);
  dim3 blockDim(1024);
  merge_heads_kernel<<<gridDim, blockDim>>>(context->gpu_buf, merged->gpu_buf, B, H, T, D);
  CHECK_CUDA(cudaDeviceSynchronize());
}

void ResidualAdd(Tensor *input, Tensor *addend, Tensor *output) {
  CHECK_ERROR(input->num_elem() == addend->num_elem() &&
                  input->num_elem() == output->num_elem(),
              "ResidualAdd shape mismatch");
#pragma omp parallel for
  for (size_t i = 0; i < input->num_elem(); ++i) {
    output->buf[i] = input->buf[i] + addend->buf[i];
  }
}

__global__ void residual_add_kernel(float* input, float* addend, float* output, size_t N) {
  size_t n = blockDim.x * blockIdx.x + threadIdx.x;
  if (n >= N) return;
  output[n] = input[n] + addend[n];
}
void ResidualAdd_gpu(Tensor *input, Tensor *addend, Tensor *output) {
  CHECK_ERROR(input->num_elem() == addend->num_elem() &&
                  input->num_elem() == output->num_elem(),
              "ResidualAdd shape mismatch");
  size_t N = input->num_elem();
  dim3 gridDim((N+1023)/1024);
  dim3 blockDim(1024);
  residual_add_kernel<<<gridDim, blockDim>>>(input->gpu_buf, addend->gpu_buf, output->gpu_buf, N);
  CHECK_CUDA(cudaDeviceSynchronize());
}

void GELUNew(Tensor *inout) {
  const float kAlpha = sqrtf(2.0f / 3.14159265358979323846f);

#pragma omp parallel for
  for (size_t i = 0; i < inout->num_elem(); ++i) {
    float x = inout->buf[i];
    float x3 = x * x * x;
    inout->buf[i] =
        0.5f * x * (1.0f + tanhf(kAlpha * (x + 0.044715f * x3)));
  }
}

__global__ void gelu_new_kernel(float* inout, float kAlpha, size_t N) {
  size_t n = blockDim.x * blockIdx.x + threadIdx.x;
  if (n >= N) return;
  float x = inout[n];
  float x3 = x * x * x;
  inout[n] = 0.5f * x * (1.0f + tanhf(kAlpha * (x + 0.044715f * x3)));
}

void GELUNew_gpu(Tensor *inout) {
  const float kAlpha = sqrtf(2.0f / 3.14159265358979323846f);

  size_t N = inout->num_elem();
  dim3 gridDim((N+1023)/1024);
  dim3 blockDim(1024);
  gelu_new_kernel<<<gridDim, blockDim>>>(inout->gpu_buf, kAlpha, N);
  CHECK_CUDA(cudaDeviceSynchronize());
}

void LMHead(Tensor *input, Tensor *embedding, Tensor *output) {
  size_t rows = flat_rows(input);
  size_t hidden = last_dim(input);
  CHECK_ERROR(hidden == C, "LMHead hidden size mismatch");
  CHECK_ERROR(embedding->shape[0] == VOCAB_SIZE && embedding->shape[1] == C,
              "Embedding table shape mismatch");
  CHECK_ERROR(output->num_elem() == rows * VOCAB_SIZE,
              "LMHead output shape mismatch");

#pragma omp parallel for
  for (size_t row = 0; row < rows; ++row) {
    const float *in = input->buf + row * hidden;
    float *out = output->buf + row * VOCAB_SIZE;
    for (size_t vocab = 0; vocab < VOCAB_SIZE; ++vocab) {
      const float *emb = embedding->buf + vocab * hidden;
      float sum = 0.0f;
      for (size_t c = 0; c < hidden; ++c) {
        sum += in[c] * emb[c];
      }
      out[vocab] = sum;
    }
  }
}

__global__ void lmhead_kernel(float *input, float *embedding, float *output, size_t hidden) {
  size_t vocab = blockIdx.x * blockDim.x + threadIdx.x;
  size_t row = blockIdx.y;
  if (vocab >= VOCAB_SIZE) return; 
  input += row * hidden;
  output += row * VOCAB_SIZE;
  embedding += vocab * hidden;
  float sum = 0.0f;
  for (size_t h = 0; h < hidden; ++h) {
    sum += input[h] * embedding[h];
  }
  output[vocab] = sum;
}

void LMHead_gpu(Tensor *input, Tensor *embedding, Tensor *output) {
  size_t rows = flat_rows(input);
  size_t hidden = last_dim(input);
  CHECK_ERROR(hidden == C, "LMHead hidden size mismatch");
  CHECK_ERROR(embedding->shape[0] == VOCAB_SIZE && embedding->shape[1] == C,
              "Embedding table shape mismatch");
  CHECK_ERROR(output->num_elem() == rows * VOCAB_SIZE,
              "LMHead output shape mismatch");

  dim3 gridDim((VOCAB_SIZE+1023)/1024, rows);
  dim3 blockDim(1024);
  lmhead_kernel<<<gridDim, blockDim>>>(input->gpu_buf, embedding->gpu_buf, output->gpu_buf, hidden);
  CHECK_CUDA(cudaDeviceSynchronize());
}
