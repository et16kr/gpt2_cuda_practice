#include "layer.h"

#include <cmath>
#include <cstring>

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

void EmbeddingPositionAdd_gpu(TokenBatch *tokens, Tensor *wte, Tensor *wpe,
                              Tensor *output) {
  EmbeddingPositionAdd(tokens, wte, wpe, output);

  // TODO(student): Move token ids and embedding tables to GPU and replace the
  // CPU reference loop with a CUDA kernel.
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

void LayerNorm_gpu(Tensor *input, Tensor *weight, Tensor *bias, Tensor *output,
                   float eps) {
  LayerNorm(input, weight, bias, output, eps);

  // TODO(student): Implement row-wise mean/variance reduction on GPU.
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

void Linear_gpu(Tensor *input, Tensor *weight, Tensor *bias, Tensor *output) {
  Linear(input, weight, bias, output);

  // TODO(student): Replace the CPU reference GEMM with CUDA kernel(s) or
  // cuBLAS while keeping the same tensor layout.
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

void SplitQKV_gpu(Tensor *qkv, Tensor *q, Tensor *k, Tensor *v) {
  SplitQKV(qkv, q, k, v);

  // TODO(student): Implement a layout transform kernel for q/k/v split.
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

void AttentionScores_gpu(Tensor *q, Tensor *k, Tensor *scores) {
  AttentionScores(q, k, scores);

  // TODO(student): Replace the dot-product loops with a CUDA kernel.
  CHECK_CUDA(cudaDeviceSynchronize());
}

void ScaleMaskSoftmax(Tensor *scores, Tensor *probs) {
  const size_t B = scores->shape[0];
  const size_t H = scores->shape[1];
  const size_t T = scores->shape[2];
  const float scale = 1.0f / sqrtf((float)HEAD_DIM);

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

void ScaleMaskSoftmax_gpu(Tensor *scores, Tensor *probs) {
  ScaleMaskSoftmax(scores, probs);

  // TODO(student): Fuse scaling, causal masking, and softmax on GPU.
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

void AttentionContext_gpu(Tensor *probs, Tensor *v, Tensor *context) {
  AttentionContext(probs, v, context);

  // TODO(student): Implement the weighted value accumulation on GPU.
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

void MergeHeads_gpu(Tensor *context, Tensor *merged) {
  MergeHeads(context, merged);

  // TODO(student): Implement the head merge layout transform on GPU.
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

void ResidualAdd_gpu(Tensor *input, Tensor *addend, Tensor *output) {
  ResidualAdd(input, addend, output);

  // TODO(student): Replace elementwise add with a CUDA kernel.
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

void GELUNew_gpu(Tensor *inout) {
  GELUNew(inout);

  // TODO(student): Implement GPT-2 gelu_new activation on GPU.
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

void LMHead_gpu(Tensor *input, Tensor *embedding, Tensor *output) {
  LMHead(input, embedding, output);

  // TODO(student): Replace the vocab projection with GPU code.
  CHECK_CUDA(cudaDeviceSynchronize());
}
