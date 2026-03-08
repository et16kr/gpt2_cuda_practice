#pragma once

#include "gpt2.h"
#include "tensor.h"

void EmbeddingPositionAdd(TokenBatch *tokens, Tensor *wte, Tensor *wpe,
                          Tensor *output);
void EmbeddingPositionAdd_gpu(TokenBatch *tokens, Tensor *wte, Tensor *wpe,
                              Tensor *output);

void LayerNorm(Tensor *input, Tensor *weight, Tensor *bias, Tensor *output,
               float eps);
void LayerNorm_gpu(Tensor *input, Tensor *weight, Tensor *bias, Tensor *output,
                   float eps);

void Linear(Tensor *input, Tensor *weight, Tensor *bias, Tensor *output);
void Linear_gpu(Tensor *input, Tensor *weight, Tensor *bias, Tensor *output);

void SplitQKV(Tensor *qkv, Tensor *q, Tensor *k, Tensor *v);
void SplitQKV_gpu(Tensor *qkv, Tensor *q, Tensor *k, Tensor *v);

void AttentionScores(Tensor *q, Tensor *k, Tensor *scores);
void AttentionScores_gpu(Tensor *q, Tensor *k, Tensor *scores);

void ScaleMaskSoftmax(Tensor *scores, Tensor *probs);
void ScaleMaskSoftmax_gpu(Tensor *scores, Tensor *probs);

void AttentionContext(Tensor *probs, Tensor *v, Tensor *context);
void AttentionContext_gpu(Tensor *probs, Tensor *v, Tensor *context);

void MergeHeads(Tensor *context, Tensor *merged);
void MergeHeads_gpu(Tensor *context, Tensor *merged);

void ResidualAdd(Tensor *input, Tensor *addend, Tensor *output);
void ResidualAdd_gpu(Tensor *input, Tensor *addend, Tensor *output);

void GELUNew(Tensor *inout);
void GELUNew_gpu(Tensor *inout);

void LMHead(Tensor *input, Tensor *embedding, Tensor *output);
void LMHead_gpu(Tensor *input, Tensor *embedding, Tensor *output);
