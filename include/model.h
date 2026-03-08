#pragma once

#include "tensor.h"

TokenBatch load_tokens(const char *path);
void initialize_model(const char *safetensors_path);
void alloc_activations(size_t batch_size, size_t seq_len);
void gpt2_forward(TokenBatch *tokens, Tensor *logits);
void validate_against_cpu(TokenBatch *tokens, Tensor *logits_gpu);
void finalize_model();
void free_activations();
