#include "model.h"

#include <cstdio>
#include <string>
#include <utility>

#include "gpt2.h"
#include "layer.h"
#include "safetensors_loader.h"
#include "util.h"

namespace {

/* [Model Parameters] */
Parameter *wte_weight = nullptr;
Parameter *wpe_weight = nullptr;
Parameter *ln1_weight[N_LAYER], *ln1_bias[N_LAYER];
Parameter *attn_c_attn_weight[N_LAYER], *attn_c_attn_bias[N_LAYER];
Parameter *attn_c_proj_weight[N_LAYER], *attn_c_proj_bias[N_LAYER];
Parameter *ln2_weight[N_LAYER], *ln2_bias[N_LAYER];
Parameter *mlp_c_fc_weight[N_LAYER], *mlp_c_fc_bias[N_LAYER];
Parameter *mlp_c_proj_weight[N_LAYER], *mlp_c_proj_bias[N_LAYER];
Parameter *ln_f_weight = nullptr;
Parameter *ln_f_bias = nullptr;

/* [Activations] */
Activation *x = nullptr;
Activation *residual = nullptr;
Activation *ln_buf = nullptr;
Activation *qkv = nullptr;
Activation *q = nullptr;
Activation *k = nullptr;
Activation *v = nullptr;
Activation *att_scores = nullptr;
Activation *att_probs = nullptr;
Activation *context = nullptr;
Activation *merged = nullptr;
Activation *attn_proj = nullptr;
Activation *mlp_hidden = nullptr;
Activation *mlp_out = nullptr;

size_t current_batch = 0;
size_t current_seq = 0;

void delete_tensor(Tensor *&tensor) {
  if (tensor != nullptr) {
    delete tensor;
    tensor = nullptr;
  }
}

template <size_t N>
void delete_tensor_array(Tensor *(&tensors)[N]) {
  for (size_t i = 0; i < N; ++i) {
    delete_tensor(tensors[i]);
  }
}

void load_transformer_block_parameters(SafetensorsLoader *loader,
                                       size_t layer_idx) {
  std::string prefix = "h." + std::to_string(layer_idx) + ".";

  ln1_weight[layer_idx] =
      loader->load_parameter((prefix + "ln_1.weight").c_str(), {C});
  ln1_bias[layer_idx] =
      loader->load_parameter((prefix + "ln_1.bias").c_str(), {C});
  attn_c_attn_weight[layer_idx] =
      loader->load_parameter((prefix + "attn.c_attn.weight").c_str(), {C, 3 * C});
  attn_c_attn_bias[layer_idx] =
      loader->load_parameter((prefix + "attn.c_attn.bias").c_str(), {3 * C});
  attn_c_proj_weight[layer_idx] =
      loader->load_parameter((prefix + "attn.c_proj.weight").c_str(), {C, C});
  attn_c_proj_bias[layer_idx] =
      loader->load_parameter((prefix + "attn.c_proj.bias").c_str(), {C});
  ln2_weight[layer_idx] =
      loader->load_parameter((prefix + "ln_2.weight").c_str(), {C});
  ln2_bias[layer_idx] =
      loader->load_parameter((prefix + "ln_2.bias").c_str(), {C});
  mlp_c_fc_weight[layer_idx] =
      loader->load_parameter((prefix + "mlp.c_fc.weight").c_str(), {C, MLP_DIM});
  mlp_c_fc_bias[layer_idx] =
      loader->load_parameter((prefix + "mlp.c_fc.bias").c_str(), {MLP_DIM});
  mlp_c_proj_weight[layer_idx] =
      loader->load_parameter((prefix + "mlp.c_proj.weight").c_str(), {MLP_DIM, C});
  mlp_c_proj_bias[layer_idx] =
      loader->load_parameter((prefix + "mlp.c_proj.bias").c_str(), {C});
}

void alloc_and_set_parameters(const char *safetensors_path) {
  SafetensorsLoader loader(safetensors_path);

  wte_weight = loader.load_parameter("wte.weight", {VOCAB_SIZE, C});
  wpe_weight = loader.load_parameter("wpe.weight", {MAX_T, C});

  for (size_t layer = 0; layer < N_LAYER; ++layer) {
    load_transformer_block_parameters(&loader, layer);
  }

  ln_f_weight = loader.load_parameter("ln_f.weight", {C});
  ln_f_bias = loader.load_parameter("ln_f.bias", {C});
}

void free_parameters() {
  delete_tensor(wte_weight);
  delete_tensor(wpe_weight);
  delete_tensor_array(ln1_weight);
  delete_tensor_array(ln1_bias);
  delete_tensor_array(attn_c_attn_weight);
  delete_tensor_array(attn_c_attn_bias);
  delete_tensor_array(attn_c_proj_weight);
  delete_tensor_array(attn_c_proj_bias);
  delete_tensor_array(ln2_weight);
  delete_tensor_array(ln2_bias);
  delete_tensor_array(mlp_c_fc_weight);
  delete_tensor_array(mlp_c_fc_bias);
  delete_tensor_array(mlp_c_proj_weight);
  delete_tensor_array(mlp_c_proj_bias);
  delete_tensor(ln_f_weight);
  delete_tensor(ln_f_bias);
}

void transformer_block(size_t layer_idx) {
  LayerNorm(x, ln1_weight[layer_idx], ln1_bias[layer_idx], ln_buf, LN_EPS);
  Linear(ln_buf, attn_c_attn_weight[layer_idx], attn_c_attn_bias[layer_idx],
         qkv);
  SplitQKV(qkv, q, k, v);
  AttentionScores(q, k, att_scores);
  ScaleMaskSoftmax(att_scores, att_probs);
  AttentionContext(att_probs, v, context);
  MergeHeads(context, merged);
  Linear(merged, attn_c_proj_weight[layer_idx], attn_c_proj_bias[layer_idx],
         attn_proj);
  ResidualAdd(x, attn_proj, residual);
  std::swap(x, residual);

  LayerNorm(x, ln2_weight[layer_idx], ln2_bias[layer_idx], ln_buf, LN_EPS);
  Linear(ln_buf, mlp_c_fc_weight[layer_idx], mlp_c_fc_bias[layer_idx],
         mlp_hidden);
  GELUNew(mlp_hidden);
  Linear(mlp_hidden, mlp_c_proj_weight[layer_idx], mlp_c_proj_bias[layer_idx],
         mlp_out);
  ResidualAdd(x, mlp_out, residual);
  std::swap(x, residual);
}

void transformer_block_gpu(size_t layer_idx) {
  LayerNorm_gpu(x, ln1_weight[layer_idx], ln1_bias[layer_idx], ln_buf, LN_EPS);
  Linear_gpu(ln_buf, attn_c_attn_weight[layer_idx], attn_c_attn_bias[layer_idx], qkv);
  SplitQKV_gpu(qkv, q, k, v);
  AttentionScores_gpu(q, k, att_scores);
  ScaleMaskSoftmax_gpu(att_scores, att_probs);
  AttentionContext_gpu(att_probs, v, context);
  MergeHeads_gpu(context, merged);
  Linear_gpu(merged, attn_c_proj_weight[layer_idx], attn_c_proj_bias[layer_idx],
         attn_proj);
  ResidualAdd_gpu(x, attn_proj, residual);
  std::swap(x, residual);

  LayerNorm_gpu(x, ln2_weight[layer_idx], ln2_bias[layer_idx], ln_buf, LN_EPS);
  Linear_gpu(ln_buf, mlp_c_fc_weight[layer_idx], mlp_c_fc_bias[layer_idx],
         mlp_hidden);
  GELUNew_gpu(mlp_hidden);
  Linear_gpu(mlp_hidden, mlp_c_proj_weight[layer_idx], mlp_c_proj_bias[layer_idx],
         mlp_out);
  ResidualAdd_gpu(x, mlp_out, residual);
  std::swap(x, residual);
}

void gpt2_forward_cpu(TokenBatch *tokens, Tensor *logits) {
  CHECK_ERROR(tokens->B == current_batch && tokens->T == current_seq,
              "Token batch shape differs from allocated activations");
  CHECK_ERROR(logits->shape[0] == tokens->B && logits->shape[1] == tokens->T &&
                  logits->shape[2] == VOCAB_SIZE,
              "Logits tensor shape mismatch");

  EmbeddingPositionAdd(tokens, wte_weight, wpe_weight, x);

  for (size_t layer = 0; layer < N_LAYER; ++layer) {
    transformer_block(layer);
  }

  LayerNorm(x, ln_f_weight, ln_f_bias, ln_buf, LN_EPS);
  LMHead(ln_buf, wte_weight, logits);
}

void gpt2_forward_gpu(TokenBatch *tokens, Tensor *logits) {
  CHECK_ERROR(tokens->B == current_batch && tokens->T == current_seq,
              "Token batch shape differs from allocated activations");
  CHECK_ERROR(logits->shape[0] == tokens->B && logits->shape[1] == tokens->T &&
                  logits->shape[2] == VOCAB_SIZE,
              "Logits tensor shape mismatch");

  EmbeddingPositionAdd_gpu(tokens, wte_weight, wpe_weight, x);

  for (size_t layer = 0; layer < N_LAYER; ++layer) {
    transformer_block_gpu(layer);
  }

  LayerNorm_gpu(x, ln_f_weight, ln_f_bias, ln_buf, LN_EPS);
  LMHead_gpu(ln_buf, wte_weight, logits);
}

}  // namespace

TokenBatch load_tokens(const char *path) {
  FILE *f = fopen(path, "rb");
  CHECK_ERROR(f != nullptr, "Failed to open token file %s", path);

  int32_t B = 0;
  int32_t T = 0;
  CHECK_ERROR(fread(&B, sizeof(int32_t), 1, f) == 1,
              "Failed to read batch size from %s", path);
  CHECK_ERROR(fread(&T, sizeof(int32_t), 1, f) == 1,
              "Failed to read sequence length from %s", path);
  CHECK_ERROR(B > 0 && T > 0, "Invalid token shape in %s", path);
  CHECK_ERROR(T <= (int32_t)MAX_T, "Sequence length %d exceeds MAX_T", T);

  TokenBatch batch((size_t)B, (size_t)T);
  size_t expected = (size_t)B * (size_t)T;
  CHECK_ERROR(fread(batch.buf, sizeof(int32_t), expected, f) == expected,
              "Failed to read token ids from %s", path);
  int trailing = fgetc(f);
  fclose(f);
  CHECK_ERROR(trailing == EOF, "Unexpected trailing bytes in token file %s",
              path);
  batch.to_gpu();
  return batch;
}

void initialize_model(const char *safetensors_path) {
  alloc_and_set_parameters(safetensors_path);
}

void alloc_activations(size_t batch_size, size_t seq_len) {
  CHECK_ERROR(batch_size > 0 && seq_len > 0, "Activation shape must be positive");
  CHECK_ERROR(seq_len <= MAX_T, "Sequence length %zu exceeds MAX_T", seq_len);

  free_activations();

  current_batch = batch_size;
  current_seq = seq_len;

  x = new Activation({batch_size, seq_len, C});
  residual = new Activation({batch_size, seq_len, C});
  ln_buf = new Activation({batch_size, seq_len, C});
  qkv = new Activation({batch_size, seq_len, 3 * C});
  q = new Activation({batch_size, N_HEAD, seq_len, HEAD_DIM});
  k = new Activation({batch_size, N_HEAD, seq_len, HEAD_DIM});
  v = new Activation({batch_size, N_HEAD, seq_len, HEAD_DIM});
  att_scores = new Activation({batch_size, N_HEAD, seq_len, seq_len});
  att_probs = new Activation({batch_size, N_HEAD, seq_len, seq_len});
  context = new Activation({batch_size, N_HEAD, seq_len, HEAD_DIM});
  merged = new Activation({batch_size, seq_len, C});
  attn_proj = new Activation({batch_size, seq_len, C});
  mlp_hidden = new Activation({batch_size, seq_len, MLP_DIM});
  mlp_out = new Activation({batch_size, seq_len, C});
}

void gpt2_forward(TokenBatch *tokens, Tensor *logits) {
  gpt2_forward_gpu(tokens, logits);

  // TODO(student): Replace the CPU path with GPU kernels layer by layer.
  CHECK_CUDA(cudaDeviceSynchronize());
}

void validate_against_cpu(TokenBatch *tokens, Tensor *logits_gpu) {
  Tensor reference({tokens->B, tokens->T, VOCAB_SIZE});
  gpt2_forward_cpu(tokens, &reference);

  int diff = validate_buffer(logits_gpu->buf, reference.buf, reference.num_elem(),
                             1e-3f, 1e-3f);
  if (diff < 0) {
    printf("Validation: PASSED\n");
    return;
  }

  printf("Validation: FAILED\n");
  printf("First mismatch at index %d: output=%f reference=%f\n", diff,
         logits_gpu->buf[diff], reference.buf[diff]);
  EXIT(EXIT_FAILURE);
}

void finalize_model() {
  free_parameters();
}

void free_activations() {
  delete_tensor(x);
  delete_tensor(residual);
  delete_tensor(ln_buf);
  delete_tensor(qkv);
  delete_tensor(q);
  delete_tensor(k);
  delete_tensor(v);
  delete_tensor(att_scores);
  delete_tensor(att_probs);
  delete_tensor(context);
  delete_tensor(merged);
  delete_tensor(attn_proj);
  delete_tensor(mlp_hidden);
  delete_tensor(mlp_out);
  current_batch = 0;
  current_seq = 0;
}
