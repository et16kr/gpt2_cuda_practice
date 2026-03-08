#include "model.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "gpt2.h"
#include "layer.h"
#include "util.h"

namespace {

struct TensorInfo {
  std::string dtype;
  std::vector<size_t> shape;
  size_t begin = 0;
  size_t end = 0;
};

struct BlockParameters {
  Parameter *ln1_weight = nullptr;
  Parameter *ln1_bias = nullptr;
  Parameter *attn_c_attn_weight = nullptr;
  Parameter *attn_c_attn_bias = nullptr;
  Parameter *attn_c_proj_weight = nullptr;
  Parameter *attn_c_proj_bias = nullptr;
  Parameter *ln2_weight = nullptr;
  Parameter *ln2_bias = nullptr;
  Parameter *mlp_c_fc_weight = nullptr;
  Parameter *mlp_c_fc_bias = nullptr;
  Parameter *mlp_c_proj_weight = nullptr;
  Parameter *mlp_c_proj_bias = nullptr;
};

Parameter *wte_weight = nullptr;
Parameter *wpe_weight = nullptr;
Parameter *ln_f_weight = nullptr;
Parameter *ln_f_bias = nullptr;
BlockParameters blocks[N_LAYER];

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

void skip_ws(const std::string &text, size_t *pos) {
  while (*pos < text.size() &&
         std::isspace(static_cast<unsigned char>(text[*pos]))) {
    ++(*pos);
  }
}

std::string parse_quoted(const std::string &text, size_t *pos) {
  skip_ws(text, pos);
  CHECK_ERROR(*pos < text.size() && text[*pos] == '"',
              "Invalid JSON string at offset %zu", *pos);
  ++(*pos);
  size_t start = *pos;
  while (*pos < text.size() && text[*pos] != '"') {
    CHECK_ERROR(text[*pos] != '\\', "Escaped JSON strings are not supported");
    ++(*pos);
  }
  CHECK_ERROR(*pos < text.size(), "Unterminated JSON string");
  std::string value = text.substr(start, *pos - start);
  ++(*pos);
  return value;
}

size_t parse_uint(const std::string &text, size_t *pos) {
  skip_ws(text, pos);
  CHECK_ERROR(*pos < text.size() &&
                  std::isdigit(static_cast<unsigned char>(text[*pos])),
              "Expected unsigned integer at offset %zu", *pos);
  size_t value = 0;
  while (*pos < text.size() &&
         std::isdigit(static_cast<unsigned char>(text[*pos]))) {
    value = value * 10 + (size_t)(text[*pos] - '0');
    ++(*pos);
  }
  return value;
}

std::vector<size_t> parse_uint_array(const std::string &text, size_t *pos) {
  skip_ws(text, pos);
  CHECK_ERROR(*pos < text.size() && text[*pos] == '[',
              "Expected array at offset %zu", *pos);
  ++(*pos);

  std::vector<size_t> values;
  while (true) {
    skip_ws(text, pos);
    CHECK_ERROR(*pos < text.size(), "Unterminated array");
    if (text[*pos] == ']') {
      ++(*pos);
      break;
    }
    values.push_back(parse_uint(text, pos));
    skip_ws(text, pos);
    CHECK_ERROR(*pos < text.size(), "Unterminated array");
    if (text[*pos] == ',') {
      ++(*pos);
      continue;
    }
    CHECK_ERROR(text[*pos] == ']', "Expected ',' or ']' in array");
  }
  return values;
}

TensorInfo parse_tensor_info(const std::string &header, const std::string &name) {
  std::string needle = "\"" + name + "\"";
  size_t key_pos = header.find(needle);
  CHECK_ERROR(key_pos != std::string::npos, "Missing tensor %s in safetensors",
              name.c_str());

  size_t pos = key_pos + needle.size();
  skip_ws(header, &pos);
  CHECK_ERROR(pos < header.size() && header[pos] == ':',
              "Malformed tensor entry for %s", name.c_str());
  ++pos;
  skip_ws(header, &pos);
  CHECK_ERROR(pos < header.size() && header[pos] == '{',
              "Expected object for %s", name.c_str());
  ++pos;

  TensorInfo info;
  while (true) {
    skip_ws(header, &pos);
    CHECK_ERROR(pos < header.size(), "Unterminated tensor object for %s",
                name.c_str());
    if (header[pos] == '}') {
      ++pos;
      break;
    }

    std::string field = parse_quoted(header, &pos);
    skip_ws(header, &pos);
    CHECK_ERROR(pos < header.size() && header[pos] == ':',
                "Expected ':' in tensor object for %s", name.c_str());
    ++pos;

    if (field == "dtype") {
      info.dtype = parse_quoted(header, &pos);
    } else if (field == "shape") {
      info.shape = parse_uint_array(header, &pos);
    } else if (field == "data_offsets") {
      std::vector<size_t> offsets = parse_uint_array(header, &pos);
      CHECK_ERROR(offsets.size() == 2,
                  "data_offsets must have size 2 for tensor %s", name.c_str());
      info.begin = offsets[0];
      info.end = offsets[1];
    } else {
      CHECK_ERROR(false, "Unsupported field %s while parsing %s", field.c_str(),
                  name.c_str());
    }

    skip_ws(header, &pos);
    CHECK_ERROR(pos < header.size(), "Unterminated tensor object for %s",
                name.c_str());
    if (header[pos] == ',') {
      ++pos;
    } else {
      CHECK_ERROR(header[pos] == '}', "Malformed tensor object for %s",
                  name.c_str());
    }
  }

  CHECK_ERROR(info.dtype == "F32", "Tensor %s must be F32, got %s", name.c_str(),
              info.dtype.c_str());
  CHECK_ERROR(!info.shape.empty(), "Tensor %s has empty shape", name.c_str());
  CHECK_ERROR(info.end >= info.begin, "Tensor %s has invalid offsets",
              name.c_str());
  return info;
}

Parameter *load_parameter(FILE *f, size_t data_base, const std::string &header,
                          const std::string &name,
                          const std::vector<size_t> &expected_shape) {
  TensorInfo info = parse_tensor_info(header, name);
  CHECK_ERROR(info.shape == expected_shape,
              "Tensor %s shape mismatch while loading parameter", name.c_str());

  size_t numel = 1;
  for (size_t dim : info.shape) {
    numel *= dim;
  }
  CHECK_ERROR((info.end - info.begin) == numel * sizeof(float),
              "Tensor %s byte size mismatch", name.c_str());

  Parameter *param = new Parameter(expected_shape);
  CHECK_ERROR(fseek(f, (long)(data_base + info.begin), SEEK_SET) == 0,
              "Failed to seek to tensor %s", name.c_str());
  size_t ret = fread(param->buf, sizeof(float), numel, f);
  CHECK_ERROR(ret == numel, "Failed to read tensor %s", name.c_str());
  param->to_gpu();
  return param;
}

void load_block_parameters(FILE *f, size_t data_base, const std::string &header,
                           size_t layer_idx) {
  std::string prefix = "h." + std::to_string(layer_idx) + ".";
  BlockParameters &block = blocks[layer_idx];

  block.ln1_weight =
      load_parameter(f, data_base, header, prefix + "ln_1.weight", {C});
  block.ln1_bias =
      load_parameter(f, data_base, header, prefix + "ln_1.bias", {C});
  block.attn_c_attn_weight = load_parameter(
      f, data_base, header, prefix + "attn.c_attn.weight", {C, 3 * C});
  block.attn_c_attn_bias =
      load_parameter(f, data_base, header, prefix + "attn.c_attn.bias", {3 * C});
  block.attn_c_proj_weight =
      load_parameter(f, data_base, header, prefix + "attn.c_proj.weight", {C, C});
  block.attn_c_proj_bias =
      load_parameter(f, data_base, header, prefix + "attn.c_proj.bias", {C});
  block.ln2_weight =
      load_parameter(f, data_base, header, prefix + "ln_2.weight", {C});
  block.ln2_bias =
      load_parameter(f, data_base, header, prefix + "ln_2.bias", {C});
  block.mlp_c_fc_weight = load_parameter(
      f, data_base, header, prefix + "mlp.c_fc.weight", {C, MLP_DIM});
  block.mlp_c_fc_bias =
      load_parameter(f, data_base, header, prefix + "mlp.c_fc.bias", {MLP_DIM});
  block.mlp_c_proj_weight = load_parameter(
      f, data_base, header, prefix + "mlp.c_proj.weight", {MLP_DIM, C});
  block.mlp_c_proj_bias =
      load_parameter(f, data_base, header, prefix + "mlp.c_proj.bias", {C});
}

void free_block_parameters(BlockParameters &block) {
  delete_tensor(block.ln1_weight);
  delete_tensor(block.ln1_bias);
  delete_tensor(block.attn_c_attn_weight);
  delete_tensor(block.attn_c_attn_bias);
  delete_tensor(block.attn_c_proj_weight);
  delete_tensor(block.attn_c_proj_bias);
  delete_tensor(block.ln2_weight);
  delete_tensor(block.ln2_bias);
  delete_tensor(block.mlp_c_fc_weight);
  delete_tensor(block.mlp_c_fc_bias);
  delete_tensor(block.mlp_c_proj_weight);
  delete_tensor(block.mlp_c_proj_bias);
}

void gpt2_forward_cpu(TokenBatch *tokens, Tensor *logits) {
  CHECK_ERROR(tokens->B == current_batch && tokens->T == current_seq,
              "Token batch shape differs from allocated activations");
  CHECK_ERROR(logits->shape[0] == tokens->B && logits->shape[1] == tokens->T &&
                  logits->shape[2] == VOCAB_SIZE,
              "Logits tensor shape mismatch");

  EmbeddingPositionAdd(tokens, wte_weight, wpe_weight, x);

  for (size_t layer = 0; layer < N_LAYER; ++layer) {
    BlockParameters &block = blocks[layer];

    LayerNorm(x, block.ln1_weight, block.ln1_bias, ln_buf, LN_EPS);
    Linear(ln_buf, block.attn_c_attn_weight, block.attn_c_attn_bias, qkv);
    SplitQKV(qkv, q, k, v);
    AttentionScores(q, k, att_scores);
    ScaleMaskSoftmax(att_scores, att_probs);
    AttentionContext(att_probs, v, context);
    MergeHeads(context, merged);
    Linear(merged, block.attn_c_proj_weight, block.attn_c_proj_bias, attn_proj);
    ResidualAdd(x, attn_proj, residual);
    std::swap(x, residual);

    LayerNorm(x, block.ln2_weight, block.ln2_bias, ln_buf, LN_EPS);
    Linear(ln_buf, block.mlp_c_fc_weight, block.mlp_c_fc_bias, mlp_hidden);
    GELUNew(mlp_hidden);
    Linear(mlp_hidden, block.mlp_c_proj_weight, block.mlp_c_proj_bias, mlp_out);
    ResidualAdd(x, mlp_out, residual);
    std::swap(x, residual);
  }

  LayerNorm(x, ln_f_weight, ln_f_bias, ln_buf, LN_EPS);
  LMHead(ln_buf, wte_weight, logits);
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
  FILE *f = fopen(safetensors_path, "rb");
  CHECK_ERROR(f != nullptr, "Failed to open model file %s", safetensors_path);

  uint64_t header_len = 0;
  CHECK_ERROR(fread(&header_len, sizeof(uint64_t), 1, f) == 1,
              "Failed to read safetensors header length");

  std::string header(header_len, '\0');
  CHECK_ERROR(fread(&header[0], 1, header_len, f) == header_len,
              "Failed to read safetensors header");

  size_t data_base = sizeof(uint64_t) + (size_t)header_len;

  wte_weight = load_parameter(f, data_base, header, "wte.weight", {VOCAB_SIZE, C});
  wpe_weight = load_parameter(f, data_base, header, "wpe.weight", {MAX_T, C});
  for (size_t layer = 0; layer < N_LAYER; ++layer) {
    load_block_parameters(f, data_base, header, layer);
  }
  ln_f_weight = load_parameter(f, data_base, header, "ln_f.weight", {C});
  ln_f_bias = load_parameter(f, data_base, header, "ln_f.bias", {C});

  fclose(f);
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
  gpt2_forward_cpu(tokens, logits);

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
  delete_tensor(wte_weight);
  delete_tensor(wpe_weight);
  for (size_t layer = 0; layer < N_LAYER; ++layer) {
    free_block_parameters(blocks[layer]);
  }
  delete_tensor(ln_f_weight);
  delete_tensor(ln_f_bias);
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
