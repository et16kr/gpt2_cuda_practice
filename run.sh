#!/bin/sh
set -eu

./main \
  -i ./data/sample_tokens_b1_t8.bin \
  -p ../inference_practice/gpt2/model.safetensors \
  -o ./data/logits.bin \
  -v
