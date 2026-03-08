#!/bin/sh
set -eu

MODEL_PATH="${MODEL_PATH:-}"

if [ -z "${MODEL_PATH}" ]; then
  echo "MODEL_PATH is not set."
  echo "Example:"
  echo "  MODEL_PATH=/path/to/model.safetensors make run"
  exit 1
fi

./main \
  -i ./data/sample_tokens_b1_t8.bin \
  -p "${MODEL_PATH}" \
  -o ./data/logits.bin \
  -v
