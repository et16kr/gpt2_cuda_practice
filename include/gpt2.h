#pragma once

#include <cstddef>

static constexpr size_t VOCAB_SIZE = 50257;
static constexpr size_t MAX_T = 1024;
static constexpr size_t N_LAYER = 12;
static constexpr size_t N_HEAD = 12;
static constexpr size_t C = 768;
static constexpr size_t HEAD_DIM = 64;
static constexpr size_t MLP_DIM = 3072;
static constexpr float LN_EPS = 1e-5f;
