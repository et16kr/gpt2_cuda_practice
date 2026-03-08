#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "gpt2.h"
#include "model.h"
#include "util.h"

static bool run_validation = false;
static bool run_warmup = false;

static char input_fname[256] = "./data/sample_tokens_b1_t8.bin";
static char param_fname[256] = "../inference_practice/gpt2/model.safetensors";
static char output_fname[256] = "./data/logits.bin";

void print_help(const char *prog) {
  printf("Usage: %s [-i input.bin] [-p model.safetensors] [-o logits.bin] "
         "[-v] [-w] [-h]\n",
         prog);
  printf("Options:\n");
  printf("  -i : token input file (default: ./data/sample_tokens_b1_t8.bin)\n");
  printf("  -p : GPT-2 safetensors path (default: "
         "../inference_practice/gpt2/model.safetensors)\n");
  printf("  -o : logits output file (default: ./data/logits.bin)\n");
  printf("  -v : validate current forward path against CPU reference\n");
  printf("  -w : warm-up once before timing\n");
  printf("  -h : print this help message\n");
}

void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "i:p:o:vwh")) != -1) {
    switch (c) {
      case 'i':
        strncpy(input_fname, optarg, sizeof(input_fname) - 1);
        input_fname[sizeof(input_fname) - 1] = '\0';
        break;
      case 'p':
        strncpy(param_fname, optarg, sizeof(param_fname) - 1);
        param_fname[sizeof(param_fname) - 1] = '\0';
        break;
      case 'o':
        strncpy(output_fname, optarg, sizeof(output_fname) - 1);
        output_fname[sizeof(output_fname) - 1] = '\0';
        break;
      case 'v': run_validation = true; break;
      case 'w': run_warmup = true; break;
      case 'h':
      default:
        print_help(argv[0]);
        exit(0);
    }
  }
}

int main(int argc, char **argv) {
  parse_args(argc, argv);

  printf("=============================================\n");
  printf(" GPT-2 Practice\n");
  printf("---------------------------------------------\n");
  printf(" Input path       : %s\n", input_fname);
  printf(" Model path       : %s\n", param_fname);
  printf(" Output path      : %s\n", output_fname);
  printf(" Validation       : %s\n", run_validation ? "ON" : "OFF");
  printf(" Warm-up          : %s\n", run_warmup ? "ON" : "OFF");
  printf("=============================================\n\n");

  printf("Loading token batch... ");
  fflush(stdout);
  TokenBatch tokens = load_tokens(input_fname);
  printf("Done! (B=%zu, T=%zu)\n", tokens.B, tokens.T);

  printf("Loading GPT-2 weights... ");
  fflush(stdout);
  initialize_model(param_fname);
  alloc_activations(tokens.B, tokens.T);
  printf("Done!\n");

  Tensor logits({tokens.B, tokens.T, VOCAB_SIZE});

  if (run_warmup) {
    printf("Warm-up... ");
    fflush(stdout);
    gpt2_forward(&tokens, &logits);
    printf("Done!\n");
  }

  printf("Running forward... ");
  fflush(stdout);
  double st = get_time();
  gpt2_forward(&tokens, &logits);
  double et = get_time();
  printf("Done!\n");

  double elapsed = et - st;
  printf("Elapsed time: %.6f sec\n", elapsed);
  printf("Throughput  : %.3f tokens/sec\n",
         (double)(tokens.B * tokens.T) / elapsed);

  print_last_token_topk(&logits, tokens.B, tokens.T, 5);

  printf("Writing logits to %s ... ", output_fname);
  fflush(stdout);
  write_binary(output_fname, logits.buf, logits.num_elem() * sizeof(float));
  printf("Done!\n");

  if (run_validation) {
    validate_against_cpu(&tokens, &logits);
  }

  free_activations();
  finalize_model();
  return 0;
}
