# gpt2_practice

CUDA 연습용 GPT-2 inference 프로젝트입니다.

- 이 프로젝트는 서울대학교 천둥연구소 가속기 학교 과정에서 사용하는 CUDA 실습 프로젝트의 구성과 학습 방식을 참고해 Codex로 만들었습니다.
- 제공되는 CPU 기준 코드를 바탕으로 GPU(CUDA) 버전을 직접 구현해 보는 것이 목적입니다.
- GPT-2 base 모델의 추론 경로를 대상으로 하며, 토큰 입력을 받아 전체 forward를 수행하고 logits를 출력합니다.
- 실행 흐름은 `main.cpp`에서 입력 로딩, 모델 초기화, 추론, 저장, 검증을 담당합니다.
- `src/layer.cu`에는 CPU 기준 연산과 GPU TODO 함수가 함께 들어 있습니다.

## 참고

- 사용 모델: Hugging Face `openai-community/gpt2`
- 모델 링크: `https://huggingface.co/openai-community/gpt2`

## 준비 사항

- CUDA Toolkit과 `nvcc`
- C++17 지원 컴파일러
- GPT-2 가중치 파일 `model.safetensors`

이 저장소에는 모델 가중치가 포함되어 있지 않습니다. 사용자는 Hugging Face `openai-community/gpt2`에서 `model.safetensors`를 별도로 준비해야 합니다.

## 프로젝트 구성

- `src/main.cpp`: 입력 로딩, 모델 초기화, 추론 실행, 출력 저장, 검증
- `src/model.cu`: 모델 파라미터 로딩과 GPT-2 전체 forward
- `src/layer.cu`: CPU 기준 연산 구현과 GPU TODO 함수
- `src/tensor.cu`: `Tensor`, `TokenBatch` 메모리 관리
- `data/`: 샘플 토큰 입력 파일
- `run.sh`: 기본 실행 스크립트

## 입력 파일 형식

- `int32 B`
- `int32 T`
- `int32 token_ids[B*T]`

토큰은 모두 동일 길이 배치라고 가정합니다.

## 빌드

```bash
make
```

## 실행

모델 파일 경로는 사용자 환경마다 다르므로 `-p` 옵션 또는 `MODEL_PATH` 환경 변수로 실제 경로를 지정해야 합니다.

```bash
MODEL_PATH=/path/to/model.safetensors

./main -i ./data/sample_tokens_b1_t8.bin \
       -p ${MODEL_PATH} \
       -o ./data/logits.bin \
       -v
```

또는:

```bash
make run
```

`make run`은 내부적으로 `MODEL_PATH` 환경 변수를 사용합니다.

```bash
MODEL_PATH=/path/to/model.safetensors make run
```

## 현재 상태

- `gpt2_forward()`는 CPU 기준 경로를 사용합니다.
- `*_gpu()` 함수는 CPU 기준 결과를 먼저 내고, 학생이 CUDA kernel로 교체할 수 있도록 TODO를 남겨 두었습니다.
- 검증 모드는 CPU 기준 결과와 현재 forward 결과를 비교합니다.
- 샘플 입력 파일은 `data/sample_tokens_b1_t1.bin`, `data/sample_tokens_b1_t8.bin`, `data/sample_tokens_b2_t8.bin`에 포함되어 있습니다.
