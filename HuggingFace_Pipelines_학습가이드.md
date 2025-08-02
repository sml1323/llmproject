# HuggingFace Pipelines 학습 가이드

## 📚 학습 목표
- HuggingFace Pipelines의 기본 개념 이해
- 다양한 AI 태스크에 대한 Pipeline 사용법 습득
- Training vs Inference의 차이점 이해
- 실무에서 활용할 수 있는 AI 모델 사용 경험

---

## 🎯 핵심 개념

### 1. Pipeline이란?
HuggingFace의 **고수준 API**로, 복잡한 AI 모델을 간단하게 사용할 수 있게 해주는 도구

```python
# 기본 사용 패턴
my_pipeline = pipeline("작업_유형")
result = my_pipeline(입력_데이터)
```

### 2. Training vs Inference 🔍

| 구분 | Training (훈련) | Inference (추론) |
|------|----------------|------------------|
| **목적** | 모델의 성능을 향상시키기 위해 데이터로 학습 | 이미 훈련된 모델로 새로운 결과 생성 |
| **데이터** | 학습용 데이터셋 (라벨 포함) | 새로운 입력 데이터 |
| **변화** | 모델의 가중치(parameters) 업데이트 | 모델 가중치 변경 없음 |
| **예시** | GPT 모델을 특정 도메인 데이터로 fine-tuning | ChatGPT에 질문하여 답변 받기 |

> 💡 **알아두기**: Pipeline은 **Inference 전용**입니다. 이미 훈련된 모델을 사용하여 결과를 생성하는 용도입니다.

---

## 🛠️ 환경 설정

### 필수 패키지 설치
```bash
# PyTorch (GPU 지원)
pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124

# HuggingFace 라이브러리
pip install transformers==4.48.3 datasets==3.2.0 diffusers
```

### HuggingFace 로그인 설정
1. [HuggingFace 회원가입](https://huggingface.co) (무료)
2. Settings → API Token 생성 (Write 권한 포함)
3. Colab 좌측 🔑 아이콘 → Secret 추가
   - Name: `HF_TOKEN`
   - Value: `hf_...` (생성한 토큰)

```python
from google.colab import userdata
from huggingface_hub import login

hf_token = userdata.get('HF_TOKEN')
login(hf_token, add_to_git_credential=True)
```

---

## 🚀 실습 예제들

### 1. 감정 분석 (Sentiment Analysis)
**용도**: 텍스트의 긍정/부정 감정 판별

```python
from transformers import pipeline

# Pipeline 생성
classifier = pipeline("sentiment-analysis", device="cuda")

# 사용 예제
result = classifier("I'm super excited to be on the way to LLM mastery!")
print(result)
# 출력: [{'label': 'POSITIVE', 'score': 0.9993460774421692}]
```

**학습 포인트**:
- `device="cuda"`: GPU 사용으로 속도 향상
- 기본 모델: `distilbert-base-uncased-finetuned-sst-2-english`
- 결과: 라벨(POSITIVE/NEGATIVE)과 신뢰도 점수

### 2. 개체명 인식 (Named Entity Recognition)
**용도**: 텍스트에서 인명, 지명, 조직명 등 추출

```python
ner = pipeline("ner", grouped_entities=True, device="cuda")
result = ner("Barack Obama was the 44th president of the United States.")
print(result)
```

**학습 포인트**:
- `grouped_entities=True`: 연속된 토큰을 하나로 그룹화
- 주요 엔티티 타입: PERSON, LOCATION, ORGANIZATION 등

### 3. 질의응답 (Question Answering)
**용도**: 주어진 문맥에서 질문에 대한 답변 추출

```python
question_answerer = pipeline("question-answering", device="cuda")
result = question_answerer(
    question="Who was the 44th president of the United States?", 
    context="Barack Obama was the 44th president of the United States."
)
print(result)
```

**학습 포인트**:
- 문맥(context) 내에서만 답변 추출
- 답변의 시작/끝 위치와 신뢰도 점수 제공

### 4. 텍스트 요약 (Text Summarization)
**용도**: 긴 텍스트를 짧게 요약

```python
summarizer = pipeline("summarization", device="cuda")
text = """긴 텍스트 내용..."""

summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
print(summary[0]['summary_text'])
```

**주요 파라미터**:
- `max_length`: 최대 요약 길이
- `min_length`: 최소 요약 길이  
- `do_sample=False`: 결정적 요약 (재현 가능)

### 5. 번역 (Translation)
**용도**: 언어 간 텍스트 번역

```python
# 영어 → 프랑스어
translator = pipeline("translation_en_to_fr", device="cuda")
result = translator("Hello, how are you?")

# 특정 모델 지정
translator = pipeline("translation_en_to_es", 
                     model="Helsinki-NLP/opus-mt-en-es", 
                     device="cuda")
```

**학습 포인트**:
- 다양한 언어쌍 지원
- 특정 모델 지정으로 성능 최적화 가능

### 6. 제로샷 분류 (Zero-shot Classification)
**용도**: 사전 훈련 없이 새로운 카테고리로 분류

```python
classifier = pipeline("zero-shot-classification", device="cuda")
result = classifier(
    "Hugging Face's Transformers library is amazing!", 
    candidate_labels=["technology", "sports", "politics"]
)
print(result)
```

**특징**:
- 미리 훈련되지 않은 카테고리도 분류 가능
- 각 라벨별 확률 점수 제공

### 7. 텍스트 생성 (Text Generation)
**용도**: 주어진 시작 텍스트를 바탕으로 문장 완성

```python
generator = pipeline("text-generation", device="cuda")
result = generator("If there's one thing I want you to remember about using HuggingFace pipelines, it's")
print(result[0]['generated_text'])
```

### 8. 이미지 생성 (Image Generation)
**용도**: 텍스트 설명으로부터 이미지 생성

```python
from diffusers import DiffusionPipeline
import torch

image_gen = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
).to("cuda")

text = "A class of Data Scientists learning about AI, in the surreal style of Salvador Dali"
image = image_gen(prompt=text).images[0]
image
```

**주요 파라미터**:
- `torch_dtype=torch.float16`: 메모리 효율성을 위한 반정밀도
- `use_safetensors=True`: 안전한 모델 로딩

### 9. 음성 합성 (Text-to-Speech)
**용도**: 텍스트를 음성으로 변환

```python
synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts", device='cuda')

# 화자 임베딩 로드
from datasets import load_dataset
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

# 음성 생성
speech = synthesiser(
    "Hi to an artificial intelligence engineer, on the way to mastery!", 
    forward_params={"speaker_embeddings": speaker_embedding}
)

# 파일 저장 및 재생
import soundfile as sf
from IPython.display import Audio

sf.write("speech.wav", speech["audio"], samplerate=speech["sampling_rate"])
Audio("speech.wav")
```

---

## 💡 실습 팁 & 문제해결

### Colab 사용 팁
1. **패키지 충돌 에러 무시**: pip 설치 시 dependency 경고는 무시해도 됨
2. **CUDA 에러 해결**: 
   - Kernel → Disconnect and delete runtime
   - 페이지 새로고침 → Edit → Clear All Outputs
   - 새 T4 GPU 연결 → View resources로 GPU 확인
   - 처음부터 다시 실행

### 성능 최적화
- `device="cuda"` 사용으로 GPU 가속
- 적절한 모델 선택 (크기 vs 성능)
- 배치 처리로 여러 입력 동시 처리

### 모델 선택 가이드
- **기본 모델**: 자동으로 선택되는 인기 모델
- **특정 모델 지정**: `model="모델명"` 파라미터 사용
- **모델 탐색**: [HuggingFace Models](https://huggingface.co/models)에서 검색

---

## 📖 추가 학습 자료

### 공식 문서
- [Transformers Pipelines](https://huggingface.co/docs/transformers/main_classes/pipelines)
- [Diffusers Pipelines](https://huggingface.co/docs/diffusers/en/api/pipelines/overview)

### 실습 확장 아이디어
1. **다국어 감정 분석**: 한국어 텍스트로 감정 분석 시도
2. **커스텀 질의응답**: 본인의 관심 분야 문서로 QA 시스템 구축
3. **창작 도우미**: 텍스트 생성으로 소설이나 시 작성 보조
4. **다중 언어 번역**: 여러 언어를 거쳐가는 번역 체인 구축

### 다음 단계
- **Week 6**: GPT 모델 Fine-tuning 실습
- **Week 7**: 나만의 모델 훈련하기
- **고급 API**: Transformers 라이브러리의 저수준 API 학습

---

## ✅ 학습 체크리스트

- [ ] Pipeline의 기본 개념과 사용법 이해
- [ ] Training과 Inference의 차이점 구분
- [ ] 9가지 Pipeline 유형별 실습 완료
- [ ] GPU 설정 및 최적화 방법 숙지
- [ ] HuggingFace Hub 로그인 및 모델 접근
- [ ] 에러 상황 대처 방법 학습
- [ ] 실무 적용 가능한 태스크 최소 3개 이상 확인

**🎉 축하합니다! HuggingFace Pipelines의 기초를 완료하셨습니다!**