# 토픽 모델링 파이프라인

LDA와 BTM 두 가지 토픽 모델링 기법을 활용해 전처리된 텍스트 데이터에서 잠재 토픽을 자동으로 추출하는 파이프라인입니다.  
**Optuna TPE 샘플러(베이즈 최적화)** 로 최적 하이퍼파라미터를 자동 탐색하고, 결과를 CSV·PNG·HTML 형태로 저장합니다.

---

## 파일 구성

```
텍스트 분석/
├── LDA_topic_modeling.py       # LDA 토픽 모델링 (gensim 기반)
├── BTM_topic_modeling.py       # BTM 토픽 모델링 (bitermplus 기반)
├── 최종정제_v2.csv             # 입력 데이터 (55,267개 댓글, final_text 컬럼)
├── README.md                   # 이 문서
├── LDA_결과/                   # LDA 실행 결과 저장 폴더 (자동 생성)
└── BTM_결과/                   # BTM 실행 결과 저장 폴더 (자동 생성)
```

---

## 요구 환경

- Python 3.8 이상
- Windows (한글 폰트: `C:/Windows/Fonts/malgun.ttf` 사용)

---

## 설치 패키지

### LDA

```bash
pip install gensim pyLDAvis matplotlib seaborn pandas openpyxl tqdm optuna
```

### BTM

```bash
pip install bitermplus matplotlib seaborn pandas openpyxl tqdm numpy scipy optuna
```

> **Windows에서 bitermplus 설치 오류 시**
> ```bash
> pip install bitermplus --no-build-isolation
> # 또는
> pip install bitermplus==0.12.3
> ```

---

## 입력 데이터

두 스크립트 모두 `최종정제_v2.csv` 단일 파일을 읽습니다.

| 항목 | 내용 |
|------|------|
| 파일 | `최종정제_v2.csv` |
| 텍스트 컬럼 | `final_text` (공백 구분 단어열) |
| 댓글 수 | 55,267개 |
| 고유 단어 수 | 4,096개 |
| 댓글당 평균 단어 수 | 12.51개 |

### 전처리 내역 요약

| 지표 | 전처리 전 | 최종 정제 | 변화율 |
|------|-----------|-----------|--------|
| 전체 댓글 수 | 74,356개 | 55,267개 | -25.7% |
| 전체 단어 수 | 약 1,120,000개 | 691,548개 | -38.3% |
| 고유 단어 수 | 약 28,000개 | 4,096개 | -85.4% |
| 댓글당 평균 단어 수 | 15.1개 | 12.51개 | -17.2% |
| 댓글 단어 수 중앙값 | 8개 | 6개 | -25.0% |

주요 정제 단계: ① 동사/형용사 어간에 `-다` 붙여 원형 복원 ② 유튜브 관련 메타 단어 제거(채널, 좋아요, 구독 등) ③ 등장 빈도 10회 이하 단어 제거 ④ 정제 후 남은 단어가 1개 이하인 댓글 삭제

---

## 설정값 변경

각 스크립트 상단의 **섹션 0. 설정값** 부분에서 수정합니다.

### 공통

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `DATA_FILE` | `최종정제_v2.csv` 경로 | 입력 CSV 파일 경로 |
| `TEXT_COL` | `"final_text"` | 텍스트가 담긴 컬럼명 |
| `OUTPUT_DIR` | `LDA_결과` / `BTM_결과` 폴더 경로 | 결과 저장 위치 |
| `K_START` | `2` | K 탐색 시작값 |
| `K_END` | `15` | K 탐색 끝값 |

### LDA 전용

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `N_TRIALS_LDA` | `50` | Optuna 탐색 시도 횟수 (많을수록 정밀, 오래 걸림) |
| `ALPHA_CANDIDATES` | `[0.01, 0.05, ..., "symmetric", "asymmetric"]` | alpha 탐색 후보 목록 |
| `ETA` | `"auto"` | 단어-토픽 분포 하이퍼파라미터 |
| `PASSES` | `20` | 전체 코퍼스 반복 횟수 |
| `ITERATIONS` | `400` | 내부 반복 횟수 |

### BTM 전용

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `N_TRIALS_BTM` | `40` | Optuna 탐색 시도 횟수 (많을수록 정밀, 오래 걸림) |
| `ALPHA_LOW` / `ALPHA_HIGH` | `0.01` / `5.0` | alpha 탐색 범위 (log scale) |
| `BETA_LOW` / `BETA_HIGH` | `0.001` / `0.5` | beta 탐색 범위 (log scale) |
| `N_ITER` | `200` | 깁스 샘플링 반복 수 |
| `WEIGHT_COH` | `0.5` | 복합 점수에서 Coherence 가중치 |
| `WEIGHT_PPL` | `0.5` | 복합 점수에서 Perplexity 가중치 |

> **시도 횟수 조정 팁**  
> `N_TRIALS_LDA` / `N_TRIALS_BTM` 을 줄이면 빠르게 실행되고, 늘리면 더 넓은 파라미터 공간을 탐색합니다.  
> Optuna TPE 샘플러는 초반 약 10~20회를 랜덤 탐색 후, 이후 시도부터 베이즈 최적화를 적용합니다.

---

## 실행 방법

```bash
# LDA 실행
python LDA_topic_modeling.py

# BTM 실행
python BTM_topic_modeling.py
```

---

## 파이프라인 흐름

### LDA (`LDA_topic_modeling.py`)

```
데이터 로드
    ↓
사전·코퍼스 구축 (gensim Dictionary + BoW)
    ↓
Optuna 탐색: K × alpha 동시 최적화
  - TPE 샘플러로 N_TRIALS_LDA 회 탐색
  - 평가 지표: Coherence c_v (최대화)
    ↓
최종 모델 학습 (최적 K + 최적 alpha)
    ↓
결과 저장 (CSV, PNG, HTML, 모델 파일)
```

### BTM (`BTM_topic_modeling.py`)

```
데이터 로드
    ↓
어휘 구축 (min_count=2, max_ratio=0.95 필터링)
    ↓
Optuna 탐색: K × alpha × beta 동시 최적화
  - TPE 샘플러로 N_TRIALS_BTM 회 탐색
  - 평가 지표: Coherence PMI (최대화, Optuna 기준)
  - best_k 선택: Coherence(↑) + Perplexity(↓) 복합 점수
      score = WEIGHT_COH × coh_norm + WEIGHT_PPL × (1 − ppl_norm)
    ↓
최종 모델 학습 (최적 K + 최적 alpha + 최적 beta)
    ↓
결과 저장 (CSV, PNG)
```

> BTM은 **단문 텍스트**(뉴스 헤드라인, SNS, 리뷰 등)에 특히 유리합니다.  
> 긴 문서에는 LDA가 일반적으로 더 적합합니다.

---

## Optuna 최적화 상세

### Grid Search vs Optuna 비교

| 항목 | 기존 Grid Search | Optuna TPE |
|------|-----------------|------------|
| 탐색 방식 | 파라미터 조합 전수 탐색 | 베이즈 최적화 (이전 결과 참고) |
| LDA 탐색 범위 | K(14) × alpha(8) = 최대 112회 | N_TRIALS_LDA 회 (기본 50회) |
| BTM 탐색 범위 | K(14)만 탐색, alpha·beta 고정 | K × alpha × beta 동시 탐색 |
| alpha·beta | 사전 정의 후보만 가능 | 연속 공간에서 세밀한 탐색 가능 |
| 효율 | 균등 탐색 (비효율 구간 낭비) | 유망 구간 집중 탐색 |

### BTM 복합 점수 (best_k 선택 기준)

기존에는 Coherence 최대값만으로 best_k를 결정했으나, Perplexity도 함께 반영하도록 개선했습니다.

1. Coherence와 Perplexity를 각각 Min-Max 정규화 → [0, 1] 범위로 통일
2. 복합 점수 계산:
   ```
   score = WEIGHT_COH × coh_norm + WEIGHT_PPL × (1 − ppl_norm)
   ```
   - Coherence가 높을수록, Perplexity가 낮을수록 높은 점수
3. 복합 점수가 가장 높은 K를 best_k로 선택

---

## 주요 함수 설명

### LDA

| 함수 | 역할 |
|------|------|
| `load_documents(data_file, text_col)` | `최종정제_v2.csv`의 `final_text` 컬럼을 읽어 토큰 리스트 반환 |
| `build_corpus(docs)` | gensim Dictionary와 BoW 코퍼스 생성 |
| `optuna_search_lda(...)` | Optuna TPE로 K·alpha 동시 탐색, 최적값 반환 |
| `train_final_model(...)` | 최적 K·alpha로 최종 LDA 모델 학습 |
| `save_topic_words(...)` | 토픽별 상위 단어를 CSV로 저장 |
| `save_document_topics(...)` | 문서별 토픽 분포를 CSV로 저장 |
| `plot_topic_word_heatmap(...)` | 토픽-단어 확률 히트맵 PNG 저장 |
| `save_pyldavis(...)` | pyLDAvis 인터랙티브 HTML 저장 |
| `plot_dominant_topic_dist(...)` | 토픽별 문서 수 분포 막대 그래프 저장 |

### BTM

| 함수 | 역할 |
|------|------|
| `load_documents(data_file, text_col)` | `최종정제_v2.csv`의 `final_text` 컬럼을 읽어 토큰 리스트 반환 |
| `build_vocab(docs, min_count, max_ratio)` | 빈도 기반 어휘 사전 구축 및 희귀·고빈도 단어 필터링 |
| `train_btm(...)` | bitermplus로 BTM 학습 |
| `_topic_coherence_btm(...)` | UCI Pointwise MI 근사로 BTM 코히런스 계산 |
| `optuna_search_btm(...)` | Optuna TPE로 K·alpha·beta 동시 탐색, 복합 점수로 best_k 선택 |
| `save_btm_topic_words(...)` | 토픽별 상위 단어를 CSV로 저장 |
| `save_btm_document_topics(...)` | 문서별 토픽 분포를 CSV로 저장 |
| `plot_btm_topic_word_heatmap(...)` | 히트맵 PNG 저장 |
| `plot_btm_topic_bar(...)` | 토픽별 상위 단어 막대 그래프 PNG 저장 |
| `plot_btm_dominant_topic_dist(...)` | 토픽별 문서 수 분포 PNG 저장 |

---

## 출력 결과물

### LDA 결과 (`LDA_결과/`)

| 파일 | 설명 |
|------|------|
| `LDA_optuna_search.png` | K별 Coherence 산점도 + Alpha별 평균 Coherence 막대 그래프 |
| `LDA_optuna_trials.csv` | Optuna 전체 시도 기록 (trial, K, alpha, coherence) |
| `LDA_optuna_summary.csv` | Optuna 탐색 요약 (최적 K·alpha 포함) |
| `LDA_topic_words.csv` | 토픽별 상위 단어 및 확률 |
| `LDA_document_topics.csv` | 문서별 토픽 분포 및 dominant topic |
| `LDA_topic_word_heatmap.png` | 토픽-단어 확률 히트맵 |
| `LDA_dominant_topic_distribution.png` | 토픽별 문서 수 분포 막대 그래프 |
| `LDA_pyLDAvis.html` | 인터랙티브 토픽 시각화 (브라우저에서 열기) |
| `lda_k{N}` | 최종 모델 파일 (gensim 형식) |

### BTM 결과 (`BTM_결과/`)

| 파일 | 설명 |
|------|------|
| `BTM_optuna_search.png` | K별 Perplexity / Coherence / 복합 점수 3개 패널 그래프 |
| `BTM_optuna_trials.csv` | Optuna 전체 시도 기록 (trial, K, alpha, beta, coherence, perplexity, combined_score) |
| `BTM_optuna_summary.csv` | Optuna 탐색 요약 (최적 K·alpha·beta 포함) |
| `BTM_topic_words.csv` | 토픽별 상위 단어 및 확률 |
| `BTM_document_topics.csv` | 문서별 토픽 분포 및 dominant topic |
| `BTM_topic_word_heatmap.png` | 토픽-단어 확률 히트맵 |
| `BTM_topic_top_words.png` | 토픽별 상위 단어 막대 그래프 |
| `BTM_dominant_topic_distribution.png` | 토픽별 문서 수 분포 막대 그래프 |

---

## LDA vs BTM 비교

| 항목 | LDA | BTM |
|------|-----|-----|
| 적합한 텍스트 | 긴 문서 | **단문** (트윗, 헤드라인, 리뷰) |
| 라이브러리 | gensim | bitermplus |
| 최적화 방식 | Optuna (K + alpha) | Optuna (K + alpha + beta) |
| best_k 선택 기준 | Coherence c_v 최대 | Coherence + Perplexity **복합 점수** |
| 인터랙티브 시각화 | pyLDAvis HTML | 없음 |

---

## 자주 발생하는 오류

**`bitermplus` 설치 실패 (Windows)**  
C 확장 컴파일러가 필요합니다. Visual Studio Build Tools를 설치하거나 아래 명령을 사용하세요.
```bash
pip install bitermplus --no-build-isolation
```

**`optuna` 설치 안 됨**
```bash
pip install optuna
```

**한글이 깨져서 그래프에 표시됨**  
`malgun.ttf` 폰트 경로를 확인하거나 스크립트 상단의 `FONT_PATH` 변수를 수정하세요.

**`컬럼 'final_text' 를 찾을 수 없습니다` 오류**  
CSV 파일의 컬럼명이 다를 경우 스크립트 상단의 `TEXT_COL` 변수를 실제 컬럼명으로 수정하세요.

**문서 수가 너무 적을 때**  
BTM의 경우 `build_vocab()`의 `min_count` 값을 `1`로 낮추어 어휘 크기를 늘려보세요.

**Optuna 탐색이 너무 오래 걸릴 때**  
`N_TRIALS_LDA` / `N_TRIALS_BTM` 값을 줄이거나, `PASSES` (LDA) / `N_ITER` (BTM) 값을 낮춰 개별 모델 학습 속도를 높이세요.
