# 토픽 모델링 파이프라인

LDA와 BTM 두 가지 토픽 모델링 기법을 활용해 전처리된 텍스트 데이터에서 잠재 토픽을 자동으로 추출하는 파이프라인입니다.  
최적 K값(토픽 수)을 자동 탐색하고, 결과를 CSV·PNG·HTML 형태로 저장합니다.

---

## 파일 구성

```
텍스트 분석/
├── LDA_topic_modeling.py       # LDA 토픽 모델링 (gensim 기반)
├── BTM_topic_modeling.py       # BTM 토픽 모델링 (bitermplus 기반)
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
pip install gensim pyLDAvis matplotlib seaborn pandas openpyxl tqdm
```

### BTM

```bash
pip install bitermplus matplotlib seaborn pandas openpyxl tqdm numpy scipy
```

> **Windows에서 bitermplus 설치 오류 시**
> ```bash
> pip install bitermplus --no-build-isolation
> # 또는
> pip install bitermplus==0.12.3
> ```

---

## 입력 데이터

두 스크립트 모두 `전처리_정리` 폴더 안의 파일을 읽습니다.

| 형식 | 처리 방식 |
|------|-----------|
| `.txt` | 한 줄 = 한 문서, 공백으로 구분된 토큰 |
| `.csv` | 토큰 수가 가장 많은 컬럼 자동 감지 |
| `.xlsx` / `.xls` | CSV와 동일 |

토큰이 파이썬 리스트 형식(`["단어1", "단어2", ...]`)으로 저장된 셀도 자동으로 파싱합니다.

---

## 설정값 변경

각 스크립트 상단의 **섹션 0. 설정값** 부분에서 수정합니다.

### 공통

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `DATA_DIR` | `전처리_정리` 폴더 경로 | 입력 데이터 위치 |
| `OUTPUT_DIR` | `LDA_결과` / `BTM_결과` 폴더 경로 | 결과 저장 위치 |
| `K_START` | `2` | K 탐색 시작값 |
| `K_END` | `15` | K 탐색 끝값 |
| `K_STEP` | `1` | K 탐색 간격 |

### LDA 전용

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `ALPHA_CANDIDATES` | `[0.01, 0.05, 0.1, 0.3, 0.5, 1.0, "symmetric", "asymmetric"]` | alpha 탐색 후보 |
| `ETA` | `"auto"` | 단어-토픽 분포 하이퍼파라미터 |
| `PASSES` | `20` | 전체 코퍼스 반복 횟수 |
| `ITERATIONS` | `400` | 내부 반복 횟수 |

### BTM 전용

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `ALPHA` | `1.0` | 문서-토픽 디리클레 파라미터 |
| `BETA` | `0.01` | 토픽-단어 디리클레 파라미터 |
| `N_ITER` | `200` | 깁스 샘플링 반복 수 |

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
최적 K 탐색 (Coherence c_v 기준, K_START ~ K_END)
    ↓
최적 Alpha 탐색 (고정 K 기준, 후보 alpha 순회)
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
최적 K 탐색 (Perplexity + Coherence PMI 기준)
    ↓
최종 모델 학습 (최적 K)
    ↓
결과 저장 (CSV, PNG)
```

> BTM은 **단문 텍스트**(뉴스 헤드라인, SNS, 리뷰 등)에 특히 유리합니다.  
> 긴 문서에는 LDA가 일반적으로 더 적합합니다.

---

## 주요 함수 설명

### LDA

| 함수 | 역할 |
|------|------|
| `load_documents(data_dir)` | 폴더 내 CSV/TXT/XLSX를 읽어 토큰 리스트 반환 |
| `build_corpus(docs)` | gensim Dictionary와 BoW 코퍼스 생성 |
| `search_optimal_k(...)` | K 범위를 순회하며 Coherence(c_v) 측정, 최적 K 반환 |
| `search_optimal_alpha(...)` | 고정 K에서 alpha 후보별 Coherence 측정, 최적 alpha 반환 |
| `train_final_model(...)` | 최적 K·alpha로 최종 LDA 모델 학습 |
| `save_topic_words(...)` | 토픽별 상위 단어를 CSV로 저장 |
| `save_document_topics(...)` | 문서별 토픽 분포를 CSV로 저장 |
| `plot_topic_word_heatmap(...)` | 토픽-단어 확률 히트맵 PNG 저장 |
| `save_pyldavis(...)` | pyLDAvis 인터랙티브 HTML 저장 |
| `plot_dominant_topic_dist(...)` | 토픽별 문서 수 분포 막대 그래프 저장 |

### BTM

| 함수 | 역할 |
|------|------|
| `load_documents(data_dir)` | LDA와 동일한 데이터 로더 |
| `build_vocab(docs, min_count, max_ratio)` | 빈도 기반 어휘 사전 구축 및 희귀·고빈도 단어 필터링 |
| `train_btm(...)` | bitermplus로 BTM 학습 |
| `search_optimal_k_btm(...)` | K 범위를 순회하며 Perplexity·Coherence(PMI) 측정 |
| `_topic_coherence_btm(...)` | UCI Pointwise MI 근사로 BTM 코히런스 계산 |
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
| `LDA_coherence_by_K.png` | K별 Coherence 점수 그래프 |
| `LDA_coherence_by_alpha.png` | Alpha별 Coherence 점수 그래프 |
| `LDA_topic_words.csv` | 토픽별 상위 단어 및 확률 |
| `LDA_document_topics.csv` | 문서별 토픽 분포 및 dominant topic |
| `LDA_topic_word_heatmap.png` | 토픽-단어 확률 히트맵 |
| `LDA_dominant_topic_distribution.png` | 토픽별 문서 수 분포 막대 그래프 |
| `LDA_pyLDAvis.html` | 인터랙티브 토픽 시각화 (브라우저에서 열기) |
| `LDA_K_search_summary.csv` | K 탐색 요약 (K, Coherence) |
| `LDA_alpha_search_summary.csv` | Alpha 탐색 요약 |
| `lda_k{N}` | 최종 모델 파일 (gensim 형식) |

### BTM 결과 (`BTM_결과/`)

| 파일 | 설명 |
|------|------|
| `BTM_K_search.png` | K별 Perplexity·Coherence 그래프 |
| `BTM_K_search_summary.csv` | K 탐색 요약 (K, Perplexity, Coherence) |
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
| 최적화 지표 | Coherence c_v | Coherence PMI + Perplexity |
| 추가 탐색 | K + Alpha | K |
| 인터랙티브 시각화 | pyLDAvis HTML | 없음 |

---

## 자주 발생하는 오류

**`bitermplus` 설치 실패 (Windows)**  
C 확장 컴파일러가 필요합니다. Visual Studio Build Tools를 설치하거나 아래 명령을 사용하세요.
```bash
pip install bitermplus --no-build-isolation
```

**한글이 깨져서 그래프에 표시됨**  
`malgun.ttf` 폰트 경로를 확인하거나 스크립트 상단의 `FONT_PATH` 변수를 수정하세요.

**`토큰화된 텍스트 컬럼을 찾지 못했습니다` 오류**  
CSV/XLSX 파일에서 평균 토큰 수가 2 미만인 컬럼만 존재할 때 발생합니다. `_best_text_column()` 함수 대신 컬럼 이름을 직접 지정하세요.

**문서 수가 너무 적을 때**  
BTM의 경우 `build_vocab()`의 `min_count` 값을 `1`로 낮추어 어휘 크기를 늘려보세요.
