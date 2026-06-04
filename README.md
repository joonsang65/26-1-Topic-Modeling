# 📽️ 유튜브 댓글 텍스트 분석 및 토픽 모델링 프로젝트

이 프로젝트는 유튜브 댓글 데이터를 활용하여 대중의 심리 지표(특히 '우울증' 관련)를 분석하기 위한 종합적인 텍스트 마이닝 파이프라인입니다. 데이터 수집부터 정교한 형태소 분석, 통계적 EDA, 그리고 최신 토픽 모델링 기법(LDA, BTM)을 아우르는 전 과정을 체계적으로 관리합니다.

---

## 🌟 주요 특징 (Key Features)

- **원천 데이터 수집**: 40개 이상의 관련 영상에서 수만 건의 댓글을 자동으로 크롤링.
- **적응적 전처리 (Adaptive Preprocessing)**: 
  - `Kiwi` 형태소 분석기를 활용한 정밀한 품사 추출 (명사, 동사, 형용사).
  - 구어체 정규화 및 서술어(동사/형용사)의 **원형 복원**('-다' 부착).
  - 저빈도 단어 및 플랫폼 특화 불용어(구독, 좋아요 등) 필터링을 통한 데이터 품질 최적화.
- **다각적 EDA**: 단어 빈도, N-gram 분석, 글자 수/토큰 수 분포 등 데이터의 통계적 특성 시각화.
- **이중 토픽 모델링**:
  - **LDA (Latent Dirichlet Allocation)**: 일반적인 문서 집합의 잠재 주제 추출.
  - **BTM (Biterm Topic Model)**: 유튜브 댓글과 같은 **단문(Short-text)** 분석에 최적화된 모델 적용.

---

## 📂 프로젝트 구조 (Project Structure)

```
사사_텍분/
├── data/
│   ├── raw/                  # [Input] 원본 크롤링 데이터 (.csv)
│   └── processed/            # [Output] 전처리 단계별 중간 및 최종 데이터셋
├── notebooks/                # [Step-by-Step] 데이터 처리 가이드
│   ├── Youtube_crawl.ipynb   # 1. 원천 데이터 수집
│   ├── preprocess.ipynb      # 2. 기초 텍스트 정제 (노이즈 제거)
│   ├── tokenize.ipynb        # 3. 기초 형태소 분석 및 토큰화
│   ├── adaptive_preprocess.ipynb # 4. 핵심 적응적 전처리 (최종 데이터 생성)
│   └── final_eda.ipynb       # 5. 최종 데이터 통계 분석
├── src/                      # [Scripts] 실행 가능한 파이썬 스크립트
│   ├── generate_eda_plots.py # EDA 결과물 자동 생성
│   ├── LDA_topic_modeling.py # LDA 모델링 및 최적 K/Alpha 탐색
│   └── BTM_topic_modeling.py # BTM 모델링 및 최적 K 탐색
├── results/                  # [Results] 시각화 및 모델링 결과
│   ├── figures/              # EDA 그래프 (.png)
│   └── modeling_results/     # LDA, BTM 상세 결과 (CSV, PNG, HTML)
├── docs/                     # [Documentation] 프로젝트 기술 문서
├── .gitignore                # Git 제외 설정 (데이터, 캐시 등)
├── requirements.txt          # 패키지 의존성 목록
└── README.md                 # 프로젝트 메인 설명서
```

---

## 🚀 빠른 시작 (Quick Start)

### 1. 환경 설정 (Setup)
Python 3.8 이상의 환경이 권장됩니다.
```bash
# 필수 라이브러리 설치
pip install -r requirements.txt
```

### 2. 분석 워크플로우 (Analysis Workflow)

1.  **데이터 수집**: `notebooks/Youtube_crawl.ipynb`를 실행하여 `data/raw/`에 원본 데이터를 확보합니다.
2.  **데이터 정제**: `preprocess.ipynb` → `tokenize.ipynb` → `adaptive_preprocess.ipynb` 순으로 실행하여 최종 데이터셋(`최종정제_v2.csv`)을 생성합니다.
3.  **시각화**: 아래 명령으로 EDA 그래프를 한꺼번에 생성할 수 있습니다.
    ```bash
    python src/generate_eda_plots.py
    ```
4.  **토픽 모델링**:
    ```bash
    # LDA 분석 실행
    python src/LDA_topic_modeling.py
    # BTM 분석 실행
    python src/BTM_topic_modeling.py
    ```

---

## 📊 데이터 요약 (Data Summary)

| 지표 | 수치 |
| :--- | :--- |
| **분석 대상 댓글 수** | 55,267개 |
| **추출된 고유 단어 수** | 4,096개 |
| **최상위 핵심 키워드** | 생각, 힘들다, 우울증, 살다, 죽다 |

---

## 🛠 기술 스택 (Tech Stack)

- **Language**: Python 3.11+
- **NLP**: `Kiwi` (kiwipiepy), `Gensim`, `bitermplus`
- **Analysis**: `Pandas`, `NumPy`, `Scikit-learn`
- **Visualization**: `Matplotlib`, `Seaborn`, `pyLDAvis`

---

## 📖 상세 문서
더 자세한 프로세스 설명은 [docs/process.md](./docs/process.md)를, 토픽 모델링 상세 가이드는 [docs/modeling_README.md](./docs/modeling_README.md)를 참고해 주세요.
