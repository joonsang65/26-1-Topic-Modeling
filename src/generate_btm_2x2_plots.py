import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ─────────────────────────────────────────────
# 0. 설정
# ─────────────────────────────────────────────
CSV_PATH = os.path.join("results", "modeling_results", "BTM", "step1_k_search", "BTM_K_metrics_summary.csv")
OUTPUT_DIR = os.path.join("results", "modeling_results", "BTM", "step1_k_search")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_2x2_plots():
    if not os.path.exists(CSV_PATH):
        print(f"[오류] 파일을 찾을 수 없습니다: {CSV_PATH}")
        return

    # 데이터 로드
    df = pd.read_csv(CSV_PATH)
    
    # 그래프 스타일 설정
    sns.set_style("whitegrid")
    plt.rcParams["font.family"] = "Malgun Gothic" # 한글 폰트 설정
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("BTM Topic Modeling: Multi-Metric K Analysis (2x2)", fontsize=16, y=0.95)

    # 1. C_umass (Decision)
    sns.lineplot(data=df, x='K', y='C_umass', marker='o', ax=axes[0, 0], color='steelblue')
    axes[0, 0].set_title("1. C_umass (Higher is better / Decision Basis)", fontsize=12)
    axes[0, 0].set_ylabel("Coherence Score")

    # 2. C_v (Reporting)
    sns.lineplot(data=df, x='K', y='C_v', marker='s', ax=axes[0, 1], color='forestgreen')
    axes[0, 1].set_title("2. C_v (Higher is better / Report Friendly)", fontsize=12)
    axes[0, 1].set_ylabel("Coherence Score")

    # 3. C_npmi (Evidence)
    sns.lineplot(data=df, x='K', y='C_npmi', marker='^', ax=axes[1, 0], color='darkorange')
    axes[1, 0].set_title("3. C_npmi (Higher is better / Strong Evidence)", fontsize=12)
    axes[1, 0].set_ylabel("Coherence Score")

    # 4. Perplexity (Complexity)
    sns.lineplot(data=df, x='K', y='Perplexity', marker='x', ax=axes[1, 1], color='crimson')
    axes[1, 1].set_title("4. Perplexity (Lower is better / Model Confusion)", fontsize=12)
    axes[1, 1].set_ylabel("Perplexity Score")

    # 공통 설정
    for ax in axes.flat:
        ax.set_xlabel("Number of Topics (K)")
        ax.set_xticks(df['K'].unique())

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = os.path.join(OUTPUT_DIR, "BTM_K_2x2_analysis.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    
    print(f"\n[시각화 완료]")
    print(f"  → 2x2 분석 그래프 저장: {save_path}")

if __name__ == "__main__":
    generate_2x2_plots()
