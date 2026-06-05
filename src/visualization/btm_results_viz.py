"""
BTM 결과 시각화 스크립트
========================
step2에서 생성된 K=3, 4, 5 결과물을 바탕으로
1. 토픽별 상위 단어 바 차트
2. 토픽별 워드클라우드
3. 토픽 비중(Topic Proportion) 파이 차트
를 자동으로 생성합니다.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from pathlib import Path

# 프로젝트 루트를 path에 추가하여 utils 임포트 가능하게 함
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from src.utils.viz_utils import set_korean_font, save_fig

# ─────────────────────────────────────────────
# 0. 설정 및 한글 폰트 설정
# ─────────────────────────────────────────────
K_LIST = [3, 4, 5]
BASE_DIR = os.path.join("results", "modeling_results", "BTM")

# 한글 폰트 설정
font_name = set_korean_font()
# wordcloud용 폰트 경로 (Windows 기준)
font_path = "C:/Windows/Fonts/malgun.ttf" if os.path.exists("C:/Windows/Fonts/malgun.ttf") else None

# ─────────────────────────────────────────────
# 1. 시각화 함수 정의
# ─────────────────────────────────────────────

def plot_top_words(df_words, k, output_dir):
    """토픽별 상위 단어 바 차트 생성"""
    n_topics = k
    cols = 3
    rows = (n_topics + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    axes = axes.flatten() if n_topics > 1 else [axes]
    
    for i in range(n_topics):
        topic_num = i + 1
        top_data = df_words[df_words['topic'] == topic_num].head(15)
        
        sns.barplot(x='prob', y='word', data=top_data, ax=axes[i], palette='viridis')
        axes[i].set_title(f"Topic {topic_num} Top Words", fontsize=13)
        axes[i].set_xlabel("Probability")
        axes[i].set_ylabel("")
        
    # 남은 빈 서브플롯 제거
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"BTM_K{k}_top_words_bar.png"), dpi=150)
    plt.close()

def plot_wordclouds(df_words, k, output_dir):
    """토픽별 워드클라우드 생성"""
    n_topics = k
    cols = 3
    rows = (n_topics + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    axes = axes.flatten() if n_topics > 1 else [axes]
    
    for i in range(n_topics):
        topic_num = i + 1
        # 단어와 확률을 딕셔너리로 변환하여 워드클라우드 생성
        top_data = df_words[df_words['topic'] == topic_num]
        word_freq = dict(zip(top_data['word'], top_data['prob']))
        
        wc = WordCloud(
            font_path=font_path if 'font_path' in locals() else None,
            background_color='white',
            width=800,
            height=600,
            colormap='Dark2'
        ).generate_from_frequencies(word_freq)
        
        axes[i].imshow(wc, interpolation='bilinear')
        axes[i].set_title(f"Topic {topic_num} WordCloud", fontsize=13)
        axes[i].axis('off')
        
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"BTM_K{k}_wordclouds.png"), dpi=150)
    plt.close()

def plot_topic_proportions(df_docs, k, output_dir):
    """전체 문서에서 각 토픽이 차지하는 비중 시각화"""
    # dominant_topic 컬럼의 빈도 계산
    prop = df_docs['dominant_topic'].value_counts().sort_index()
    labels = [f"Topic {i}" for i in prop.index]
    
    plt.figure(figsize=(8, 8))
    plt.pie(prop, labels=labels, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
    plt.title(f"BTM K={k} Topic Proportions", fontsize=15)
    plt.savefig(os.path.join(output_dir, f"BTM_K{k}_proportions.png"), dpi=150)
    plt.close()

# ─────────────────────────────────────────────
# 2. 메인 실행
# ─────────────────────────────────────────────

def main():
    print("[시각화 시작] BTM 결과 분석 중...")
    
    for k in K_LIST:
        target_dir = os.path.join(BASE_DIR, f"step2_K{k}")
        if not os.path.exists(target_dir):
            print(f"Skipping K={k}: 디렉토리를 찾을 수 없습니다.")
            continue
            
        print(f"\n[ K={k} 시각화 생성 중 ]")
        
        # 데이터 로드
        try:
            df_words = pd.read_csv(os.path.join(target_dir, f"BTM_topic_words_K{k}.csv"), encoding="utf-8-sig")
            df_docs = pd.read_csv(os.path.join(target_dir, f"BTM_document_topics_K{k}.csv"), encoding="utf-8-sig")
        except Exception as e:
            print(f"Error loading data for K={k}: {e}")
            continue
            
        # 시각화 실행
        plot_top_words(df_words, k, target_dir)
        plot_wordclouds(df_words, k, target_dir)
        plot_topic_proportions(df_docs, k, target_dir)
        
        print(f"K={k} 시각화 완료: {target_dir}")

    print("\n[ 모든 시각화 작업이 완료되었습니다. ]")

if __name__ == "__main__":
    main()
