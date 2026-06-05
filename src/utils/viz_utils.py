import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import platform

def set_korean_font():
    """OS에 따른 한글 폰트 설정"""
    system = platform.system()
    if system == 'Windows':
        font_name = 'Malgun Gothic'
    elif system == 'Darwin': # macOS
        font_name = 'AppleGothic'
    else: # Linux (Ubuntu/Debian)
        font_name = 'NanumGothic'
        
    plt.rcParams['font.family'] = font_name
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_style("whitegrid")
    return font_name

def save_fig(fig, path, dpi=300):
    """그래프 저장 공통 함수"""
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
