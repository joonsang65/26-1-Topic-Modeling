import os
import ast
import pandas as pd
from tqdm import tqdm

def parse_token_cell(cell) -> list:
    """
    다양한 형식의 토큰 컬럼을 리스트로 변환
    - '["word1", "word2"]' -> ['word1', 'word2']
    - 'word1 word2' -> ['word1', 'word2']
    """
    if pd.isna(cell) or cell == '':
        return []
    if isinstance(cell, list):
        return cell
    
    cell_str = str(cell).strip()
    if (cell_str.startswith('[') and cell_str.endswith(']')):
        try:
            return ast.literal_eval(cell_str)
        except:
            pass
    return cell_str.split()

def get_best_text_column(df: pd.DataFrame) -> str:
    """토큰 데이터가 포함된 가장 적절한 컬럼명 탐색"""
    candidates = ['final_text', 'tokenized_text', 'tokens', 'text', 'content']
    for c in candidates:
        if c in df.columns:
            return c
    return df.columns[0]

def load_documents(data_path: str) -> list:
    """
    CSV, Excel, TXT 파일에서 문서(토큰 리스트)를 로드
    """
    if not os.path.exists(data_path):
        print(f"[오류] 파일을 찾을 수 없습니다: {data_path}")
        return []

    print(f"Loading data from: {data_path}")
    ext = os.path.splitext(data_path)[1].lower()
    
    if ext == '.csv':
        df = pd.read_csv(data_path)
    elif ext in ['.xlsx', '.xls']:
        df = pd.read_excel(data_path)
    elif ext == '.txt':
        with open(data_path, 'r', encoding='utf-8') as f:
            return [line.strip().split() for line in f if line.strip()]
    else:
        print(f"[오류] 지원하지 않는 파일 형식: {ext}")
        return []

    text_col = get_best_text_column(df)
    print(f"  - detected text column: '{text_col}'")
    
    docs = []
    for cell in tqdm(df[text_col], desc="Parsing documents"):
        tokens = parse_token_cell(cell)
        if tokens:
            docs.append(tokens)
            
    return docs
