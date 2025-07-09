import os
import json
import time
import requests
from pathlib import Path
from tqdm import tqdm

# 配置DeepSeek API
DEEPSEEK_API_KEY = "sk-f5520e9c3cb64425a5ebe02fec10ed9a"  # 请替换为您的API密钥
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# 请求头
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
}

def generate_variants(text, variant_type):
    """
    使用DeepSeek API生成文本变体
    """
    prompts = {
        "formal": "请将以下文本改写为更正式的商务风格，保持原意不变但使用更专业的词汇和句式。只返回改写后的文本，不要添加任何解释或额外内容。",
        "casual": "请将以下文本改写为更口语化的表达，使其读起来像日常对话一样自然。只返回改写后的文本，不要添加任何解释或额外内容。",
        "detailed": "请扩展以下文本，添加更多细节和描述，使其长度增加约50%。只返回扩展后的文本，不要添加任何解释或额外内容。",
        "concise": "请将以下文本压缩为更简洁的版本，保留关键信息但减少约30%的篇幅。只返回压缩后的文本，不要添加任何解释或额外内容。",
        "synonym": "请将以下文本中的词汇替换为同义词，保持句子结构基本不变。只返回改写后的文本，不要添加任何解释或额外内容。",
        "passive": "请将以下文本改写为使用被动语态。只返回改写后的文本，不要添加任何解释或额外内容。"
    }
    
    prompt = f"{prompts[variant_type]}\n\n{text}"
    
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 2000
    }
    
    try:
        response = requests.post(
            DEEPSEEK_API_URL,
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"Error generating {variant_type} variant: {e}")
        return None

def process_file(file_path):
    """
    处理单个文件，生成6种变体并保存
    """
    print(f"\n{'='*50}\nProcessing file: {file_path}\n{'='*50}")
    
    # 读取原始文件内容
    with open(file_path, 'r', encoding='utf-8') as f:
        original_text = f.read()
    
    print(f"Original text length: {len(original_text)} characters")
    
    # 生成6种变体
    variants = {
        "formal": "正式风格：使用更专业的词汇和句式",
        "casual": "口语化表达：更自然的日常对话风格",
        "detailed": "详细扩展：增加更多细节和描述",
        "concise": "简洁版本：压缩内容保留关键信息",
        "synonym": "同义词替换：保持结构替换词汇",
        "passive": "被动语态：转换为被动语态表达"
    }
    
    # 为每种变体生成内容
    results = {}
    for variant_type, description in variants.items():
        print(f"\n{'*'*30}")
        print(f"Generating {variant_type} variant: {description}")
        result = generate_variants(original_text, variant_type)
        if result:
            results[variant_type] = result
            print(f"Generated {variant_type} variant: {len(result)} characters")
            # 打印前200个字符作为预览
            preview = result[:200].replace('\n', ' ').strip()
            print(f"Preview: {preview}...")
        else:
            print(f"Failed to generate {variant_type} variant")
        time.sleep(1)  # 避免API限流
    
    # 保存变体到文件
    base_path = Path(file_path).with_suffix('')
    for variant_type, content in results.items():
        output_path = f"{base_path}_{variant_type}.txt"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"\n✅ Saved {variant_type} variant to: {output_path}")
        
    print("\n" + "="*50)
    print(f"Finished processing: {file_path}")
    print("="*50 + "\n")

def should_skip_file(file_path):
    """检查是否应该跳过该文件（如果所有变体都已存在）"""
    base_path = str(file_path).replace('_combined_info.txt', '')
    required_variants = ['_formal', '_casual', '_detailed', '_concise', '_synonym', '_passive']
    
    # 检查是否所有变体文件都已存在
    all_exist = all(Path(f"{base_path}{variant}.txt").exists() for variant in required_variants)
    return all_exist

def main():
    # 设置数据集根目录
    dataset_root = Path("/Users/lizeyan/Desktop/TS/climb/Code/dataset")
    
    # 查找所有 _combined_info.txt 文件
    txt_files = list(dataset_root.glob("**/*_combined_info.txt"))
    
    print(f"Found {len(txt_files)} files in total")
    
    # 过滤出需要处理的文件（未完全生成变体的文件）
    files_to_process = []
    for file_path in txt_files:
        if not should_skip_file(file_path):
            files_to_process.append(file_path)
    
    print(f"Found {len(files_to_process)} files to process (excluding already processed)")
    
    if not files_to_process:
        print("All files have been processed. Exiting...")
        return
    
    # 处理每个文件
    for i, file_path in enumerate(files_to_process, 1):
        print(f"\n\n{'#'*80}")
        print(f"# Processing file {i}/{len(files_to_process)}: {file_path}")
        print(f"{'#'*80}")
        
        # 确保输出目录存在
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            process_file(str(file_path))
            # 处理完成后暂停2秒，避免API限流
            time.sleep(2)
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            # 发生错误时暂停5秒再继续
            time.sleep(5)

if __name__ == "__main__":
    main()
