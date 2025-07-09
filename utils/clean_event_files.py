import os
import re
import glob

def clean_event_file(file_path):
    """
    清理事件文件，删除没有实际事件内容的转折点
    
    Args:
        file_path: 事件文件路径
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 按照分隔符分割内容
    sections = content.split('---')
    
    # 过滤掉没有实际事件内容的部分
    filtered_sections = []
    for section in sections:
        # 检查是否包含"未找到相关政策或事件"
        if "未找到相关政策或事件" in section:
            continue
            
        # 检查是否为空或只包含空白字符
        if not section.strip():
            continue
            
        # 保留有实际内容的部分
        filtered_sections.append(section)
    
    # 重新组合内容
    new_content = '---'.join(filtered_sections)
    
    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"已清理文件: {file_path}")

def clean_all_event_files(root_dir):
    """
    清理指定目录下所有的combined_info.txt文件
    
    Args:
        root_dir: 根目录
    """
    # 查找所有combined_info.txt文件
    combined_files = glob.glob(f"{root_dir}/**/*_combined_info.txt", recursive=True)
    
    for file_path in combined_files:
        clean_event_file(file_path)
    
    print(f"共清理了 {len(combined_files)} 个文件")

if __name__ == "__main__":
    # 清理dataset目录下所有的combined_info.txt文件
    clean_all_event_files("./dataset")
