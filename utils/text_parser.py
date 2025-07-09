import os
import re
import glob
import pandas as pd
from datetime import datetime

def parse_turning_points_file(file_path):
    """
    解析转折点信息文件，提取日期和事件内容
    
    Args:
        file_path: 转折点信息文件路径
        
    Returns:
        events_dict: 以日期为键，事件内容为值的字典
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取所有转折点块
    turning_point_blocks = re.split(r'---', content)
    
    events_dict = {}
    
    for block in turning_point_blocks:
        # 跳过空块
        if not block.strip():
            continue
            
        # 跳过没有实际事件内容的块
        if "未找到相关政策或事件" in block:
            continue
        
        # 提取转折点行
        turning_point_match = re.search(r'## 转折点: (\d{4}-\d{2}-\d{2}) \(Event\) - 置信度: ([\d\.]+)', block)
        if not turning_point_match:
            continue
        
        date_str = turning_point_match.group(1)
        confidence = turning_point_match.group(2)
        
        # 尝试不同的模式提取事件内容
        event_content = ""
        
        # 模式1: 尝试提取事件名称和相关内容
        event_sections = re.findall(r'\d+\. \*\*(.+?)\*\*[\uff1a:]([\s\S]*?)(?=\d+\. \*\*|$)', block)
        if event_sections:
            event_content = "\n".join([f"{name.strip()}: {content.strip()}" for name, content in event_sections])
        
        # 模式2: 如果模式1失败，尝试提取事件名称
        if not event_content:
            event_names = re.findall(r'\*\*事件名称\*\*[\uff1a:](.+?)\n', block)
            if event_names:
                event_content = "\n".join([name.strip() for name in event_names])
        
        # 模式3: 如果前两种模式都失败，使用整个块作为事件内容
        if not event_content:
            # 去除转折点行
            content_lines = block.split('\n')
            filtered_lines = [line for line in content_lines if not line.startswith('## 转折点:')]
            event_content = '\n'.join(filtered_lines).strip()
        
        # 如果有内容且有日期，添加到字典
        if event_content and date_str:
            events_dict[date_str] = event_content
    
    return events_dict

def load_events_for_dataset(dataset_name, root_path, filename=None):
    """
    加载特定数据集的事件信息
    
    Args:
        dataset_name: 数据集名称
        root_path: 数据根目录
        filename: 可选的事件文件名，如果为None则使用默认命名规则
        
    Returns:
        events_dict: 以日期为键，核心内容为值的字典
    """
    if filename:
        # 如果指定了文件名，则直接使用
        info_file = os.path.join(root_path, filename)
    else:
        # 否则使用默认命名规则
        parts = root_path.split('/')
        folder_name = parts[-1]  # 例如 "Anhui_Industry"
        info_file = os.path.join(root_path, f"{folder_name}_combined_info.txt")
    
    try:
        events_dict = parse_turning_points_file(info_file)
        print(f"成功加载事件信息文件: {info_file}")
        print(f"共找到 {len(events_dict)} 个事件")
        return events_dict
    except Exception as e:
        print(f"加载事件信息文件失败: {e}")
        print(f"尝试查找其他可能的事件信息文件...")
        
        # 尝试查找其他可能的事件信息文件
        try:
            # 查找所有可能的事件信息文件
            possible_files = glob.glob(f"{root_path}/*_info.txt")
            
            if possible_files:
                # 选择combined_info.txt文件，如果存在的话
                combined_files = [f for f in possible_files if "combined_info" in f]
                if combined_files:
                    info_file = combined_files[0]
                else:
                    # 否则选择第一个找到的文件
                    info_file = possible_files[0]
                    
                events_dict = parse_turning_points_file(info_file)
                print(f"成功加载替代事件信息文件: {info_file}")
                print(f"共找到 {len(events_dict)} 个事件")
                return events_dict
            else:
                print(f"未找到任何事件信息文件")
                return {}
        except Exception as e2:
            print(f"尝试加载替代事件信息文件失败: {e2}")
            return {}
