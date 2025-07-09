import os
import json
import re
import csv
from datetime import datetime, timedelta
import time
import sys
from openai import OpenAI
import pandas as pd
from tqdm import tqdm
import logging
import random
import configparser
from pathlib import Path
import argparse

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("policy_collector.log"),
        logging.StreamHandler()
    ]
)

def configure_logging():
    """配置日志设置"""
    # 设置日志级别
    logging.getLogger().setLevel(logging.INFO)
    
    # 确保log文件夹存在
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建一个专门的文件处理器，包含日期信息
    log_file = os.path.join(log_dir, f"policy_collector_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # 添加到根日志器
    root_logger = logging.getLogger()
    
    # 清除现有处理器，避免重复
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 添加新的处理器
    root_logger.addHandler(file_handler)
    
    # 添加控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    root_logger.addHandler(console_handler)
    
    logging.info("日志系统初始化完成")

# 设置路径
CHANGE_POINTS_DIR = "/Users/lizeyan/Desktop/climb/enhanced_change_points"
OUTPUT_DIR = "/Users/lizeyan/Desktop/climb/policy_information"

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_api_key():
    """从环境变量或配置文件中获取API密钥"""
    # 1. 首先检查环境变量
    api_key = os.environ.get("MOONSHOT_API_KEY")
    if api_key:
        return api_key
    
    # 2. 检查配置文件
    config_paths = [
        Path.home() / ".moonshot" / "config.ini",
        Path.home() / ".config" / "moonshot.ini",
        Path("config.ini"),
        Path("moonshot.ini")
    ]
    
    for config_path in config_paths:
        if config_path.exists():
            config = configparser.ConfigParser()
            config.read(config_path)
            if "API" in config and "key" in config["API"]:
                return config["API"]["key"]
    
    # 3. 交互式请求API密钥
    print("未找到API密钥。请提供Moonshot API密钥:")
    api_key = input().strip()
    if api_key:
        return api_key
    
    logging.error("未能获取API密钥，程序退出")
    sys.exit(1)

# 初始化OpenAI客户端
try:
    client = OpenAI(
        base_url="https://api.moonshot.cn/v1",
        api_key=get_api_key()
    )
    logging.info("成功初始化API客户端")
except Exception as e:
    logging.error(f"初始化API客户端失败: {str(e)}")
    sys.exit(1)

def parse_change_point_file(file_path):
    """解析变点文件，提取所有变点信息"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # 提取地区和部门信息
        file_match = re.search(r"File: (.*?)\.csv", content)
        if not file_match:
            logging.warning(f"无法从文件中提取名称: {file_path}")
            return None, []
        
        file_name = file_match.group(1)
        parts = file_name.split('_')
        if len(parts) < 2:
            logging.warning(f"文件名格式不正确: {file_name}")
            return None, []
        
        region = parts[0]
        sector = parts[1]
        
        # 提取变点信息 - 适应新的格式
        points = []
        point_blocks = re.findall(r"Point (\d+):\s+Date: ([\d-]+)\s+Type: (\w+)\s+Confidence: ([\d.]+)", content)
        
        for idx, date, point_type, confidence in point_blocks:
            points.append({
                'index': int(idx),
                'date': date.strip(),
                'type': point_type.strip(),
                'confidence': float(confidence.strip())
            })
        
        logging.info(f"从文件 {file_path} 中解析出 {len(points)} 个变点")
        return {'region': region, 'sector': sector}, points
    except Exception as e:
        logging.error(f"解析文件 {file_path} 出错: {str(e)}")
        return None, []

def generate_question(region, sector, date, point_type):
    """生成查询问题"""
    # 将日期格式化为更友好的形式
    try:
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        formatted_date = date_obj.strftime("%Y年%m月")
        # 获取前后3个月的时间范围
        three_months_before = (date_obj - timedelta(days=90)).strftime("%Y年%m月")
        three_months_after = (date_obj + timedelta(days=90)).strftime("%Y年%m月")
        date_range = f"{three_months_before}至{three_months_after}"
    except:
        try:
            # 尝试其他可能的日期格式
            if '/' in date:
                parts = date.split('/')
                if len(parts) == 3:
                    formatted_date = f"{parts[2]}年{parts[1]}月"
                    # 简单估计前后3个月
                    month = int(parts[1])
                    year = int(parts[2])
                    
                    # 计算前3个月
                    before_month = ((month - 3 - 1) % 12) + 1
                    before_year = year - 1 if month <= 3 else year
                    
                    # 计算后3个月
                    after_month = ((month + 3 - 1) % 12) + 1
                    after_year = year + 1 if month >= 10 else year
                    
                    date_range = f"{before_year}年{before_month}月至{after_year}年{after_month}月"
                else:
                    formatted_date = date
                    date_range = "前后6个月内"
            else:
                formatted_date = date
                date_range = "前后6个月内"
        except:
            formatted_date = date
            date_range = "前后6个月内"
    
    # 根据不同的部门和类型生成不同的问题
    if point_type.lower() == "policy":
        if sector.lower() == "aviation":
            question = f"搜索{region}省/市在{formatted_date}（{date_range}）航空业碳排放相关的具体政策。请查找：1）民航局发布的与碳排放、节能减排相关的政策；2）{region}省/市针对航空业的碳排放管控政策；3）机场、航空公司在{region}的减排措施。必须提供政策名称、发布机构、发布日期和政策文号，每条信息必须注明官方信息来源网址。"
        elif sector.lower() == "power":
            question = f"搜索{region}省/市在{formatted_date}（{date_range}）电力部门碳排放相关的具体政策。请查找：1）国家发改委、能源局针对电力行业的碳排放政策；2）{region}省/市电力部门的节能减排政策；3）影响电力结构的可再生能源政策；4）火电厂减排或关停政策。必须提供政策名称、发布机构、发布日期和政策文号，每条信息必须注明官方信息来源网址。"
        elif sector.lower() == "industry":
            question = f"搜索{region}省/市在{formatted_date}（{date_range}）工业部门碳排放相关的具体政策。请查找：1）工信部发布的工业减排政策；2）{region}省/市工业部门的节能减排政策；3）针对高能耗工业企业的限产停产政策；4）工业能效提升相关政策。必须提供政策名称、发布机构、发布日期和政策文号，每条信息必须注明官方信息来源网址。"
        elif sector.lower() == "residential":
            question = f"搜索{region}省/市在{formatted_date}（{date_range}）居民生活领域碳排放相关的具体政策。请查找：1）住建部关于建筑节能的政策；2）{region}省/市民用建筑节能标准或政策；3）居民用电、用气、供暖的价格或补贴政策变化；4）绿色生活相关政策。必须提供政策名称、发布机构、发布日期和政策文号，每条信息必须注明官方信息来源网址。"
        elif sector.lower() == "ground_transport":
            question = f"搜索{region}省/市在{formatted_date}（{date_range}）地面交通碳排放相关的具体政策。请查找：1）交通部或发改委关于交通减排的政策；2）{region}省/市针对汽车限行、限购的政策；3）公共交通发展政策；4）新能源汽车推广政策。必须提供政策名称、发布机构、发布日期和政策文号，每条信息必须注明官方信息来源网址。"
        else:
            question = f"搜索{region}省/市在{formatted_date}（{date_range}）碳排放{sector}部门出现变化的具体政策。请查找：1）国家层面与{sector}部门碳排放相关的政策法规；2）{region}省/市地方层面的{sector}部门碳排放相关政策。必须提供具体政策名称、发布机构、发布日期、文号和政策内容要点。请列出所有查找到的政策，每项需注明具体信息来源网址。"
    elif point_type.lower() == "event":
        if sector.lower() == "aviation":
            question = f"搜索{region}省/市在{formatted_date}（{date_range}）影响航空业碳排放的具体重大事件。请查找：1）{region}机场的建设、扩建或关闭事件；2）影响航空运输的重大自然灾害或突发事件；3）航空公司在{region}的航线调整或机队变化；4）机场能源系统重大变化。必须提供事件名称、发生时间、地点、规模和影响，每条信息必须注明信息来源网址。"
        elif sector.lower() == "power":
            question = f"搜索{region}省/市在{formatted_date}（{date_range}）影响电力部门碳排放的具体重大事件。请查找：1）大型电厂投产或关停事件；2）可再生能源项目并网事件；3）电力系统重大事故；4）跨区域输电线路建设。必须提供事件名称、发生时间、地点、规模和影响，每条信息必须注明信息来源网址。"
        elif sector.lower() == "industry":
            question = f"搜索{region}省/市在{formatted_date}（{date_range}）影响工业部门碳排放的具体重大事件。请查找：1）高耗能企业投产或关停事件；2）工业园区建设或调整；3）重大技术改造项目；4）重大工业项目节能降碳措施实施。必须提供事件名称、发生时间、地点、规模和影响，每条信息必须注明信息来源网址。"
        elif sector.lower() == "residential":
            question = f"搜索{region}省/市在{formatted_date}（{date_range}）影响居民部门碳排放的具体重大事件。请查找：1）大型住宅小区建成或改造事件；2）供暖系统改造或清洁能源替代项目；3）居民用能模式变化的重大事件；4）绿色建筑示范项目实施。必须提供事件名称、发生时间、地点、规模和影响，每条信息必须注明信息来源网址。"
        elif sector.lower() == "ground_transport":
            question = f"搜索{region}省/市在{formatted_date}（{date_range}）影响地面交通碳排放的具体重大事件。请查找：1）地铁或轻轨线路开通；2）公交系统重大调整；3）新能源汽车推广重大活动；4）高速公路或城市道路建设。必须提供事件名称、发生时间、地点、规模和影响，每条信息必须注明信息来源网址。"
        else:
            question = f"搜索{region}省/市在{formatted_date}（{date_range}）碳排放{sector}部门出现变化的具体事件。请查找：1）可能影响{sector}部门碳排放的重大事件，如自然灾害、经济事件、重大基础设施建设等；2）{region}省/市{sector}部门的重大项目或设施变动情况。必须提供具体事件名称、发生时间、地点、规模和影响。请列出所有查找到的事件，每项需注明具体信息来源网址。"
    elif point_type.lower() == "seasonal":
        if sector.lower() == "aviation":
            question = f"搜索{region}省/市在{formatted_date}（{date_range}）航空运输领域的具体季节性政策和事件。请查找：1）民航局或{region}省交通厅发布的季节性航班计划、航线调整政策；2）{formatted_date}季节性客流量变化的官方数据；3）该时期{region}省内机场的航班量、客运量变化的具体报道；4）航空燃油使用、机场能源消耗的季节性变化数据；5）如果在2020年期间，请特别关注疫情对航空业恢复的相关政策和报道。必须提供准确的信息来源，包括政府网站、民航局官网、航空公司公告或权威媒体报道的具体网址。每条信息必须包含具体日期、发布机构和主要内容。"
        elif sector.lower() == "power":
            question = f"搜索{region}省/市在{formatted_date}（{date_range}）电力部门的具体季节性政策和事件。请查找：1）电网公司或能源局发布的季节性电力调度政策；2）季节性电价调整政策；3）针对季节性用电高峰的应对措施；4）电力供需季节性波动的官方数据。必须提供具体政策名称、发布时间、发布机构和主要内容，每条信息必须注明具体信息来源网址。"
        elif sector.lower() == "residential":
            question = f"搜索{region}省/市在{formatted_date}（{date_range}）居民用能的具体季节性政策和事件。请查找：1）与季节相关的供暖或制冷政策；2）季节性电价、气价调整政策；3）针对季节性用能高峰的节能措施；4）居民季节性用能模式变化的具体报道。必须提供具体政策名称、发布时间、发布机构和主要内容，每条信息必须注明具体信息来源网址。"
        else:
            question = f"搜索{region}省/市在{formatted_date}（{date_range}）碳排放{sector}部门出现的具体季节性政策和事件。请查找：1）{region}省发改委、能源局等部门发布的针对{sector}部门季节性调整的具体政策文件；2）{region}省{sector}部门季节性需求变化的官方统计数据；3）与{sector}部门季节性碳排放相关的地方政府措施；4）影响{sector}部门季节性能源使用的具体项目或举措。每条信息必须包含具体名称、发布/实施日期、发布机构和主要内容，并注明准确信息来源网址。如搜索不到具体政策或事件，请明确说明。"
    elif point_type.lower() == "trend":
        if sector.lower() == "aviation":
            question = f"搜索{region}省/市在{formatted_date}（{date_range}）航空业碳排放长期趋势变化的具体政策和事件。请查找：1）影响航空业长期发展的规划政策；2）航空公司机队更新或技术改进计划；3）{region}机场长期扩建或改造计划；4）航空燃油效率提升或替代燃料的技术应用。必须提供具体政策名称、事件、计划启动时间和主要内容，每条信息必须注明信息来源网址。"
        elif sector.lower() == "power":
            question = f"搜索{region}省/市在{formatted_date}（{date_range}）电力部门碳排放长期趋势变化的具体政策和事件。请查找：1）电力结构调整的长期规划；2）可再生能源开发规划；3）火电改造或清洁能源替代项目；4）电网技术升级计划。必须提供具体政策名称、事件、计划启动时间和主要内容，每条信息必须注明信息来源网址。"
        else:
            question = f"搜索{region}省/市在{formatted_date}（{date_range}）碳排放{sector}部门出现趋势变化的具体政策和事件。请查找：1）可能导致{sector}部门碳排放趋势变化的经济政策或产业政策；2）{region}省/市{sector}部门的技术革新、能源结构调整等重大动态；3）影响长期碳排放趋势的重大基础设施项目或产业升级计划。必须提供具体政策名称、事件、项目启动时间和主要内容。请列出所有查找到的信息，每项需注明具体信息来源网址。"
    else:
        question = f"搜索{region}省/市在{formatted_date}（{date_range}）碳排放{sector}部门出现变化的具体政策和事件。请查找：1）与{sector}部门碳排放相关的国家和地方政策法规；2）影响{region}省/市{sector}部门碳排放的重大事件、项目或举措。必须提供具体政策名称、事件名称、发生时间、主要内容和影响。请列出所有查找到的信息，每项需注明具体信息来源网址。"
    
    return question

def chat_with_kimi(user_question, max_retries=3):
    """使用Kimi API查询信息"""
    # 设置初始消息，包含鼓励网络搜索的系统提示
    messages = [
        {
            "role": "system", 
            "content": "你是 Kimi，由 Moonshot AI 提供的人工智能助手。你的任务是搜索能够直接影响碳排放的具体政策和事件，而非学术研究或分析性报道。\n\n请务必遵循以下规则：\n1) 只提供可能直接改变实际碳排放量的信息，如政府限排政策、能源结构调整措施、重大基础设施变化、大型工业项目启停\n2) 必须排除纯学术研究、理论分析报告、排放统计等不会实际影响排放的内容\n3) 每项内容必须包含：具体政策名称/事件、发布/发生时间、责任机构、核心内容、预期影响\n4) 如果找不到确实能影响排放的政策或事件，请简单地写\"未找到相关政策或事件\"，不要列出每个类别都未找到的详细说明\n5) 不要提供未能确认其实际影响力的信息\n6) 所有内容必须有可验证的官方来源链接\n7) 只列出成功找到的信息，完全跳过未找到信息的类别，不要写\"未找到\"之类的说明\n\n重要：绝对不要提供学术文章、研究报告或排放分析作为答案，除非它们直接导致了政策或实践的变化。请按时间顺序或重要性排序列出所有查找到的内容，并使用编号形式呈现。如果完全没有找到任何相关信息，只需简单写\"未找到任何相关政策或事件\"。"
        },
        {"role": "user", "content": user_question}
    ]
    
    retry_count = 0
    while retry_count < max_retries:
        try:
            # 处理对话，可能包含工具调用
            finish_reason = None
            while finish_reason is None or finish_reason == "tool_calls":
                response = client.chat.completions.create(
                    model="moonshot-v1-128k",
                    messages=messages,
                    temperature=0.3,
                    tools=[
                        {
                            "type": "builtin_function",
                            "function": {
                                "name": "$web_search",
                            },
                        }
                    ]
                )
                
                choice = response.choices[0]
                finish_reason = choice.finish_reason
                
                if finish_reason == "tool_calls":
                    # 将助手的消息添加到对话中
                    messages.append(choice.message)
                    
                    # 处理每个工具调用
                    for tool_call in choice.message.tool_calls:
                        tool_call_name = tool_call.function.name
                        tool_call_arguments = json.loads(tool_call.function.arguments)
                        
                        if tool_call_name == "$web_search":
                            # 对于Kimi的内置网络搜索，只需返回参数
                            tool_result = tool_call_arguments
                        else:
                            tool_result = f"Error: unable to find tool by name '{tool_call_name}'"
                        
                        # 将工具响应添加到对话中
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call_name,
                            "content": json.dumps(tool_result)
                        })
            
            # 返回最终响应
            return choice.message.content
            
        except Exception as e:
            retry_count += 1
            wait_time = random.uniform(2, 5) * retry_count  # 指数退避
            logging.warning(f"API调用错误 (尝试 {retry_count}/{max_retries}): {str(e)}")
            logging.info(f"等待 {wait_time:.1f} 秒后重试...")
            time.sleep(wait_time)
    
    # 所有重试都失败
    logging.error(f"API调用在 {max_retries} 次尝试后仍然失败")
    return f"抱歉，无法获取相关信息。API调用失败: 尝试了 {max_retries} 次。"

def process_all_change_points(start_from=None, max_files=None, append_mode=False):
    """处理所有变点文件并收集相关信息"""
    # 创建结果CSV文件
    result_csv = os.path.join(OUTPUT_DIR, "all_change_points_info.csv")
    
    # 如果从中断处继续或者追加模式，则附加到现有文件
    file_mode = 'a' if start_from or append_mode else 'w'
    
    with open(result_csv, file_mode, newline='', encoding='utf-8') as csvfile:
        fieldnames = ['region', 'sector', 'date', 'type', 'confidence', 'info_summary']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # 只有在新文件中才写入标题
        if file_mode == 'w':
            writer.writeheader()
    
    # 获取所有变点文件
    change_point_files = [f for f in os.listdir(CHANGE_POINTS_DIR) if f.endswith('_change_points.txt')]
    
    # 按字母顺序排序文件以便于管理
    change_point_files.sort()
    
    # 如果指定了起始文件，则从该文件开始处理
    if start_from:
        if start_from in change_point_files:
            # 如果指定的是精确文件名，只处理这一个文件
            change_point_files = [start_from]
            logging.info(f"只处理指定文件: {start_from}")
        else:
            try:
                start_index = change_point_files.index(start_from)
                change_point_files = change_point_files[start_index:]
                logging.info(f"从文件 {start_from} 开始继续处理")
            except ValueError:
                logging.warning(f"未找到指定的起始文件 {start_from}，将从头开始处理")
    
    # 如果指定了最大文件数，则限制处理文件数量
    if max_files:
        change_point_files = change_point_files[:max_files]
        logging.info(f"将处理 {len(change_point_files)} 个文件")
    
    # 创建结果汇总
    all_results = []
    
    # 创建或追加处理进度文件
    progress_file = os.path.join(OUTPUT_DIR, "processing_progress.txt")
    progress_mode = 'a' if append_mode else 'w'
    with open(progress_file, progress_mode, encoding='utf-8') as f:
        f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"文件数: {len(change_point_files)}\n\n")
    
    # 记录当前处理的CSV文件及其所有变点信息
    current_csv = None
    combined_points_info = {}
    
    # 如果是追加模式，加载现有的合并信息
    if append_mode:
        # 遍历目录中的所有combined_info文件，加载它们
        for filename in os.listdir(OUTPUT_DIR):
            if filename.endswith('_combined_info.txt'):
                csv_name = filename.replace('_combined_info.txt', '')
                # 从合并文件中提取区域和部门
                parts = csv_name.split('_')
                if len(parts) >= 2:
                    combined_points_info[csv_name] = []
    
    for file_idx, file in enumerate(tqdm(change_point_files, desc="处理文件")):
        file_path = os.path.join(CHANGE_POINTS_DIR, file)
        metadata, points = parse_change_point_file(file_path)
        
        # 更新进度文件
        with open(progress_file, 'a', encoding='utf-8') as f:
            f.write(f"[{file_idx+1}/{len(change_point_files)}] 处理文件: {file}\n")
        
        if not metadata:
            logging.warning(f"无法解析文件: {file}")
            continue
        
        # 提取当前CSV文件名
        csv_name = f"{metadata['region']}_{metadata['sector']}"
        
        # 如果是新的CSV文件，保存前一个文件的汇总信息
        if current_csv is not None and current_csv != csv_name and current_csv in combined_points_info:
            save_combined_info(current_csv, combined_points_info[current_csv])
        
        # 设置当前CSV文件
        current_csv = csv_name
        
        # 确保该CSV文件在字典中有一个条目
        if current_csv not in combined_points_info:
            combined_points_info[current_csv] = []
            
        # 加载现有的合并文件内容（如果存在）
        combined_file = os.path.join(OUTPUT_DIR, f"{csv_name}_combined_info.txt")
        existing_points = []
        if os.path.exists(combined_file) and append_mode:
            # 从现有文件中提取已处理的点
            with open(combined_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # 使用正则表达式匹配转折点信息
                point_blocks = re.findall(r"## 转折点: ([\d-]+) \((\w+)\) - 置信度: ([\d.]+)", content)
                for date, point_type, confidence in point_blocks:
                    existing_points.append((date, point_type))
        
        # 筛选出置信度高的变点，降低阈值以获取更多点
        significant_points = [p for p in points if p['confidence'] > 0.55]
        
        # 按置信度排序，取前10个最重要的点
        significant_points.sort(key=lambda x: x['confidence'], reverse=True)
        top_points = significant_points[:10]
        
        # 记录处理的变点数
        with open(progress_file, 'a', encoding='utf-8') as f:
            f.write(f"  找到 {len(points)} 个变点, {len(significant_points)} 个重要变点, 选取前 {len(top_points)} 个处理\n")
        
        file_results = []
        
        for point in top_points:
            # 检查是否已经处理过这个点
            point_key = (point['date'], point['type'])
            if point_key in existing_points:
                logging.info(f"跳过已处理的转折点: {point['date']} ({point['type']})")
                continue
                
            # 生成问题
            question = generate_question(
                metadata['region'], 
                metadata['sector'], 
                point['date'], 
                point['type']
            )
            
            # 查询Kimi API
            logging.info(f"\n正在查询: {metadata['region']} {metadata['sector']} {point['date']} ({point['type']})")
            logging.info(f"问题: {question}")
            
            # 构建点信息标识符
            point_identifier = f"{point['date']}_{point['type']}"
            
            # 检查是否已存在相同的查询结果
            detail_file = os.path.join(
                OUTPUT_DIR, 
                f"{metadata['region']}_{metadata['sector']}_{point['date'].replace('/', '-')}_{point['type']}_info.txt"
            )
            
            if os.path.exists(detail_file):
                logging.info(f"已存在查询结果: {detail_file}，读取已有内容")
                with open(detail_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # 提取回答部分
                    answer_match = re.search(r"回答:\n(.*)", content, re.DOTALL)
                    if answer_match:
                        info = answer_match.group(1)
                    else:
                        info = "无法从现有文件中提取回答"
            else:
                try:
                    # 执行查询
                    info = chat_with_kimi(question)
                    logging.info(f"获取到信息，长度: {len(info)}")
                    
                    # 保存单独的详细信息文件（保留这一步以便调试）
                    with open(detail_file, 'w', encoding='utf-8') as f:
                        f.write(f"问题: {question}\n\n")
                        f.write(f"回答:\n{info}")
                    
                    # 更新进度文件
                    with open(progress_file, 'a', encoding='utf-8') as f:
                        f.write(f"  - 查询 {point['date']} ({point['type']}): 成功, 长度 {len(info)}\n")
                    
                except Exception as e:
                    error_msg = f"处理出错: {str(e)}"
                    logging.error(error_msg)
                    info = f"查询失败: {error_msg}"
                    
                    # 更新进度文件
                    with open(progress_file, 'a', encoding='utf-8') as f:
                        f.write(f"  - 查询 {point['date']} ({point['type']}): 失败, {error_msg}\n")
                
                # 在API调用之间添加延迟，避免限速
                time.sleep(2)
            
            # 添加到结果
            result = {
                'region': metadata['region'],
                'sector': metadata['sector'],
                'date': point['date'],
                'type': point['type'],
                'confidence': point['confidence'],
                'info_summary': info[:200] + "..." if len(info) > 200 else info,
                'full_info': info,
                'point_identifier': point_identifier
            }
            file_results.append(result)
            all_results.append(result)
            
            # 将信息添加到合并的信息中
            combined_points_info[current_csv].append(result)
            
            # 将结果追加到CSV
            with open(result_csv, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({k: result[k] for k in fieldnames})
        
        # 每个文件处理完后，更新汇总报告
        with open(progress_file, 'a', encoding='utf-8') as f:
            f.write(f"  处理了 {len(file_results)} 个变点\n")
            f.write(f"  累计处理: {len(all_results)} 个变点\n\n")
        
        # 定期保存中间结果，避免长时间运行导致数据丢失
        if file_idx % 10 == 0 and file_idx > 0:
            logging.info(f"已处理 {file_idx} 个文件，保存中间结果...")
            create_summary_report(all_results, suffix=f"_interim_{file_idx}")
            
            # 保存当前处理的所有CSV文件的汇总信息
            for csv_file, points_info in combined_points_info.items():
                save_combined_info(csv_file, points_info)
    
    # 保存最后一个CSV文件的汇总信息
    if current_csv and current_csv in combined_points_info:
        save_combined_info(current_csv, combined_points_info[current_csv])
    
    # 创建汇总报告
    create_summary_report(all_results)
    
    # 完成进度记录
    with open(progress_file, 'a', encoding='utf-8') as f:
        f.write(f"\n处理完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总共处理了 {len(all_results)} 个变点\n")
    
    return combined_points_info

def save_combined_info(csv_name, points_info):
    """将同一个CSV文件的所有变点信息保存到一个文件中"""
    if not points_info:
        return
    
    # 确保按日期排序
    points_info.sort(key=lambda x: x['date'])
    
    # 生成合并文件的路径
    combined_file = os.path.join(OUTPUT_DIR, f"{csv_name}_combined_info.txt")
    
    # 提取第一个条目的区域和部门信息
    region = points_info[0]['region']
    sector = points_info[0]['sector']
    
    with open(combined_file, 'w', encoding='utf-8') as f:
        f.write(f"# {region}省/市 {sector}部门 碳排放变点相关政策与事件汇总\n\n")
        f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"共包含 {len(points_info)} 个重要转折点的相关信息\n\n")
        f.write("---\n\n")
        
        # 按日期分组写入信息
        for point_info in points_info:
            f.write(f"## 转折点: {point_info['date']} ({point_info['type']}) - 置信度: {point_info['confidence']:.2f}\n\n")
            f.write(f"{point_info['full_info']}\n\n")
            f.write("---\n\n")
    
    logging.info(f"已保存合并信息文件: {combined_file}")

def create_summary_report(results, suffix=""):
    """创建变点信息汇总报告"""
    if not results:
        logging.warning("没有结果可以生成报告")
        return
    
    # 转换为DataFrame以便于分析
    df = pd.DataFrame(results)
    
    # 生成汇总报告
    report_path = os.path.join(OUTPUT_DIR, f"all_change_points_summary{suffix}.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 碳排放变点全类型分析汇总报告\n\n")
        
        f.write(f"## 总体统计\n")
        f.write(f"- 总分析变点数: {len(df)}\n")
        f.write(f"- 地区覆盖: {', '.join(df['region'].unique())}\n")
        f.write(f"- 部门覆盖: {', '.join(df['sector'].unique())}\n\n")
        
        f.write("## 按变点类型统计\n")
        for type_name in df['type'].unique():
            type_data = df[df['type'] == type_name]
            f.write(f"### {type_name}\n")
            f.write(f"- 变点总数: {len(type_data)}\n")
            f.write(f"- 涉及地区: {', '.join(type_data['region'].unique())}\n")
            f.write(f"- 涉及部门: {', '.join(type_data['sector'].unique())}\n")
            f.write(f"- 平均置信度: {type_data['confidence'].mean():.2f}\n\n")
        
        f.write("## 按地区统计\n")
        for region in df['region'].unique():
            region_data = df[df['region'] == region]
            f.write(f"### {region}\n")
            f.write(f"- 变点总数: {len(region_data)}\n")
            f.write(f"- 涉及部门: {', '.join(region_data['sector'].unique())}\n")
            
            # 按类型统计
            f.write(f"- 类型分布:\n")
            type_counts = region_data['type'].value_counts()
            for type_name, count in type_counts.items():
                f.write(f"  - {type_name}: {count}个\n")
            f.write("\n")
        
        f.write("## 按部门统计\n")
        for sector in df['sector'].unique():
            sector_data = df[df['sector'] == sector]
            f.write(f"### {sector}\n")
            f.write(f"- 变点总数: {len(sector_data)}\n")
            f.write(f"- 涉及地区: {', '.join(sector_data['region'].unique())}\n")
            
            # 按类型统计
            f.write(f"- 类型分布:\n")
            type_counts = sector_data['type'].value_counts()
            for type_name, count in type_counts.items():
                f.write(f"  - {type_name}: {count}个\n")
            f.write("\n")
        
        # 按时间分布的统计
        try:
            df['year'] = df['date'].apply(lambda x: x.split('-')[0] if '-' in x else x.split('/')[2] if '/' in x else 'unknown')
            yearly_counts = df.groupby('year').size()
            f.write("## 按年份分布\n")
            for year, count in yearly_counts.items():
                f.write(f"- {year}年: {count}个变点\n")
                
            # 按年份和类型的交叉统计
            f.write("\n## 年份-类型交叉统计\n")
            for year in sorted(df['year'].unique()):
                if year == 'unknown':
                    continue
                year_data = df[df['year'] == year]
                f.write(f"### {year}年\n")
                type_counts = year_data['type'].value_counts()
                for type_name, count in type_counts.items():
                    f.write(f"- {type_name}: {count}个\n")
                f.write("\n")
                
        except Exception as e:
            f.write(f"按时间分析出错: {str(e)}\n")
    
    logging.info(f"生成汇总报告: {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='处理变点文件并收集政策信息')
    parser.add_argument('--start', help='从哪个文件开始处理', type=str)
    parser.add_argument('--max', help='最多处理多少个文件', type=int)
    parser.add_argument('--region', help='指定处理的地区', type=str)
    parser.add_argument('--sector', help='指定处理的行业', type=str)
    parser.add_argument('--append', action='store_true', help='追加模式，不覆盖已有的合并信息文件')
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 配置日志
    configure_logging()
    
    # 如果指定了区域和行业，则只处理匹配的文件
    if args.region or args.sector:
        # 选择匹配的文件
        change_point_files = [f for f in os.listdir(CHANGE_POINTS_DIR) if f.endswith('_change_points.txt')]
        
        # 筛选符合条件的文件
        filtered_files = []
        for file in change_point_files:
            file_path = os.path.join(CHANGE_POINTS_DIR, file)
            metadata, _ = parse_change_point_file(file_path)
            
            if not metadata:
                continue
                
            # 匹配区域和行业
            region_match = not args.region or metadata['region'] == args.region
            sector_match = not args.sector or metadata['sector'] == args.sector
            
            if region_match and sector_match:
                filtered_files.append(file)
        
        if not filtered_files:
            logging.error(f"未找到匹配的文件: 区域={args.region}, 行业={args.sector}")
            sys.exit(1)
            
        logging.info(f"找到 {len(filtered_files)} 个匹配的文件")
        
        # 处理所有匹配的文件
        process_all_change_points(start_from=None, max_files=None, append_mode=args.append)
        
    else:
        # 原有的处理逻辑
        process_all_change_points(start_from=args.start, max_files=args.max, append_mode=args.append)
    logging.info(f"完成! 结果保存在: {OUTPUT_DIR}") 