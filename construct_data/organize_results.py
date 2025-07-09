#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import re
import logging
import glob
from datetime import datetime
import argparse

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"organize_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)

# 设置路径
BASE_DIR = "/Users/lizeyan/Desktop/climb"
CHANGE_POINTS_DIR = os.path.join(BASE_DIR, "enhanced_change_points")
POLICY_INFO_DIR = os.path.join(BASE_DIR, "policy_information")
OUTPUT_DIR = os.path.join(BASE_DIR, "organized_results")
# 可能的图像文件目录
IMAGE_DIRS = [
    os.path.join(BASE_DIR, "images"),
    os.path.join(BASE_DIR, "visualizations"),
    os.path.join(BASE_DIR, "plots"),
    os.path.join(BASE_DIR, "figures"),
    CHANGE_POINTS_DIR
]

def organize_files():
    """整理文件，将同一区域和行业的文件复制到同一文件夹中"""
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 获取所有CSV对应的合并信息文件
    combined_files = [f for f in os.listdir(POLICY_INFO_DIR) if f.endswith('_combined_info.txt')]
    
    for combined_file in combined_files:
        # 提取区域和行业
        match = re.match(r'([^_]+)_([^_]+)_combined_info\.txt', combined_file)
        if not match:
            logging.warning(f"无法从文件名解析区域和行业: {combined_file}")
            continue
            
        region, sector = match.groups()
        logging.info(f"处理 {region} 的 {sector} 部门")
        
        # 创建对应的文件夹
        target_dir = os.path.join(OUTPUT_DIR, f"{region}_{sector}")
        os.makedirs(target_dir, exist_ok=True)
        
        # 复制合并的TXT文件
        combined_src = os.path.join(POLICY_INFO_DIR, combined_file)
        combined_dst = os.path.join(target_dir, combined_file)
        shutil.copy2(combined_src, combined_dst)
        logging.info(f"已复制合并信息文件: {combined_file}")
        
        # 寻找并复制原始CSV文件
        csv_filename = f"{region}_{sector}.csv"
        csv_path = os.path.join(CHANGE_POINTS_DIR, f"../split_by_state_sector/{csv_filename}")
        if os.path.exists(csv_path):
            csv_dst = os.path.join(target_dir, csv_filename)
            shutil.copy2(csv_path, csv_dst)
            logging.info(f"已复制CSV文件: {csv_filename}")
        else:
            logging.warning(f"未找到CSV文件: {csv_path}")
        
        # 寻找并复制对应的转折点文件
        cp_filename = f"{region}_{sector}_change_points.txt"
        cp_path = os.path.join(CHANGE_POINTS_DIR, cp_filename)
        if os.path.exists(cp_path):
            cp_dst = os.path.join(target_dir, cp_filename)
            shutil.copy2(cp_path, cp_dst)
            logging.info(f"已复制转折点文件: {cp_filename}")
        else:
            logging.warning(f"未找到转折点文件: {cp_path}")
        
        # 查找并复制所有相关的单个转折点信息文件
        point_files = [f for f in os.listdir(POLICY_INFO_DIR) 
                      if f.startswith(f"{region}_{sector}_") and f.endswith('_info.txt') 
                      and not f.endswith('_combined_info.txt')]
        
        for point_file in point_files:
            point_src = os.path.join(POLICY_INFO_DIR, point_file)
            point_dst = os.path.join(target_dir, point_file)
            shutil.copy2(point_src, point_dst)
            logging.info(f"已复制转折点信息文件: {point_file}")
        
        # 查找并复制可能的图像文件
        copied_images = 0
        for image_dir in IMAGE_DIRS:
            if not os.path.exists(image_dir):
                continue
                
            # 查找可能的图像文件，支持多种格式和命名模式
            patterns = [
                f"{region}_{sector}*.png",
                f"{region}_{sector}*.jpg",
                f"{region}_{sector}*.jpeg",
                f"{region}_{sector}*.svg",
                f"{region}_{sector}*.pdf",
                f"{region}*{sector}*.png",
                f"{region}*{sector}*.jpg",
                f"{region}*{sector}*.jpeg",
                f"{region}*{sector}*.svg",
                f"{region}*{sector}*.pdf"
            ]
            
            for pattern in patterns:
                image_path = os.path.join(image_dir, pattern)
                image_files = glob.glob(image_path)
                
                for image_file in image_files:
                    image_filename = os.path.basename(image_file)
                    image_dst = os.path.join(target_dir, image_filename)
                    shutil.copy2(image_file, image_dst)
                    logging.info(f"已复制图像文件: {image_filename}")
                    copied_images += 1
        
        if copied_images == 0:
            logging.warning(f"未找到关于 {region}_{sector} 的图像文件")
        else:
            logging.info(f"共复制了 {copied_images} 个图像文件")
        
        logging.info(f"完成 {region}_{sector} 的文件整理")
    
    # 复制汇总报告
    summary_files = [f for f in os.listdir(POLICY_INFO_DIR) if 'summary' in f.lower()]
    for summary_file in summary_files:
        summary_src = os.path.join(POLICY_INFO_DIR, summary_file)
        summary_dst = os.path.join(OUTPUT_DIR, summary_file)
        shutil.copy2(summary_src, summary_dst)
        logging.info(f"已复制汇总报告: {summary_file}")
    
    # 复制其他可能的支持文件
    progress_file = os.path.join(POLICY_INFO_DIR, "processing_progress.txt")
    if os.path.exists(progress_file):
        progress_dst = os.path.join(OUTPUT_DIR, "processing_progress.txt")
        shutil.copy2(progress_file, progress_dst)
        logging.info(f"已复制处理进度文件")
    
    logging.info(f"所有文件整理完成，结果保存在: {OUTPUT_DIR}")
    return len(combined_files)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='整理政策信息和变点文件')
    parser.add_argument('--image-dirs', nargs='+', help='指定额外的图像文件目录')
    args = parser.parse_args()
    
    # 如果指定了额外的图像目录，添加到搜索列表中
    if args.image_dirs:
        for dir_path in args.image_dirs:
            if os.path.exists(dir_path):
                IMAGE_DIRS.append(dir_path)
                logging.info(f"添加图像目录: {dir_path}")
            else:
                logging.warning(f"指定的图像目录不存在: {dir_path}")
    
    start_time = datetime.now()
    logging.info(f"开始整理文件: {start_time}")
    
    file_count = organize_files()
    
    end_time = datetime.now()
    duration = end_time - start_time
    logging.info(f"文件整理完成，共处理了 {file_count} 个区域-行业组合")
    logging.info(f"开始时间: {start_time}")
    logging.info(f"结束时间: {end_time}")
    logging.info(f"总用时: {duration}") 