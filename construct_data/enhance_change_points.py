#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import glob
from tqdm import tqdm
from scipy import stats
from scipy.signal import find_peaks
import ruptures as rpt

# 输入和输出目录
INPUT_DIR = '/Users/lizeyan/Desktop/climb/split_by_state_sector'
OUTPUT_DIR = '/Users/lizeyan/Desktop/climb/enhanced_change_points'

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

def detect_change_points_advanced(data, method='pelt', min_size=20, jump=5, penalty_value=2):
    """
    使用更先进的变点检测算法
    
    参数:
    - data: 时间序列数据
    - method: 检测方法，可选 'pelt', 'window', 'binseg'
    - min_size: 最小分段长度
    - jump: 窗口滑动步长
    - penalty_value: 惩罚参数，较小的值使算法更敏感
    
    返回:
    - change_points: 变点索引列表
    """
    # 数据标准化
    std_data = (data - np.mean(data)) / (np.std(data) if np.std(data) > 0 else 1)
    
    # 选择算法
    if method == 'pelt':
        algo = rpt.Pelt(model="rbf", min_size=min_size, jump=jump).fit(std_data)
        change_points = algo.predict(pen=penalty_value)
    elif method == 'window':
        algo = rpt.Window(model="rbf", width=40, min_size=min_size, jump=jump).fit(std_data)
        change_points = algo.predict(pen=penalty_value)
    elif method == 'binseg':
        algo = rpt.Binseg(model="rbf", min_size=min_size, jump=jump).fit(std_data)
        n_bkps = min(25, len(data)//min_size)  # 限制最大变点数
        change_points = algo.predict(n_bkps=n_bkps)
    else:
        raise ValueError(f"不支持的检测方法: {method}")
    
    # 移除最后一个点（可能是数据末尾）
    if change_points and change_points[-1] >= len(data) - 5:
        change_points = change_points[:-1]
    
    return change_points

def detect_local_extremes(data, window_size=30, threshold=1.5):
    """
    检测局部极值变点
    
    参数:
    - data: 时间序列数据
    - window_size: 滑动窗口大小
    - threshold: 检测阈值(标准差的倍数)
    
    返回:
    - change_points: 变点索引列表
    """
    # 计算滑动窗口内的局部极值
    extreme_scores = np.zeros(len(data))
    
    for i in range(window_size, len(data) - window_size):
        window = data[i-window_size:i+window_size]
        local_mean = np.mean(window)
        local_std = np.std(window)
        
        if local_std == 0:
            continue
        
        # 计算当前点相对于局部窗口的极值分数
        extreme_scores[i] = abs(data[i] - local_mean) / local_std
    
    # 找出得分高的点作为变点
    peaks, _ = find_peaks(extreme_scores, height=threshold, distance=window_size//2)
    
    return peaks.tolist()

def detect_change_points_ensemble(data, window_size=30):
    """
    集成多种方法进行变点检测
    
    参数:
    - data: 时间序列数据
    - window_size: 滑动窗口大小
    
    返回:
    - change_points: 变点索引列表
    """
    # 使用PELT算法（主要方法）
    cp1 = detect_change_points_advanced(data, method='pelt', min_size=15, penalty_value=1.5)
    
    # 使用局部极值检测（补充方法）
    cp2 = detect_local_extremes(data, window_size=window_size, threshold=1.2)
    
    # 计算一阶差分，寻找突变点
    diff = np.abs(np.diff(data))
    # 使用percentile而不是固定阈值
    threshold = np.percentile(diff, 90)  # 取90%分位数
    cp3 = [i+1 for i in range(len(diff)) if diff[i] > threshold]
    
    # 合并所有检测结果
    all_cp = sorted(list(set(cp1 + cp2 + cp3)))
    
    # 过滤太近的点（距离小于窗口大小的一半）
    if len(all_cp) > 1:
        filtered = [all_cp[0]]
        for i in range(1, len(all_cp)):
            if all_cp[i] - filtered[-1] >= window_size // 2:
                filtered.append(all_cp[i])
        return filtered
    
    return all_cp

# 变点特征分析
def analyze_change_point(data, change_point, window=30):
    """
    分析变点的特征
    
    参数:
    - data: 时间序列数据
    - change_point: 变点索引
    - window: 分析窗口大小
    
    返回:
    - 特征字典
    """
    if change_point < window or change_point >= len(data) - window:
        # 如果变点靠近数据起始或结束点，使用可用数据
        before = data[max(0, change_point-window):change_point]
        after = data[change_point:min(len(data), change_point+window)]
        if len(before) < 5 or len(after) < 5:  # 确保至少有足够的数据进行分析
            return None
    else:
        before = data[change_point-window:change_point]
        after = data[change_point:change_point+window]
    
    # 计算特征
    magnitude = abs(np.mean(after) - np.mean(before))
    
    # 计算相对变化幅度（相对于之前的数据范围）
    before_range = max(before) - min(before) if len(before) > 0 else 1
    relative_magnitude = magnitude / before_range if before_range > 0 else magnitude
    
    # 计算斜率变化
    try:
        slope_before = np.polyfit(np.arange(len(before)), before, 1)[0]
        slope_after = np.polyfit(np.arange(len(after)), after, 1)[0]
        slope_change = abs(slope_after - slope_before)
    except:
        slope_change = 0
    
    # 计算方差变化
    var_before = np.var(before) if len(before) > 1 else 0
    var_after = np.var(after) if len(after) > 1 else 0
    var_change = abs(var_after - var_before)
    
    # 计算时间序列稳定性
    stability = 1.0 / (1.0 + np.std(np.concatenate([before, after])))
    
    # 季节性检测 - 增强版
    # 1. 自相关检测（ACF）
    periodicity = 0
    seasonal_strength = 0
    
    # 如果数据足够长，计算自相关
    if len(before) > 10:
        # 计算ACF
        acf_before = np.correlate(before - np.mean(before), before - np.mean(before), mode='full')
        acf_before = acf_before[len(acf_before)//2:]
        
        # 标准化ACF
        if acf_before[0] > 0:  
            acf_norm = acf_before / acf_before[0]
            
            # 查找峰值（可能的周期）
            peaks, _ = find_peaks(acf_norm[1:], height=0.3)  # 只寻找高于0.3的峰
            
            if len(peaks) > 0:
                periodicity = max(0, np.max(acf_norm[1:][peaks]))  # 使用最强的周期信号
                
                # 周期峰值计数
                peak_count = sum(1 for p in acf_norm[1:] if p > 0.3)
                seasonal_strength = peak_count / len(acf_norm[1:]) if len(acf_norm[1:]) > 0 else 0
    
    # 2. 窗口间的相似性比较 - 检查前后窗口的模式相似性
    similar_pattern = 0
    if len(before) > 20 and len(after) > 20:
        # 计算前后窗口的相关性
        corr_matrix = np.corrcoef(before[:20], after[:20])
        if corr_matrix.size > 1:  # 确保矩阵足够大
            similar_pattern = max(0, corr_matrix[0, 1])  # 相关系数
    
    # 3. 季节性指标 - 固定长度窗口间的比较（例如相隔365/30天）
    year_cycle = 0
    month_cycle = 0
    
    # 检查年度周期
    if change_point >= 365 and change_point + 365 < len(data):
        year_window_prev = data[change_point-365:min(change_point-335, len(data))]
        year_window_next = data[change_point:min(change_point+30, len(data))]
        
        if len(year_window_prev) > 0 and len(year_window_next) > 0:
            # 年度窗口相关系数
            year_corr = np.corrcoef(year_window_prev, year_window_next[:len(year_window_prev)])
            if year_corr.size > 1:
                year_cycle = max(0, year_corr[0, 1])
    
    # 检查月度周期
    if change_point >= 30 and change_point + 30 < len(data):
        month_window_prev = data[change_point-30:change_point]
        month_window_next = data[change_point:change_point+30]
        
        if len(month_window_prev) == 30 and len(month_window_next) == 30:
            # 月度窗口相关系数
            month_corr = np.corrcoef(month_window_prev, month_window_next)
            if month_corr.size > 1:
                month_cycle = max(0, month_corr[0, 1])
    
    # 统计测试
    try:
        # KS检验（检验两个样本是否来自相同分布）
        ks_stat, ks_pvalue = stats.ks_2samp(before, after)
        
        # Mann-Whitney U检验（非参数检验，对异常值更稳健）
        mw_stat, mw_pvalue = stats.mannwhitneyu(before, after, alternative='two-sided')
        
        # t检验（检验两个样本均值是否相同）
        t_stat, t_pvalue = stats.ttest_ind(before, after, equal_var=False)
    except:
        ks_stat, ks_pvalue = 0, 1
        mw_stat, mw_pvalue = 0, 1
        t_stat, t_pvalue = 0, 1
    
    # 计算突变强度（变化前后数据的最大差异）
    jump_strength = abs(after[0] - before[-1]) if len(before) > 0 and len(after) > 0 else 0
    
    # 综合季节性得分
    seasonal_score = 0.4 * periodicity + 0.3 * similar_pattern + 0.15 * year_cycle + 0.15 * month_cycle
    
    return {
        'magnitude': magnitude,
        'relative_magnitude': relative_magnitude,
        'slope_change': slope_change,
        'var_change': var_change,
        'stability': stability,
        'periodicity': periodicity,
        'seasonal_strength': seasonal_strength,
        'similar_pattern': similar_pattern,
        'year_cycle': year_cycle,
        'month_cycle': month_cycle,
        'seasonal_score': seasonal_score,
        'jump_strength': jump_strength,
        'ks_stat': ks_stat,
        'ks_pvalue': ks_pvalue,
        'mw_stat': mw_stat / (len(before) * len(after)) if len(before) > 0 and len(after) > 0 else 0,
        't_stat': abs(t_stat),
        't_pvalue': t_pvalue
    }

# 变点分类 - 调整权重，增强季节性识别
def classify_change_point(features):
    """
    基于特征将变点分类
    
    参数:
    - features: 变点特征字典
    
    返回:
    - 分类结果和置信度
    """
    if not features:
        return "Unknown", 0.0
    
    # 定义不同类型的得分权重
    weights = {
        'Policy': {
            'magnitude': 0.15, 
            'relative_magnitude': 0.20,
            'slope_change': 0.10, 
            'var_change': 0.05, 
            'stability': 0.05,
            'periodicity': -0.15,  # 周期性越低越可能是政策变化
            'seasonal_strength': -0.05,
            'similar_pattern': -0.10,
            'year_cycle': -0.05,
            'month_cycle': -0.05,
            'seasonal_score': -0.20,  # 季节性得分越低越可能是政策
            'jump_strength': 0.15,
            'ks_stat': 0.05,
            'mw_stat': 0.05,
            't_stat': 0.05,
            'ks_pvalue': -0.05,
            'mw_pvalue': -0.05,
            't_pvalue': -0.05
        },
        'Event': {
            'magnitude': 0.10, 
            'relative_magnitude': 0.30,
            'slope_change': 0.05, 
            'var_change': 0.10, 
            'stability': 0.05,
            'periodicity': -0.10,
            'seasonal_strength': -0.05,
            'similar_pattern': -0.10,
            'year_cycle': -0.05,
            'month_cycle': -0.05,
            'seasonal_score': -0.15,  # 季节性得分越低越可能是事件
            'jump_strength': 0.25,  # 突变强度对事件影响很大
            'ks_stat': 0.05,
            'mw_stat': 0.10,
            't_stat': 0.10,
            'ks_pvalue': -0.02,
            'mw_pvalue': -0.02,
            't_pvalue': -0.02
        },
        'Seasonal': {
            'magnitude': 0.05, 
            'relative_magnitude': 0.05,
            'slope_change': 0.05, 
            'var_change': 0.05, 
            'stability': 0.05,
            'periodicity': 0.20,
            'seasonal_strength': 0.10,
            'similar_pattern': 0.15,
            'year_cycle': 0.15,
            'month_cycle': 0.15,
            'seasonal_score': 0.30,  # 季节性得分越高越可能是季节性变化
            'jump_strength': -0.10,  # 突变强度对季节性变化影响较小
            'ks_stat': -0.05,
            'mw_stat': -0.05,
            't_stat': -0.05,
            'ks_pvalue': 0.05,
            'mw_pvalue': 0.05,
            't_pvalue': 0.05
        },
        'Trend': {
            'magnitude': 0.05, 
            'relative_magnitude': 0.10,
            'slope_change': 0.40, 
            'var_change': 0.05, 
            'stability': 0.20,
            'periodicity': 0.05,
            'seasonal_strength': 0.00,
            'similar_pattern': -0.05,
            'year_cycle': 0.00,
            'month_cycle': 0.00,
            'seasonal_score': -0.05,
            'jump_strength': 0.05,
            'ks_stat': 0.02,
            'mw_stat': 0.02,
            't_stat': 0.05,
            'ks_pvalue': -0.02,
            'mw_pvalue': -0.02,
            't_pvalue': -0.02
        }
    }
    
    # 计算每种类型的得分
    scores = {}
    for cp_type, weight in weights.items():
        score = 0
        for feat, w in weight.items():
            if feat in features:
                score += w * features[feat]
        scores[cp_type] = max(0, score)  # 确保分数不为负
    
    # 获取得分最高的类型
    best_type = max(scores, key=scores.get)
    total_score = sum(s for s in scores.values() if s > 0)
    confidence = scores[best_type] / total_score if total_score > 0 else 0
    
    # 季节性特殊判断 - 如果季节性得分很高，直接归类为季节性
    if 'seasonal_score' in features and features['seasonal_score'] > 0.6:
        if best_type != 'Seasonal':
            # 如果季节性得分高但分类不是季节性，可能需要调整
            seasonal_conf = min(features['seasonal_score'], 0.95)
            if seasonal_conf > confidence:
                return "Seasonal", seasonal_conf
    
    # 统计学显著性检验：如果p值小于0.05，增加置信度
    if 'ks_pvalue' in features and features['ks_pvalue'] < 0.05:
        if best_type != 'Seasonal':  # 对季节性不增加这种加成
            confidence = min(confidence * 1.1, 1.0)
    
    if 't_pvalue' in features and features['t_pvalue'] < 0.05:
        if best_type != 'Seasonal':
            confidence = min(confidence * 1.05, 1.0)
    
    # 如果是突变强度很大的点，增加Event和Policy的得分
    if 'jump_strength' in features and features['jump_strength'] > 0.1:
        if best_type in ['Event', 'Policy']:
            confidence = min(confidence * 1.2, 1.0)
    
    return best_type, confidence

# 处理单个文件
def process_file(file_path, output_dir):
    print(f"Processing {file_path}...")
    
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        # 确保日期列转换为日期格式，使用正确的格式
        df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
        
        # 获取数值列
        values = df['value'].values
        
        # 检测变点 - 使用更灵敏的集成检测方法
        change_points = detect_change_points_ensemble(values, window_size=20)
        
        # 分析和分类变点
        all_points = []
        for cp in change_points:
            features = analyze_change_point(values, cp, window=20)  # 使用较小的窗口增加灵敏度
            if features:
                cp_type, confidence = classify_change_point(features)
                # 降低置信度阈值，包含所有类型的变点
                if confidence > 0.4:  # 降低置信度阈值
                    # 获取对应的日期
                    if 0 <= cp < len(df):
                        date = df['date'].iloc[cp]
                        # 将日期转换为字符串
                        date_str = date.strftime('%Y-%m-%d')
                        all_points.append({
                            'date': date_str,
                            'index': cp,
                            'type': cp_type,
                            'confidence': confidence,
                            'features': features
                        })
        
        # 根据置信度排序，取前10个
        MAX_POINTS = 10
        if len(all_points) > MAX_POINTS:
            print(f"Limiting to top {MAX_POINTS} points by confidence from {len(all_points)} detected points")
            # 按置信度降序排序，取前10个
            significant_points = sorted(all_points, key=lambda x: x['confidence'], reverse=True)[:MAX_POINTS]
            # 恢复原始顺序（按日期/索引排序）
            significant_points.sort(key=lambda x: x['index'])
        else:
            significant_points = all_points
        
        # 生成输出文件名
        base_name = os.path.basename(file_path)
        output_base = os.path.splitext(base_name)[0]
        
        # 保存结果
        output_file = os.path.join(output_dir, f"{output_base}_change_points.txt")
        with open(output_file, 'w') as f:
            f.write(f"File: {base_name}\n")
            f.write(f"Total change points detected: {len(change_points)}\n")
            f.write(f"Total significant points found: {len(all_points)}\n")
            f.write(f"Top significant points (limit={MAX_POINTS}): {len(significant_points)}\n\n")
            
            for i, point in enumerate(significant_points):
                f.write(f"Point {i+1}:\n")
                f.write(f"  Date: {point['date']}\n")
                f.write(f"  Type: {point['type']}\n")
                f.write(f"  Confidence: {point['confidence']:.2f}\n")
                f.write(f"  Features:\n")
                for feat, value in point['features'].items():
                    if not feat.endswith('pvalue'):  # 不输出p值，避免太多小数
                        f.write(f"    {feat}: {value:.4f}\n")
                    else:
                        # 对于p值，只显示是否显著
                        is_significant = "Yes" if value < 0.05 else "No"
                        f.write(f"    {feat}_significant: {is_significant}\n")
                f.write("\n")
        
        # 可视化
        plt.figure(figsize=(12, 6))
        plt.plot(df['date'], values)
        
        # 使用不同颜色标记不同类型的变点
        colors = {'Policy': 'r', 'Event': 'g', 'Seasonal': 'b', 'Trend': 'y', 'Unknown': 'k'}
        for point in significant_points:
            color = colors.get(point['type'], 'k')
            plt.axvline(x=pd.to_datetime(point['date']), color=color, linestyle='--', alpha=0.7)
            # 添加标签，但要错开以避免重叠
            y_pos = max(values) * (0.95 - (significant_points.index(point) % 10) * 0.02)
            plt.text(pd.to_datetime(point['date']), y_pos, 
                    f"{point['type']}\n{point['date']}", 
                    rotation=90, verticalalignment='top', color=color, fontsize=8)
        
        # 添加图例
        for cp_type, color in colors.items():
            plt.plot([], [], color=color, linestyle='--', label=cp_type)
        plt.legend()
        
        plt.title(f"Change Points - {output_base}")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.tight_layout()
        
        # 保存图像
        plt.savefig(os.path.join(output_dir, f"{output_base}_visualization.png"), dpi=300)
        plt.close()
        
        return len(significant_points)
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0

# 单个文件处理函数 - 专门针对航空行业
def process_aviation_file(file_path, output_dir):
    print(f"Processing aviation file {file_path}...")
    
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        # 确保日期列转换为日期格式，使用正确的格式
        df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
        
        # 获取数值列
        values = df['value'].values
        
        # 对于航空行业，使用不同的参数 - 更加灵敏
        # 使用ruptures的二分段算法，参数调整为更灵敏
        algo = rpt.Binseg(model="l2", min_size=10, jump=2).fit(values)
        n_bkps = min(50, len(values)//10)  # 允许更多的变点
        cp1 = algo.predict(n_bkps=n_bkps)
        
        # 直接使用一阶差分检测突变
        diff = np.abs(np.diff(values))
        threshold = np.percentile(diff, 85)  # 降低阈值，捕捉更多突变
        cp2 = [i+1 for i in range(len(diff)) if diff[i] > threshold]
        
        # 合并检测结果
        change_points = sorted(list(set(cp1 + cp2)))
        
        # 过滤太近的点（距离小于10）
        if len(change_points) > 1:
            filtered = [change_points[0]]
            for i in range(1, len(change_points)):
                if change_points[i] - filtered[-1] >= 10:
                    filtered.append(change_points[i])
            change_points = filtered
        
        # 分析和分类变点
        all_points = []
        for cp in change_points:
            if cp < len(values):  # 确保索引有效
                features = analyze_change_point(values, cp, window=15)  # 使用更小的窗口
                if features:
                    cp_type, confidence = classify_change_point(features)
                    # 航空数据使用更低的置信度阈值
                    if confidence > 0.3:  # 更低的阈值
                        date = df['date'].iloc[cp]
                        date_str = date.strftime('%Y-%m-%d')
                        all_points.append({
                            'date': date_str,
                            'index': cp,
                            'type': cp_type,
                            'confidence': confidence,
                            'features': features
                        })
        
        # 根据置信度排序，取前10个
        MAX_POINTS = 10
        if len(all_points) > MAX_POINTS:
            print(f"Limiting to top {MAX_POINTS} points by confidence from {len(all_points)} detected points")
            # 按置信度降序排序，取前10个
            significant_points = sorted(all_points, key=lambda x: x['confidence'], reverse=True)[:MAX_POINTS]
            # 恢复原始顺序（按日期/索引排序）
            significant_points.sort(key=lambda x: x['index'])
        else:
            significant_points = all_points
        
        # 生成输出文件
        base_name = os.path.basename(file_path)
        output_base = os.path.splitext(base_name)[0]
        
        # 保存结果和可视化（与通用处理类似）
        output_file = os.path.join(output_dir, f"{output_base}_change_points.txt")
        with open(output_file, 'w') as f:
            f.write(f"File: {base_name}\n")
            f.write(f"Total change points detected: {len(change_points)}\n")
            f.write(f"Total significant points found: {len(all_points)}\n")
            f.write(f"Top significant points (limit={MAX_POINTS}): {len(significant_points)}\n\n")
            
            for i, point in enumerate(significant_points):
                f.write(f"Point {i+1}:\n")
                f.write(f"  Date: {point['date']}\n")
                f.write(f"  Type: {point['type']}\n")
                f.write(f"  Confidence: {point['confidence']:.2f}\n")
                f.write(f"  Features:\n")
                for feat, value in point['features'].items():
                    if not feat.endswith('pvalue'):
                        f.write(f"    {feat}: {value:.4f}\n")
                    else:
                        is_significant = "Yes" if value < 0.05 else "No"
                        f.write(f"    {feat}_significant: {is_significant}\n")
                f.write("\n")
        
        # 可视化
        plt.figure(figsize=(14, 7))
        plt.plot(df['date'], values)
        
        # 标记变点
        colors = {'Policy': 'r', 'Event': 'g', 'Seasonal': 'b', 'Trend': 'y', 'Unknown': 'k'}
        for point in significant_points:
            color = colors.get(point['type'], 'k')
            plt.axvline(x=pd.to_datetime(point['date']), color=color, linestyle='--', alpha=0.7)
            # 添加标签，错开以避免重叠
            y_pos = max(values) * (0.95 - (significant_points.index(point) % 10) * 0.02)
            plt.text(pd.to_datetime(point['date']), y_pos, 
                   f"{point['type']}\n{point['date']}", 
                   rotation=90, verticalalignment='top', color=color, fontsize=7)
        
        # 添加图例
        for cp_type, color in colors.items():
            plt.plot([], [], color=color, linestyle='--', label=cp_type)
        plt.legend()
        
        plt.title(f"Change Points - {output_base}")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.tight_layout()
        
        # 保存图像
        plt.savefig(os.path.join(output_dir, f"{output_base}_visualization.png"), dpi=300)
        plt.close()
        
        return len(significant_points)
    
    except Exception as e:
        print(f"Error processing aviation file {file_path}: {e}")
        return 0

# 主函数修改，针对居民部门数据增加特殊处理
def main():
    # 获取所有CSV文件
    csv_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    
    print(f"Found {len(csv_files)} CSV files.")
    
    # 创建总结文件
    summary_file = os.path.join(OUTPUT_DIR, "change_points_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("Enhanced Change Points Detection Summary\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total files processed: {len(csv_files)}\n\n")
        f.write("File Name | Sector | Number of Significant Change Points\n")
        f.write("-"*70 + "\n")
    
    # 处理每个文件，根据不同行业使用不同处理方法
    total_significant_points = 0
    for file_path in tqdm(csv_files):
        try:
            base_name = os.path.basename(file_path)
            
            # 针对不同行业采用不同处理策略
            if 'Aviation' in base_name:
                # 航空行业 - 更敏感的检测
                significant_points = process_aviation_file(file_path, OUTPUT_DIR)
            elif 'Residential' in base_name:
                # 居民行业 - 处理季节性更强的数据
                significant_points = process_residential_file(file_path, OUTPUT_DIR)
            else:
                # 其他行业 - 标准处理
                significant_points = process_file(file_path, OUTPUT_DIR)
                
            total_significant_points += significant_points
            
            # 提取行业信息
            sector = "Unknown"
            if '_' in base_name:
                sector = base_name.split('_')[1].split('.')[0]
            
            # 更新总结文件
            with open(summary_file, 'a') as f:
                file_base = os.path.splitext(base_name)[0]
                f.write(f"{file_base} | {sector} | {significant_points}\n")
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            
            # 记录错误到总结文件
            with open(summary_file, 'a') as f:
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                f.write(f"{base_name} | ERROR | {str(e)}\n")
    
    # 保存汇总信息
    with open(os.path.join(OUTPUT_DIR, "summary.txt"), 'w') as f:
        f.write(f"Total files processed: {len(csv_files)}\n")
        f.write(f"Total significant change points found: {total_significant_points}\n")
    
    print("Enhanced processing complete. Results saved to:", OUTPUT_DIR)

# 添加居民行业特殊处理函数
def process_residential_file(file_path, output_dir):
    """特别为居民行业数据设计的处理函数，加强季节性识别"""
    print(f"Processing residential file {file_path}...")
    
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        # 确保日期列转换为日期格式
        df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
        
        # 获取数值列
        values = df['value'].values
        
        # 首先使用STL分解，提取季节性成分和趋势成分
        # 由于没有直接使用statsmodels，这里使用简单的方法模拟季节性识别
        # 1. 首先检测年度周期模式
        
        # 检测变点 - 对于居民数据，使用更保守的设置
        algo = rpt.Pelt(model="rbf", min_size=20, jump=5).fit(values)
        change_points = algo.predict(pen=3.0)  # 更大的惩罚参数，减少检出的变点
        
        # 补充使用更适合识别居民用能模式的方法
        # 计算季节性强度，过滤掉季节性太强的变点
        seasonal_indices = detect_seasonal_patterns(values, df['date'])
        
        # 过滤掉季节性变点（如果变点恰好落在季节性变化的时间点上）
        filtered_points = []
        for cp in change_points:
            # 检查是否是季节性索引
            if cp not in seasonal_indices and cp < len(values):
                filtered_points.append(cp)
        
        # 分析和分类变点
        all_points = []
        for cp in filtered_points:
            features = analyze_change_point(values, cp, window=30)  # 使用稍大的窗口
            if features:
                cp_type, confidence = classify_change_point(features)
                
                # 居民数据使用更严格的季节性判断
                is_seasonal = False
                if features.get('seasonal_score', 0) > 0.5:
                    is_seasonal = True
                    cp_type = 'Seasonal'
                    confidence = max(confidence, features['seasonal_score'])
                
                # 只保留非季节性的重要变点或高置信度的季节性变点
                if (not is_seasonal and confidence > 0.5) or (is_seasonal and confidence > 0.7):
                    if 0 <= cp < len(df):
                        date = df['date'].iloc[cp]
                        date_str = date.strftime('%Y-%m-%d')
                        all_points.append({
                            'date': date_str,
                            'index': cp,
                            'type': cp_type,
                            'confidence': confidence,
                            'features': features
                        })
        
        # 根据置信度排序，取10个
        MAX_POINTS = 10
        if len(all_points) > MAX_POINTS:
            print(f"Limiting to top {MAX_POINTS} points by confidence from {len(all_points)} detected points")
            # 按置信度降序排序，取前10个
            significant_points = sorted(all_points, key=lambda x: x['confidence'], reverse=True)[:MAX_POINTS]
            # 恢复原始顺序（按日期/索引排序）
            significant_points.sort(key=lambda x: x['index'])
        else:
            significant_points = all_points
        
        # 生成输出文件
        base_name = os.path.basename(file_path)
        output_base = os.path.splitext(base_name)[0]
        
        # 保存结果
        output_file = os.path.join(output_dir, f"{output_base}_change_points.txt")
        with open(output_file, 'w') as f:
            f.write(f"File: {base_name}\n")
            f.write(f"Total change points detected: {len(change_points)}\n")
            f.write(f"After filtering seasonal patterns: {len(filtered_points)}\n")
            f.write(f"Total significant points found: {len(all_points)}\n")
            f.write(f"Top significant points (limit={MAX_POINTS}): {len(significant_points)}\n\n")
            
            for i, point in enumerate(significant_points):
                f.write(f"Point {i+1}:\n")
                f.write(f"  Date: {point['date']}\n")
                f.write(f"  Type: {point['type']}\n")
                f.write(f"  Confidence: {point['confidence']:.2f}\n")
                f.write(f"  Features:\n")
                for feat, value in point['features'].items():
                    if not feat.endswith('pvalue'):
                        f.write(f"    {feat}: {value:.4f}\n")
                    else:
                        is_significant = "Yes" if value < 0.05 else "No"
                        f.write(f"    {feat}_significant: {is_significant}\n")
                f.write("\n")
        
        # 可视化
        plt.figure(figsize=(15, 8))
        plt.plot(df['date'], values)
        
        # 先标记季节性索引（但透明度较低）
        for si in seasonal_indices:
            if 0 <= si < len(df):
                plt.axvline(x=df['date'].iloc[si], color='lightgray', linestyle=':', alpha=0.2)
        
        # 标记变点
        colors = {'Policy': 'r', 'Event': 'g', 'Seasonal': 'b', 'Trend': 'y', 'Unknown': 'k'}
        for point in significant_points:
            color = colors.get(point['type'], 'k')
            plt.axvline(x=pd.to_datetime(point['date']), color=color, linestyle='--', alpha=0.7)
            # 添加标签，错开以避免重叠
            y_pos = max(values) * (0.95 - (significant_points.index(point) % 10) * 0.02)
            plt.text(pd.to_datetime(point['date']), y_pos, 
                    f"{point['type']}\n{point['date']}", 
                    rotation=90, verticalalignment='top', color=color, fontsize=8)
        
        # 添加图例
        for cp_type, color in colors.items():
            plt.plot([], [], color=color, linestyle='--', label=cp_type)
        plt.plot([], [], color='lightgray', linestyle=':', label='Seasonal Pattern', alpha=0.2)
        plt.legend()
        
        plt.title(f"Change Points - {output_base}")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.tight_layout()
        
        # 保存图像
        plt.savefig(os.path.join(output_dir, f"{output_base}_visualization.png"), dpi=300)
        plt.close()
        
        return len(significant_points)
    
    except Exception as e:
        print(f"Error processing residential file {file_path}: {e}")
        return 0

def detect_seasonal_patterns(values, dates):
    """
    检测季节性模式，返回可能是季节性变化的索引
    
    参数:
    - values: 时间序列数据
    - dates: 对应的日期序列
    
    返回:
    - seasonal_indices: 可能是季节性变化的索引列表
    """
    seasonal_indices = []
    
    # 1. 识别年周期的季节性模式 - 通常在相同月份有相似模式
    months = [d.month for d in dates]
    month_changes = [i for i in range(1, len(months)) if months[i] != months[i-1]]
    seasonal_indices.extend(month_changes)
    
    # 2. 使用自相关寻找可能的季节性周期
    if len(values) > 365:  # 至少一年的数据
        # 计算季节性差分 - 简单实现
        seasonal_diff = np.zeros(len(values) - 365)
        for i in range(len(seasonal_diff)):
            seasonal_diff[i] = abs(values[i+365] - values[i])
        
        # 找出季节性差分较小的点（表示季节性模式）
        threshold = np.percentile(seasonal_diff, 25)  # 取25%分位数作为阈值
        for i in range(len(seasonal_diff)):
            if seasonal_diff[i] < threshold:
                seasonal_indices.append(i)
                seasonal_indices.append(i + 365)  # 对应的下一年同期点
    
    # 3. 如果是住宅用电/气，冬夏季可能有明显的峰值
    # 检查6-8月和12-2月的变化点
    summer_winter_months = [1, 2, 6, 7, 8, 12]
    for i in range(1, len(dates)):
        if dates[i].month in summer_winter_months and dates[i-1].month not in summer_winter_months:
            seasonal_indices.append(i)
        elif dates[i].month not in summer_winter_months and dates[i-1].month in summer_winter_months:
            seasonal_indices.append(i)
    
    # 去重并排序
    return sorted(list(set(seasonal_indices)))

if __name__ == "__main__":
    main() 