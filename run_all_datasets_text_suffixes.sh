#!/bin/bash

# 设置基本参数
MODELS=("FEDformerWithText")  # 要运行的模型列表
USE_EVENT_TEXT=1  # 是否使用事件文本

# 定义所有事件文本变体及其后缀
EVENT_TEXT_SUFFIXES=(
    "_combined_info.txt"  # 原始版本
    "_formal.txt"        # 正式风格
    "_casual.txt"        # 口语化表达
    "_detailed.txt"      # 详细扩展
    "_concise.txt"       # 简洁版本
    "_synonym.txt"       # 同义词替换
    "_passive.txt"       # 被动语态
)

# 为不同的事件文本变体添加描述性名称
declare -A EVENT_TEXT_NAMES
EVENT_TEXT_NAMES[_combined_info.txt]="original"
EVENT_TEXT_NAMES[_formal.txt]="formal"
EVENT_TEXT_NAMES[_casual.txt]="casual"
EVENT_TEXT_NAMES[_detailed.txt]="detailed"
EVENT_TEXT_NAMES[_concise.txt]="concise"
EVENT_TEXT_NAMES[_synonym.txt]="synonym"
EVENT_TEXT_NAMES[_passive.txt]="passive"

# 其他训练参数
GPU_TYPE="cuda"
GPU=0
EPOCHS=10  # 训练轮数
PATIENCE=3  # 早停耐心值
SEQ_LEN=48  # 输入序列长度
PRED_LENS=(6 12 24 48)  # 要测试的预测长度列表

# 创建日志目录
LOG_DIR="./logs"
mkdir -p $LOG_DIR

# 获取当前时间作为运行ID
RUN_ID=$(date +"%Y%m%d_%H%M%S")
echo "运行ID: $RUN_ID"

# 查找所有数据集目录
DATASET_DIRS=$(find ./dataset -maxdepth 1 -mindepth 1 -type d)
TOTAL_DATASETS=$(echo "$DATASET_DIRS" | wc -l)

# 为每个模型运行实验
for MODEL in "${MODELS[@]}"; do
    echo -e "\n\n===== 开始运行 $MODEL 模型 ====="
    
    # 为当前模型创建结果汇总文件
    SUMMARY_FILE="$LOG_DIR/summary_${MODEL}_$RUN_ID.csv"
    echo "数据集,预测长度,MSE,MAE,RMSE,MAPE,MSPE,运行时间(秒)" > $SUMMARY_FILE
    
    # 计数器
    CURRENT=0
    
    for DATASET_DIR in $DATASET_DIRS; do
        # 提取数据集名称
        DATASET_NAME=$(basename $DATASET_DIR)
        
        # 检查是否有CSV文件
        CSV_FILE="$DATASET_DIR/$DATASET_NAME.csv"
        if [ ! -f "$CSV_FILE" ]; then
            echo "警告: $DATASET_NAME 没有找到CSV文件，跳过"
            continue
        fi
        
        # 更新计数器
        CURRENT=$((CURRENT + 1))
        
        # 为每个预测长度运行实验
        for PRED_LEN in "${PRED_LENS[@]}"; do
            echo "[$MODEL] 处理数据集 $CURRENT/$TOTAL_DATASETS: $DATASET_NAME (预测长度: $PRED_LEN)"
            
            # 设置日志文件
            LOG_FILE="$LOG_DIR/${MODEL}_${DATASET_NAME}_L${PRED_LEN}_$RUN_ID.log"
            
            echo "开始处理 $DATASET_NAME (预测长度: $PRED_LEN), 日志文件: $LOG_FILE"
            
            # 为每个事件文本变体运行实验
            for EVENT_SUFFIX in "${EVENT_TEXT_SUFFIXES[@]}"; do
                # 从映射中获取变体的描述性名称
                EVENT_NAME=${EVENT_TEXT_NAMES[$EVENT_SUFFIX]:-$EVENT_SUFFIX}
                
                # 构建模型ID，包含所有相关配置
                MODEL_ID="${MODEL}_${DATASET_NAME}_L${PRED_LEN}_E${EVENT_NAME}_${RUN_ID}"
                
                # 设置日志文件
                LOG_FILE="$LOG_DIR/${MODEL_ID}.log"
                
                echo "开始处理 $DATASET_NAME (预测长度: $PRED_LEN, 事件文本: $EVENT_NAME), 日志文件: $LOG_FILE"
                
                # 运行模型
                python run.py \
                    --model $MODEL \
                    --data custom \
                    --root_path $DATASET_DIR \
                    --data_path $DATASET_NAME.csv \
                    --use_event_text $USE_EVENT_TEXT \
                    --event_text_suffix "$EVENT_SUFFIX" \
                    --gpu_type $GPU_TYPE \
                    --gpu $GPU \
                    --train_epochs $EPOCHS \
                    --patience $PATIENCE \
                    --seq_len $SEQ_LEN \
                    --pred_len $PRED_LEN \
                    --model_id "$MODEL_ID" 2>&1 | tee $LOG_FILE
                
            done
            
            echo "----------------------------------------"
        done  # 结束预测长度循环
    done  # 结束数据集循环
done  # 结束模型循环

echo -e "\n===== 所有模型运行完成 ====="