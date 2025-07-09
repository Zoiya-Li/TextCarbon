#!/bin/bash

# 设置基本参数
MODELS=("FEDformerWithText")  # 要运行的模型列表
USE_EVENT_TEXT=1
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
            
            # 运行模型
            python run.py \
                --model $MODEL \
                --data custom \
                --root_path $DATASET_DIR \
                --data_path $DATASET_NAME.csv \
                --use_event_text $USE_EVENT_TEXT \
                --gpu_type $GPU_TYPE \
                --gpu $GPU \
                --train_epochs $EPOCHS \
                --patience $PATIENCE \
                --seq_len $SEQ_LEN \
                --pred_len $PRED_LEN \
                --model_id "${MODEL}_${DATASET_NAME}_L${PRED_LEN}_$RUN_ID" 2>&1 | tee $LOG_FILE
            
            # 检查运行是否成功
            if [ $? -eq 0 ]; then
                echo "$DATASET_NAME (预测长度: $PRED_LEN) 运行成功"
                
                # 从日志中提取指标
                MSE=$(grep -oP 'MSE: \\K[0-9.]+' $LOG_FILE | tail -1)
                MAE=$(grep -oP 'MAE: \\K[0-9.]+' $LOG_FILE | tail -1)
                RMSE=$(grep -oP 'RMSE: \\K[0-9.]+' $LOG_FILE | tail -1)
                MAPE=$(grep -oP 'MAPE: \\K[0-9.]+' $LOG_FILE | tail -1)
                MSPE=$(grep -oP 'MSPE: \\K[0-9.]+' $LOG_FILE | tail -1)
                RUNTIME=$(grep -oP '训练和测试总耗时: \\K[0-9.]+' $LOG_FILE | tail -1)
                
                # 写入汇总文件，包含预测长度
                echo "$DATASET_NAME,$PRED_LEN,$MSE,$MAE,$RMSE,$MAPE,$MSPE,$RUNTIME" >> $SUMMARY_FILE
            else
                echo "$DATASET_NAME (预测长度: $PRED_LEN) 运行失败，请检查日志: $LOG_FILE"
                echo "$DATASET_NAME,$PRED_LEN,ERROR,,,,," >> $SUMMARY_FILE
            fi
            
            echo "----------------------------------------"
        done  # 结束预测长度循环
    done  # 结束数据集循环
done  # 结束模型循环

echo -e "\n===== 所有模型运行完成 ====="