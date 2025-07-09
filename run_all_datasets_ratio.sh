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
TRAIN_RATIOS=(0.2 0.3 0.4 0.5 0.6 0.7 0.8)  # 要测试的训练集比例
VAL_RATIO=0.1  # 固定验证集比例

# 创建日志和结果目录
LOG_DIR="./logs"
RESULT_DIR="./"
mkdir -p $LOG_DIR

# 获取当前时间作为运行ID
RUN_ID=$(date +"%Y%m%d_%H%M%S")
echo "运行ID: $RUN_ID"

# 查找所有数据集目录
DATASET_DIRS=$(find ./dataset -maxdepth 1 -mindepth 1 -type d)
TOTAL_DATASETS=$(echo "$DATASET_DIRS" | wc -l)

# 创建结果文件
RESULT_FILE="$RESULT_DIR/result_long_term_forecast.txt"
echo "# 时间序列预测结果 - 训练集比例对比实验" > $RESULT_FILE
echo "# 运行时间: $(date)" >> $RESULT_FILE
echo "# 模型: ${MODELS[*]}" >> $RESULT_FILE
echo "# 预测长度: ${PRED_LENS[*]}" >> $RESULT_FILE
echo "# 训练集比例: ${TRAIN_RATIOS[*]}" >> $RESULT_FILE
echo "# 验证集比例: $VAL_RATIO" >> $RESULT_FILE
echo "" >> $RESULT_FILE

# 为每个训练集比例运行实验
for TRAIN_RATIO in "${TRAIN_RATIOS[@]}"; do
    # 计算测试集比例
    TEST_RATIO=$(python -c "print(1 - $TRAIN_RATIO - $VAL_RATIO)")
    
    echo -e "\n\n===== 训练集比例: $TRAIN_RATIO (验证集: $VAL_RATIO, 测试集: $TEST_RATIO) =====" | tee -a $RESULT_FILE
    
    # 为每个模型运行实验
    for MODEL in "${MODELS[@]}"; do
        echo -e "\n## 模型: $MODEL" | tee -a $RESULT_FILE
        
        # 为当前模型和训练集比例创建结果汇总文件
        SUMMARY_FILE="$LOG_DIR/summary_${MODEL}_train${TRAIN_RATIO}_$RUN_ID.csv"
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
                echo "[$MODEL] 处理数据集 $CURRENT/$TOTAL_DATASETS: $DATASET_NAME (训练集: $TRAIN_RATIO, 预测长度: $PRED_LEN)"
                
                # 设置日志文件
                LOG_FILE="$LOG_DIR/${MODEL}_${DATASET_NAME}_T${TRAIN_RATIO}_L${PRED_LEN}_$RUN_ID.log"
                
                echo "开始处理 $DATASET_NAME (训练集: $TRAIN_RATIO, 预测长度: $PRED_LEN), 日志文件: $LOG_FILE"
                
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
                    --train_ratio $TRAIN_RATIO \
                    --val_ratio $VAL_RATIO \
                    --model_id "${MODEL}_${DATASET_NAME}_T${TRAIN_RATIO}_L${PRED_LEN}_$RUN_ID" 2>&1 | tee $LOG_FILE
                
                # 检查运行是否成功
                if [ $? -eq 0 ]; then
                    echo "$DATASET_NAME (训练集: $TRAIN_RATIO, 预测长度: $PRED_LEN) 运行成功"
                    
                    # 从日志中提取指标
                    MSE=$(grep -oP 'MSE: \\K[0-9.]+' $LOG_FILE | tail -1)
                    MAE=$(grep -oP 'MAE: \\K[0-9.]+' $LOG_FILE | tail -1)
                    RMSE=$(grep -oP 'RMSE: \\K[0-9.]+' $LOG_FILE | tail -1)
                    MAPE=$(grep -oP 'MAPE: \\K[0-9.]+' $LOG_FILE | tail -1)
                    MSPE=$(grep -oP 'MSPE: \\K[0-9.]+' $LOG_FILE | tail -1)
                    RUNTIME=$(grep -oP '训练和测试总耗时: \\K[0-9.]+' $LOG_FILE | tail -1)
                    
                    # 写入汇总文件
                    echo "$DATASET_NAME,$PRED_LEN,$MSE,$MAE,$RMSE,$MAPE,$MSPE,$RUNTIME" >> $SUMMARY_FILE
                    
                    # 写入结果文件
                    echo "### 数据集: $DATASET_NAME, 训练集: $TRAIN_RATIO, 预测长度: $PRED_LEN" >> $RESULT_FILE
                    echo "- MSE: $MSE" >> $RESULT_FILE
                    echo "- MAE: $MAE" >> $RESULT_FILE
                    echo "- RMSE: $RMSE" >> $RESULT_FILE
                    echo "- MAPE: $MAPE" >> $RESULT_FILE
                    echo "- MSPE: $MSPE" >> $RESULT_FILE
                    echo "- 运行时间: $RUNTIME 秒" >> $RESULT_FILE
                    echo "" >> $RESULT_FILE
                else
                    echo "$DATASET_NAME (训练集: $TRAIN_RATIO, 预测长度: $PRED_LEN) 运行失败，请检查日志: $LOG_FILE"
                    echo "$DATASET_NAME,$PRED_LEN,ERROR,,,,," >> $SUMMARY_FILE
                    
                    # 写入错误信息到结果文件
                    echo "### 数据集: $DATASET_NAME, 训练集: $TRAIN_RATIO, 预测长度: $PRED_LEN" >> $RESULT_FILE
                    echo "- 错误: 运行失败，请查看日志文件: $LOG_FILE" >> $RESULT_FILE
                    echo "" >> $RESULT_FILE
                fi
                
                echo "----------------------------------------"
            done  # 结束预测长度循环
        done  # 结束数据集循环
    done  # 结束模型循环
done  # 结束训练集比例循环

echo -e "\n===== 所有实验运行完成 =====" | tee -a $RESULT_FILE
echo "结果已保存到: $RESULT_FILE"