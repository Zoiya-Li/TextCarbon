#!/bin/bash

# 设置基本参数
MODEL="FEDformerAblation"  # 使用消融模型
ABLATION_TYPES=("no_text" "frozen_bert" "additive" "late_fusion" "full")  # 消融实验类型
USE_EVENT_TEXT=1
GPU_TYPE="cuda"
GPU=0
EPOCHS=10  # 训练轮数
PATIENCE=3  # 早停耐心值
SEQ_LEN=48  # 输入序列长度
PRED_LENS=(6 12 24 48)  # 预测长度列表

# 创建日志目录
LOG_DIR="./logs/ablation"
mkdir -p $LOG_DIR

# 获取当前时间作为运行ID
RUN_ID=$(date +"%Y%m%d_%H%M%S")
echo "消融实验运行ID: $RUN_ID"

# 查找所有数据集目录
DATASET_DIRS=$(find ./dataset -maxdepth 1 -mindepth 1 -type d)
TOTAL_DATASETS=$(echo "$DATASET_DIRS" | wc -l)

# 为每个消融类型和预测长度运行实验
for PRED_LEN in "${PRED_LENS[@]}"; do
    echo -e "\n\n===== 开始预测长度: $PRED_LEN ====="
    
    for ABLATION in "${ABLATION_TYPES[@]}"; do
        echo -e "\n===== 开始消融实验: $ABLATION (预测长度: $PRED_LEN) ====="
        
        # 为当前消融类型和预测长度创建结果汇总文件
        SUMMARY_FILE="$LOG_DIR/summary_L${PRED_LEN}_${ABLATION}_$RUN_ID.csv"
        echo "数据集,预测长度,MSE,MAE,RMSE,MAP,MSPE,运行时间(秒)" > $SUMMARY_FILE
        
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
            echo -e "\n[${CURRENT}/${TOTAL_DATASETS}] 处理数据集: $DATASET_NAME (消融: $ABLATION, 预测长度: $PRED_LEN)"
            
            # 创建日志文件
            LOG_FILE="$LOG_DIR/${DATASET_NAME}_L${PRED_LEN}_${ABLATION}_$RUN_ID.log"
            
            # 运行模型
            echo "数据文件夹: $DATASET_NAME"
            echo "事件文本目录: $DATASET_DIR"
            echo "消融类型: $ABLATION"
            echo "预测长度: $PRED_LEN"
            
            # 设置事件文本后缀
            if [ "$ABLATION" == "no_text" ]; then
                EVENT_TEXT_ARG="--use_event_text 0"
            else
                EVENT_TEXT_ARG="--use_event_text 1"
            fi
            
            # 设置消融类型描述
            case $ABLATION in
                "no_text")
                    ABLATION_DESC="no_text"
                    ;;
                "frozen_bert")
                    ABLATION_DESC="frozen_bert"
                    ;;
                "additive")
                    ABLATION_DESC="fuse_add"
                    ;;
                "late_fusion")
                    ABLATION_DESC="fuse_late"
                    ;;
                "full")
                    ABLATION_DESC="fuse_gate"
                    ;;
                *)
                    ABLATION_DESC=$ABLATION
                    ;;
            esac
            
            # 运行命令
            python -u run.py \
                --model $MODEL \
                --ablation_type $ABLATION \
                --data custom \
                --root_path $DATASET_DIR \
                --data_path $DATASET_NAME.csv \
                $EVENT_TEXT_ARG \
                --gpu_type $GPU_TYPE \
                --gpu $GPU \
                --train_epochs $EPOCHS \
                --patience $PATIENCE \
                --seq_len $SEQ_LEN \
                --pred_len $PRED_LEN \
                --model_id "${MODEL}_${DATASET_NAME}_L${PRED_LEN}_${ABLATION_DESC}" \
                --des "Ablation_L${PRED_LEN}_${ABLATION_DESC}" 2>&1 | tee $LOG_FILE
            
            # 记录运行状态
            if [ ${PIPESTATUS[0]} -eq 0 ]; then
                echo "$DATASET_NAME 完成 (消融: $ABLATION, 预测长度: $PRED_LEN)"
            else
                echo "$DATASET_NAME 失败 (消融: $ABLATION, 预测长度: $PRED_LEN)"
            fi
        done
        
        echo -e "\n===== 消融实验 $ABLATION (预测长度: $PRED_LEN) 完成 ====="
    done
    
    echo -e "\n===== 预测长度 $PRED_LEN 的所有实验完成 ====="
done

echo -e "\n\n===== 所有消融实验完成 ====="
echo "结果已保存到 $LOG_DIR"
