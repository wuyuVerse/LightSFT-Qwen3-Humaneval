#!/usr/bin/env bash

# 简单的 QwenCoder SFT 训练启动脚本
set -euo pipefail

# 定义路径
DATA_PATH="/volume/pt-train/users/wzhang/wjj-workspace/LightSFT-Qwen3-Humaneval/data/processed/sampled_data_20000.processed.npy"
PRETRAINED_MODEL="/volume/pt-train/models/Qwen2.5-7B"
OUTPUT_DIR="/volume/pt-train/users/wzhang/wjj-workspace/LightSFT-Qwen3-Humaneval/saves/qwen2.5-7b-sft"

# 确保输出目录存在
mkdir -p "${OUTPUT_DIR}"
mkdir -p ../../logs

# 生成带时间戳的日志文件名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="../../logs/qwencoder_training_${TIMESTAMP}.log"

echo "🚀 启动 QwenCoder SFT 训练..."
echo "📊 参数设置:"
echo "  - 数据路径: ${DATA_PATH}"
echo "  - 预训练模型: ${PRETRAINED_MODEL}"
echo "  - 输出目录: ${OUTPUT_DIR}"
echo "📝 日志文件: ${LOG_FILE}"
echo ""

# 设置环境变量改善内存管理
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

if [[ "$1" == "background" ]]; then
    echo "🔄 后台启动训练..."
    nohup bash ../../src/training/Qwen3-Coder/finetuning/sft/scripts/sft_qwencoder.sh \
        "${DATA_PATH}" "${PRETRAINED_MODEL}" "${OUTPUT_DIR}" \
        > "${LOG_FILE}" 2>&1 &
    
    TRAIN_PID=$!
    echo "✅ 训练已在后台启动！"
    echo "🆔 进程 PID: ${TRAIN_PID}"
    echo "📋 进程状态查看: ps aux | grep ${TRAIN_PID}"
    echo "🛑 终止训练: kill ${TRAIN_PID}"
    echo "🔄 查看日志: tail -f ${LOG_FILE}"
    
    # 等待几秒显示初始日志
    sleep 5
    if [[ -f "${LOG_FILE}" ]]; then
        echo ""
        echo "📄 最新日志内容:"
        echo "===================="
        tail -n 10 "${LOG_FILE}"
        echo "===================="
    fi
else
    echo "🔄 前台启动训练..."
    bash ../../src/training/Qwen3-Coder/finetuning/sft/scripts/sft_qwencoder.sh \
        "${DATA_PATH}" "${PRETRAINED_MODEL}" "${OUTPUT_DIR}" \
        2>&1 | tee "${LOG_FILE}"
fi
