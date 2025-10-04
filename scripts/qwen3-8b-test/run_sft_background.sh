#!/usr/bin/env bash

# 后台训练启动脚本 - 自动处理 OOM 和环境设置
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# 默认参数 (可通过环境变量覆盖)
INPUT_PATH=${INPUT_PATH:-"/volume/pt-train/users/wzhang/wjj-workspace/LightSFT-Qwen3-Humaneval/data/sampled_data_20000.jsonl"}
OUTPUT_PATH=${OUTPUT_PATH:-"/volume/pt-train/users/wzhang/wjj-workspace/LightSFT-Qwen3-Humaneval/data/processed/sampled_data_20000.processed"}
TOKENIZER_PATH=${TOKENIZER_PATH:-"/volume/pt-train/models/Qwen2.5-7B"}
DATA_PATH=${DATA_PATH:-"/volume/pt-train/users/wzhang/wjj-workspace/LightSFT-Qwen3-Humaneval/data/processed/sampled_data_20000.processed.npy"}
PRETRAINED_MODEL=${PRETRAINED_MODEL:-"/volume/pt-train/models/Qwen2.5-7B"}
OUTPUT_DIR=${OUTPUT_DIR:-"/volume/pt-train/users/wzhang/wjj-workspace/LightSFT-Qwen3-Humaneval/saves/qwen2.5-7b-sft"}

# 训练参数 (适合 8x V100 140GB 显存)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}
export NPROC_PER_NODE=${NPROC_PER_NODE:-8}
export TF32=${TF32:-False}
export BF16=${BF16:-True}
export MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-2}     # 降低到 2 避免 OOM
export BATCH_SIZE=${BATCH_SIZE:-64}               # 对应减小全局 batch
export MAX_LENGTH=${MAX_LENGTH:-1280}              # 序列长度从 1280 -> 1024
export EPOCHS=${EPOCHS:-3}

# 设置环境变量改善内存管理
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# 生成带时间戳的日志文件名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${PROJECT_ROOT}/logs/qwen_sft_train_${TIMESTAMP}.log"

# 确保日志目录存在
mkdir -p "${PROJECT_ROOT}/logs"

echo "🚀 启动后台训练..."
echo "📊 参数设置:"
echo "  - GPU: ${CUDA_VISIBLE_DEVICES}"
echo "  - 每卡 batch size: ${MICRO_BATCH_SIZE}"
echo "  - 全局 batch size: ${BATCH_SIZE}"
echo "  - 序列长度: ${MAX_LENGTH}"
echo "  - 训练轮数: ${EPOCHS}"
echo "📝 日志文件: ${LOG_FILE}"
echo "🔄 可通过以下命令查看训练进度:"
echo "  tail -f ${LOG_FILE}"
echo ""

# 使用 nohup 在后台运行训练
nohup bash "${SCRIPT_DIR}/sft_minimal.sh" train > "${LOG_FILE}" 2>&1 &

# 获取进程 PID
TRAIN_PID=$!
echo "✅ 训练已在后台启动！"
echo "🆔 进程 PID: ${TRAIN_PID}"
echo "📋 进程状态查看: ps aux | grep ${TRAIN_PID}"
echo "🛑 终止训练: kill ${TRAIN_PID}"
echo ""
echo "⏰ 等待 5 秒后显示初始日志..."
sleep 5

# 显示最初几行日志确认启动成功
if [[ -f "${LOG_FILE}" ]]; then
    echo "📄 最新日志内容:"
    echo "===================="
    tail -n 20 "${LOG_FILE}"
    echo "===================="
    echo ""
    echo "💡 继续监控: tail -f ${LOG_FILE}"
else
    echo "⚠️  日志文件尚未生成，请稍等..."
fi
