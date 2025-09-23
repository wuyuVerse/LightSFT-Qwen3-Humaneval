#!/bin/bash

# 同步所有offline wandb runs到指定项目
PROJECT_NAME=${1:-"LightSFT-Qwen3-Humaneval"}

echo "Syncing wandb runs to project: $PROJECT_NAME"

# 查找所有offline run目录并同步
for run_dir in ./wandb_logs/wandb/offline-run-*/; do
    if [ -d "$run_dir" ]; then
        echo "Syncing: $run_dir"
        wandb sync "$run_dir" --project "$PROJECT_NAME"
    fi
done

echo "All runs synced successfully!"
