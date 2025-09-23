#!/usr/bin/env python3
"""
设置离线环境，让EvalPlus能够使用本地数据集
"""

import os
import sys
from pathlib import Path

def setup_offline_env():
    """设置离线环境变量"""
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    
    # 设置环境变量，让EvalPlus使用本地数据集
    os.environ['EVALPLUS_CACHE_DIR'] = str(data_dir)
    os.environ['HF_HOME'] = str(data_dir / "hf_cache")
    
    # 创建必要的目录
    (data_dir / "hf_cache").mkdir(parents=True, exist_ok=True)
    
    print(f"✓ 设置离线环境")
    print(f"  EVALPLUS_CACHE_DIR: {data_dir}")
    print(f"  HF_HOME: {data_dir / 'hf_cache'}")
    
    return True

def main():
    """主函数"""
    print("=== 设置离线环境 ===")
    
    if setup_offline_env():
        print("✓ 离线环境设置完成!")
        print("现在可以运行评测脚本了:")
        print("python scripts/evaluate_model.py")
        return 0
    else:
        print("✗ 离线环境设置失败!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
