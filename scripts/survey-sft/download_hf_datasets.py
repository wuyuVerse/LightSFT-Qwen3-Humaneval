#!/usr/bin/env python3
"""
批量下载HuggingFace代码数据集脚本
自动下载多个代码相关数据集到独立文件夹
"""

import os
import sys
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download
from datetime import datetime


# 定义所有需要下载的数据集
DATASETS = [
    "codeparrot/apps",
    "mlfoundations-dev/stackexchange_codereview",
    "nampdn-ai/tiny-codes",
    "bigcode/commitpackft",
    "deepmind/code_contests",
    "SenseLLM/ReflectionSeq-GPT",
    "MatrixStudio/Codeforces-Python-Submissions",
    "Magpie-Align/Magpie-Qwen2.5-Coder-Pro-300K-v0.1",
    "bigcode/self-oss-instruct-sc2-exec-filter-50k",
    "PrimeIntellect/real-world-swe-problems",
    "lvwerra/stack-exchange-paired",
    "cfahlgren1/react-code-instructions",
    "PrimeIntellect/stackexchange-question-answering",
    "PrimeIntellect/SYNTHETIC-2-SFT-verified",
    "bugdaryan/sql-create-context-instruction",
]


def download_dataset(dataset_name: str, data_root: Path, resume: bool = True, 
                     skip_existing: bool = False, auto_yes: bool = False) -> bool:
    """
    下载单个数据集到指定目录
    
    Args:
        dataset_name: HuggingFace数据集名称 (格式: org/dataset)
        data_root: 数据根目录
        resume: 是否支持断点续传
        skip_existing: 自动跳过已存在的数据集
        auto_yes: 非交互模式，自动确认所有操作
    
    Returns:
        下载是否成功
    """
    # 提取数据集短名称作为文件夹名（去掉org前缀）
    folder_name = dataset_name.split('/')[-1]
    target_dir = data_root / folder_name
    
    print(f"\n{'='*80}")
    print(f"📦 数据集: {dataset_name}")
    print(f"📁 目标目录: {target_dir}")
    print(f"{'='*80}")
    
    # 检查是否已存在
    if target_dir.exists() and resume:
        print(f"⚠️  目录已存在: {target_dir}")
        if skip_existing or auto_yes:
            print("⏭️  自动跳过（目录已存在）")
            return True
        elif not sys.stdin.isatty():
            # 后台执行时默认跳过已存在的
            print("⏭️  后台模式：自动跳过已存在的数据集")
            return True
        else:
            response = input("是否跳过此数据集？(y/n，默认n): ").strip().lower()
            if response == 'y':
                print("⏭️  跳过")
                return True
    
    try:
        # 使用snapshot_download下载整个数据集
        start_time = datetime.now()
        print(f"⏱️  开始下载: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        downloaded_path = snapshot_download(
            repo_id=dataset_name,
            repo_type="dataset",
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,  # 直接下载文件，不使用符号链接
            resume_download=resume,  # 支持断点续传
            max_workers=4,  # 并行下载线程数
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"✅ 下载成功: {downloaded_path}")
        print(f"⏱️  耗时: {duration:.2f}秒 ({duration/60:.2f}分钟)")
        
        # 显示文件夹大小
        total_size = sum(f.stat().st_size for f in target_dir.rglob('*') if f.is_file())
        print(f"📊 数据集大小: {total_size / (1024**3):.2f} GB")
        
        return True
        
    except Exception as e:
        print(f"❌ 下载失败: {dataset_name}")
        print(f"错误信息: {str(e)}")
        return False


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="批量下载HuggingFace代码数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 交互模式下载所有数据集
  python scripts/download_hf_datasets.py
  
  # 后台非交互模式（自动跳过已存在的数据集）
  nohup python scripts/download_hf_datasets.py --auto-yes &
  
  # 下载指定数据集
  python scripts/download_hf_datasets.py --datasets codeparrot/apps bigcode/commitpackft
  
  # 跳过已存在的数据集
  python scripts/download_hf_datasets.py --skip-existing
        """
    )
    parser.add_argument(
        "--auto-yes", "-y",
        action="store_true",
        help="非交互模式，自动确认所有操作（适合后台执行）"
    )
    parser.add_argument(
        "--skip-existing", "-s",
        action="store_true",
        help="自动跳过已存在的数据集"
    )
    parser.add_argument(
        "--datasets", "-d",
        nargs="+",
        help="指定要下载的数据集（默认下载所有）"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        help="数据保存目录（默认: <project_root>/data）"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="并行下载线程数（默认: 4）"
    )
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    print("="*80)
    print("🚀 HuggingFace代码数据集批量下载工具")
    print("="*80)
    
    # 确定数据根目录 (从 scripts/survey-sft/ 到项目根目录)
    project_root = Path(__file__).parent.parent.parent
    if args.data_dir:
        data_root = Path(args.data_dir)
    else:
        data_root = project_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    
    # 确定要下载的数据集
    if args.datasets:
        datasets_to_download = args.datasets
        print(f"\n📋 指定下载 {len(datasets_to_download)} 个数据集")
    else:
        datasets_to_download = DATASETS
        print(f"\n📋 待下载数据集数量: {len(datasets_to_download)}")
    
    print(f"📂 数据根目录: {data_root}")
    print(f"\n数据集列表:")
    for i, dataset in enumerate(datasets_to_download, 1):
        print(f"  {i:2d}. {dataset}")
    
    # 确认开始下载
    print("\n" + "="*80)
    if args.auto_yes:
        print("⚡ 非交互模式：自动开始下载")
    elif not sys.stdin.isatty():
        print("⚡ 后台模式：自动开始下载")
    else:
        response = input("是否开始下载？(y/n，默认y): ").strip().lower()
        if response == 'n':
            print("❌ 取消下载")
            return 1
    
    # 统计信息
    success_count = 0
    failed_datasets = []
    total_start_time = datetime.now()
    
    # 逐个下载数据集
    for i, dataset_name in enumerate(datasets_to_download, 1):
        print(f"\n\n{'#'*80}")
        print(f"进度: [{i}/{len(datasets_to_download)}]")
        print(f"{'#'*80}")
        
        if download_dataset(
            dataset_name, 
            data_root, 
            skip_existing=args.skip_existing,
            auto_yes=args.auto_yes
        ):
            success_count += 1
        else:
            failed_datasets.append(dataset_name)
    
    # 下载总结
    total_duration = (datetime.now() - total_start_time).total_seconds()
    
    print("\n\n" + "="*80)
    print("📊 下载总结")
    print("="*80)
    print(f"✅ 成功: {success_count}/{len(datasets_to_download)}")
    print(f"❌ 失败: {len(failed_datasets)}/{len(datasets_to_download)}")
    print(f"⏱️  总耗时: {total_duration/60:.2f}分钟 ({total_duration/3600:.2f}小时)")
    
    if failed_datasets:
        print(f"\n❌ 失败的数据集:")
        for dataset in failed_datasets:
            print(f"  - {dataset}")
        print(f"\n💡 可以重新运行此脚本，支持断点续传")
    
    # 计算总数据大小
    total_size = sum(f.stat().st_size for f in data_root.rglob('*') if f.is_file())
    print(f"\n📊 数据总大小: {total_size / (1024**3):.2f} GB")
    print(f"📁 数据位置: {data_root}")
    
    print("\n" + "="*80)
    print("✅ 所有下载任务完成!")
    print("="*80)
    
    return 0 if len(failed_datasets) == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断下载")
        print("💡 可以重新运行脚本继续下载（支持断点续传）")
        sys.exit(130)

