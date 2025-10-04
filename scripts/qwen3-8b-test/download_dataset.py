#!/usr/bin/env python3
"""
下载HumanEval+数据集脚本
"""

import os
import sys
from pathlib import Path

def download_dataset():
    """下载HumanEval+数据集"""
    print("=== 下载HumanEval+数据集 ===")
    
    # 使用项目data目录
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # 数据集URL - 使用HumanEval+数据集
    dataset_url = "https://github.com/evalplus/humanevalplus_release/releases/download/v0.1.10/HumanEvalPlus.jsonl.gz"
    dataset_gz_path = data_dir / "HumanEvalPlus.jsonl.gz"
    dataset_jsonl_path = data_dir / "HumanEvalPlus.jsonl"
    
    # 检查是否已存在解压后的文件
    if dataset_jsonl_path.exists():
        print(f"✓ 数据集已存在: {dataset_jsonl_path}")
        print(f"文件大小: {dataset_jsonl_path.stat().st_size / 1024 / 1024:.2f} MB")
        return True
    
    # 检查是否已存在压缩文件
    if dataset_gz_path.exists():
        print(f"✓ 压缩文件已存在: {dataset_gz_path}")
        print("正在解压...")
    else:
        print(f"正在下载数据集到: {dataset_gz_path}")
        print(f"URL: {dataset_url}")
        
        try:
            import wget
            wget.download(dataset_url, str(dataset_gz_path))
            print(f"\n✓ 数据集下载完成: {dataset_gz_path}")
        except ImportError:
            print("wget模块未安装，尝试使用urllib...")
            try:
                import urllib.request
                urllib.request.urlretrieve(dataset_url, str(dataset_gz_path))
                print(f"✓ 数据集下载完成: {dataset_gz_path}")
            except Exception as e:
                print(f"✗ 数据集下载失败: {e}")
                return False
        except Exception as e:
            print(f"✗ 数据集下载失败: {e}")
            return False
    
    # 解压文件
    print("正在解压数据集...")
    try:
        import gzip
        with gzip.open(dataset_gz_path, 'rt', encoding='utf-8') as f_in:
            with open(dataset_jsonl_path, 'w', encoding='utf-8') as f_out:
                f_out.write(f_in.read())
        
        print(f"✓ 数据集解压完成: {dataset_jsonl_path}")
        print(f"文件大小: {dataset_jsonl_path.stat().st_size / 1024 / 1024:.2f} MB")
        
        # 删除压缩文件以节省空间
        dataset_gz_path.unlink()
        print("✓ 已删除压缩文件")
        
        return True
    except Exception as e:
        print(f"✗ 数据集解压失败: {e}")
        return False

def main():
    """主函数"""
    if download_dataset():
        print("\n✓ 数据集准备完成!")
        print("现在可以运行评测脚本了:")
        print("python scripts/evaluate_model.py")
        return 0
    else:
        print("\n✗ 数据集下载失败!")
        print("请检查网络连接或手动下载数据集")
        return 1

if __name__ == "__main__":
    sys.exit(main())
