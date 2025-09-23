#!/usr/bin/env python3
"""
模型评测脚本 - 使用EvalPlus进行HumanEval评测
支持离线模式，使用本地数据集
"""

import subprocess
import sys
import yaml
import json
import os
from pathlib import Path
from datetime import datetime


def load_config():
    """加载配置文件"""
    project_root = Path(__file__).parent.parent
    config_path = project_root / "config" / "eval.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def check_dataset():
    """检查HumanEval+数据集是否存在"""
    print("=== 检查HumanEval+数据集 ===")
    
    # 使用项目data目录
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    dataset_path = data_dir / "HumanEvalPlus.jsonl"
    
    if dataset_path.exists():
        print(f"✓ 数据集已存在: {dataset_path}")
        print(f"文件大小: {dataset_path.stat().st_size / 1024 / 1024:.2f} MB")
        return True
    else:
        print(f"✗ 数据集不存在: {dataset_path}")
        print("请先在有网络的环境下运行: python scripts/download_dataset.py")
        return False


def run_evaluation(config):
    """运行评测 - 使用EvalPlus v0.3.1的一体化命令 (HumanEval+)"""
    eval_config = config['evaluation']
    # 结果根目录（可配置），默认 evalplus_results
    results_root = (
        eval_config.get('evalplus_root')
        or eval_config.get('evalplus_results_dir')
        or 'evalplus_results'
    )
    backend = eval_config['backend']
    # 可选限制可见GPU（单卡/多卡）
    cuda_visible = eval_config.get('cuda_visible_devices')
    # 对于 hf 后端，默认强制单卡以避免 accelerate 多卡分片导致设备不一致
    if cuda_visible is None and backend == 'hf':
        cuda_visible = '0'
    if cuda_visible is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_visible)
    
    # 构建一体化评测命令 - 直接使用evalplus.evaluate进行代码生成+评测
    # 移除 --base_only 以使用 HumanEval+ 全量测试
    cmd = [
        "evalplus.evaluate",
        "--model", eval_config['model_path'],
        "--dataset", eval_config['benchmark'],
        "--backend", backend,
        "--greedy",
        "--root", str(results_root),
        "--force_base_prompt"
    ]
    # vLLM 后端支持张量并行 --tp
    if backend == 'vllm':
        tp = eval_config.get('tp') or eval_config.get('tensor_parallel_size')
        if tp:
            cmd.extend(["--tp", str(tp)])
    
    print(f"运行一体化评测命令: {' '.join(cmd)}")
    print("注意: 这包括代码生成和评测，可能需要较长时间，请耐心等待...")
    
    try:
        # 不捕获输出，让用户看到实时进度
        result = subprocess.run(cmd, check=True)
        print("评测完成!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"评测失败: {e}")
        return False


def find_results(results_root: str):
    """查找评测结果文件"""
    results_dir = Path(results_root)
    if not results_dir.exists():
        print(f"未找到结果目录: {results_dir}")
        return None
    
    print(f"查找结果文件在: {results_dir}")
    
    # 查找结果文件
    result_files = list(results_dir.glob("**/*.json")) + list(results_dir.glob("**/*.jsonl"))
    if result_files:
        print(f"找到 {len(result_files)} 个结果文件:")
        for f in result_files:
            print(f"  - {f}")
        # 返回最新的文件
        return max(result_files, key=lambda x: x.stat().st_mtime)
    return None


def generate_report(results_file, output_dir):
    """生成评测报告"""
    if not results_file or not results_file.exists():
        print("未找到结果文件，跳过报告生成")
        return
    
    try:
        print(f"读取结果文件: {results_file}")
        
        # 读取结果
        with open(results_file, 'r', encoding='utf-8') as f:
            if results_file.suffix == '.jsonl':
                results = [json.loads(line) for line in f]
            else:
                results = json.load(f)
        
        # 生成报告
        report_path = Path(output_dir) / "evaluation_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# HumanEval评测报告\n\n")
            f.write(f"**生成时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**结果文件:** {results_file}\n\n")
            
            # 添加结果摘要
            if isinstance(results, list):
                total = len(results)
                passed = sum(1 for r in results if r.get('passed', False))
                f.write(f"**总题数:** {total}\n")
                f.write(f"**通过数:** {passed}\n")
                f.write(f"**通过率:** {passed/total:.2%}\n\n")
            elif isinstance(results, dict):
                # 处理字典格式的结果
                if 'results' in results:
                    result_list = results['results']
                    total = len(result_list)
                    passed = sum(1 for r in result_list if r.get('passed', False))
                    f.write(f"**总题数:** {total}\n")
                    f.write(f"**通过数:** {passed}\n")
                    f.write(f"**通过率:** {passed/total:.2%}\n\n")
                
                # 添加其他指标
                for key, value in results.items():
                    if key != 'results':
                        f.write(f"**{key}:** {value}\n")
                f.write("\n")
            
            # 添加详细结果
            f.write("## 详细结果\n\n")
            f.write("```json\n")
            f.write(json.dumps(results, indent=2, ensure_ascii=False))
            f.write("\n```\n")
        
        print(f"报告已生成: {report_path}")
        
    except Exception as e:
        print(f"生成报告失败: {e}")


def setup_offline_env():
    """设置离线环境变量"""
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    
    # 设置环境变量，让EvalPlus使用本地数据集
    os.environ['EVALPLUS_CACHE_DIR'] = str(data_dir)
    os.environ['HF_HOME'] = str(data_dir / "hf_cache")
    
    # 设置HumanEval+覆盖路径，使用我们下载的HumanEval+文件
    humaneval_plus_path = data_dir / "HumanEvalPlus.jsonl"
    if humaneval_plus_path.exists():
        os.environ['HUMANEVAL_OVERRIDE_PATH'] = str(humaneval_plus_path)
        print(f"✓ 设置离线环境")
        print(f"  EVALPLUS_CACHE_DIR: {data_dir}")
        print(f"  HF_HOME: {data_dir / 'hf_cache'}")
        print(f"  HUMANEVAL_OVERRIDE_PATH: {humaneval_plus_path}")
    else:
        print("⚠ 警告: HumanEvalPlus.jsonl文件不存在，将使用默认下载方式")
        print(f"✓ 设置离线环境")
        print(f"  EVALPLUS_CACHE_DIR: {data_dir}")
        print(f"  HF_HOME: {data_dir / 'hf_cache'}")
    
    # 创建必要的目录
    (data_dir / "hf_cache").mkdir(parents=True, exist_ok=True)


def main():
    """主函数"""
    print("=== HumanEval模型评测 (离线模式) ===")
    print("使用EvalPlus v0.3.1进行评测")
    
    # 设置离线环境
    setup_offline_env()
    
    # 加载配置
    try:
        config = load_config()
        print("✓ 配置加载成功")
    except Exception as e:
        print(f"✗ 配置加载失败: {e}")
        return 1
    
    # 检查模型路径
    model_path = config['evaluation']['model_path']
    if not Path(model_path).exists():
        print(f"✗ 模型路径不存在: {model_path}")
        print("请先训练模型或检查路径配置")
        return 1
    else:
        print(f"✓ 模型路径存在: {model_path}")
    
    # 创建输出目录
    output_dir = config['evaluation']['output_dir']
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"✓ 输出目录: {output_dir}")
    
    # 结果根目录（可配置），默认 evalplus_results
    results_root = (
        config['evaluation'].get('evalplus_root')
        or config['evaluation'].get('evalplus_results_dir')
        or 'evalplus_results'
    )
    
    # 检查是否已有生成的结果文件
    results_file = find_results(results_root)
    if results_file:
        print(f"✓ 找到已有的结果文件: {results_file}")
        print("使用已有结果文件生成报告...")
        generate_report(results_file, output_dir)
        return 0
    
    # 检查数据集 (HumanEval+)
    print("\n=== 检查数据集 ===")
    if not check_dataset():
        print("数据集不存在，无法继续评测")
        return 1
    
    # 运行一体化评测（代码生成+评测）
    print("\n=== 开始一体化评测 ===")
    if not run_evaluation(config):
        print("✗ 评测失败!")
        return 1
    
    print("✓ 评测成功完成!")
    
    # 查找并处理结果
    results_file = find_results(results_root)
    if results_file:
        print(f"✓ 找到结果文件: {results_file}")
        generate_report(results_file, output_dir)
    else:
        print("⚠ 未找到结果文件")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())