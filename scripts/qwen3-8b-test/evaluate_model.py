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
    project_root = Path(__file__).parent.parent.parent
    config_path = project_root / "config" / "eval.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def check_dataset():
    """检查HumanEval+数据集是否存在"""
    print("=== 检查HumanEval+数据集 ===")
    
    # 使用项目data目录
    project_root = Path(__file__).parent.parent.parent
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
    
    # 根据模型类型确定结果存储路径
    model_name = eval_config.get('model_name', 'unknown')
    eval_mode = eval_config.get('eval_mode', 'auto')  # base, chat, auto
    
    # 自动检测模式：根据模型路径判断是base还是chat模型
    if eval_mode == 'auto':
        model_path = eval_config['model_path']
        if 'saves' in model_path or 'sft' in model_path or 'chat' in model_path:
            eval_mode = 'chat'
        else:
            eval_mode = 'base'
    
    # 为不同模式设置不同的结果目录，避免结果混淆
    base_results_root = eval_config.get('evalplus_root', 'evalplus_results')
    results_root = f"{base_results_root}_{eval_mode}"
    
    print(f"📊 评测模式: {eval_mode} ({'微调模型' if eval_mode == 'chat' else '基础模型'})")
    print(f"📁 结果存储路径: {results_root}")
    backend = eval_config['backend']
    # 可选限制可见GPU（单卡/多卡）
    cuda_visible = eval_config.get('cuda_visible_devices')
    # 对于 hf 后端，默认强制单卡以避免 accelerate 多卡分片导致设备不一致
    # 对于 vllm 后端，支持多卡张量并行，性能更好
    if cuda_visible is None and backend == 'hf':
        cuda_visible = '0'
    if cuda_visible is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_visible)
        print(f"设置 CUDA_VISIBLE_DEVICES={cuda_visible} (后端: {backend})")
    
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
    # vLLM 后端支持张量并行 --tp，大幅提升推理速度
    if backend == 'vllm':
        tp = eval_config.get('tp') or eval_config.get('tensor_parallel_size')
        if tp:
            cmd.extend(["--tp", str(tp)])
            print(f"使用 vLLM 后端，张量并行度: {tp}")
        else:
            print("使用 vLLM 后端，单GPU模式")
    elif backend == 'hf':
        print("使用 HuggingFace 后端（较慢，建议使用 vLLM）")
    
    print(f"运行一体化评测命令: {' '.join(cmd)}")
    if backend == 'vllm' and eval_config.get('tp', 1) > 1:
        print("🚀 使用多GPU vLLM后端，评测速度将大幅提升！")
    elif backend == 'vllm':
        print("⚡ 使用单GPU vLLM后端，比HF后端更快")
    else:
        print("🐌 使用HF后端较慢，建议切换到vLLM后端")
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


def generate_report(results_file, output_dir, eval_mode="unknown"):
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
        
        # 为不同模式生成不同的报告文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_filename = f"evaluation_report_{eval_mode}_{timestamp}.md"
        report_path = Path(output_dir) / report_filename
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# HumanEval评测报告 - {eval_mode.upper()}模型\n\n")
            f.write(f"**生成时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**评测模式:** {eval_mode} ({'微调模型' if eval_mode == 'chat' else '基础模型'})\n")
            f.write(f"**结果文件:** {results_file}\n\n")
            
            # 计算和显示指标
            eval_results = results
            # 检查是否是包装格式（有date, hash, eval字段）
            if isinstance(results, dict) and 'eval' in results:
                eval_results = results['eval']
            
            if isinstance(eval_results, dict) and any(task_id.startswith('HumanEval/') for task_id in eval_results.keys()):
                # 处理EvalPlus格式的结果
                total_tasks = len(eval_results)
                base_pass = 0
                plus_pass = 0
                
                for task_id, task_results in eval_results.items():
                    if task_results and len(task_results) > 0:
                        result = task_results[0]  # 取第一个结果
                        if result.get('base_status') == 'pass':
                            base_pass += 1
                        if result.get('plus_status') == 'pass':
                            plus_pass += 1
                
                f.write("## 📊 评测指标说明\n\n")
                f.write("- **Pass@1 (base)**: 基础HumanEval测试集的一次通过率\n")
                f.write("- **Pass@1 (plus)**: HumanEval+增强测试集的一次通过率\n")
                f.write("- **基础测试**: 验证代码的基本功能正确性\n")
                f.write("- **增强测试**: 包含更多边界情况和测试用例，更严格\n\n")
                
                f.write("## 🎯 评测结果\n\n")
                f.write(f"**总题数:** {total_tasks}\n")
                f.write(f"**Pass@1 (base):** {base_pass}/{total_tasks} = {base_pass/total_tasks:.1%}\n")
                f.write(f"**Pass@1 (plus):** {plus_pass}/{total_tasks} = {plus_pass/total_tasks:.1%}\n\n")
                
                # 分类统计
                both_pass = sum(1 for task_results in eval_results.values() 
                              if task_results and len(task_results) > 0 and 
                              task_results[0].get('base_status') == 'pass' and 
                              task_results[0].get('plus_status') == 'pass')
                base_only = base_pass - both_pass
                plus_only = plus_pass - both_pass
                both_fail = total_tasks - base_pass - plus_only
                
                f.write("### 📈 通过情况分析\n\n")
                f.write(f"- ✅ **两种测试都通过:** {both_pass} 题 ({both_pass/total_tasks:.1%})\n")
                f.write(f"- 🟡 **仅基础测试通过:** {base_only} 题 ({base_only/total_tasks:.1%})\n")
                f.write(f"- 🟠 **仅增强测试通过:** {plus_only} 题 ({plus_only/total_tasks:.1%})\n")
                f.write(f"- ❌ **两种测试都失败:** {both_fail} 题 ({both_fail/total_tasks:.1%})\n\n")
                
                # 失败案例分析
                failed_tasks = []
                for task_id, task_results in eval_results.items():
                    if task_results and len(task_results) > 0:
                        result = task_results[0]
                        if result.get('base_status') != 'pass' or result.get('plus_status') != 'pass':
                            failed_tasks.append((task_id, result))
                
                if failed_tasks:
                    f.write("### ❌ 失败题目分析\n\n")
                    f.write("| 题目ID | 基础测试 | 增强测试 | 失败原因 |\n")
                    f.write("|--------|----------|----------|----------|\n")
                    for task_id, result in failed_tasks[:10]:  # 只显示前10个失败案例
                        base_status = "✅" if result.get('base_status') == 'pass' else "❌"
                        plus_status = "✅" if result.get('plus_status') == 'pass' else "❌"
                        base_fails = result.get('base_fail_tests', [])
                        plus_fails = result.get('plus_fail_tests', [])
                        reason = ""
                        if base_fails:
                            reason = f"基础测试失败: {len(base_fails)}个用例"
                        if plus_fails:
                            if reason:
                                reason += f"; 增强测试失败: {len(plus_fails)}个用例"
                            else:
                                reason = f"增强测试失败: {len(plus_fails)}个用例"
                        f.write(f"| {task_id} | {base_status} | {plus_status} | {reason} |\n")
                    
                    if len(failed_tasks) > 10:
                        f.write(f"\n*注：还有{len(failed_tasks)-10}个失败题目未在表格中显示*\n")
                    f.write("\n")
            
            elif isinstance(results, list):
                # 处理列表格式的结果
                total = len(results)
                passed = sum(1 for r in results if r.get('passed', False))
                f.write(f"**总题数:** {total}\n")
                f.write(f"**通过数:** {passed}\n")
                f.write(f"**Pass@1:** {passed/total:.1%}\n\n")
            
            # 添加模型对比建议
            f.write("## 💡 性能建议\n\n")
            if eval_mode == 'base':
                f.write("- 这是基础模型的评测结果，作为对比基准\n")
                f.write("- 建议与微调后的模型结果进行对比，查看训练效果\n")
            elif eval_mode == 'chat':
                f.write("- 这是微调模型的评测结果\n")
                f.write("- 可以与基础模型对比，评估微调的提升效果\n")
                f.write("- 如果结果不理想，建议检查训练数据质量和训练参数\n")
            f.write("\n")
            
            # 简化详细结果显示（只显示摘要）
            f.write("## 📋 结果文件信息\n\n")
            f.write(f"完整结果数据已保存在: `{results_file}`\n")
            f.write(f"结果格式: {'JSONL' if results_file.suffix == '.jsonl' else 'JSON'}\n")
            f.write(f"数据大小: {len(str(results))} 字符\n")
        
        print(f"📄 报告已生成: {report_path}")
        
    except Exception as e:
        print(f"生成报告失败: {e}")


def setup_offline_env():
    """设置离线环境变量"""
    project_root = Path(__file__).parent.parent.parent
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
    
    # 确定评测模式和结果存储路径
    eval_config = config['evaluation']
    model_name = eval_config.get('model_name', 'unknown')
    eval_mode = eval_config.get('eval_mode', 'auto')  # base, chat, auto
    
    # 自动检测模式：根据模型路径判断是base还是chat模型
    if eval_mode == 'auto':
        model_path = eval_config['model_path']
        if 'saves' in model_path or 'sft' in model_path or 'chat' in model_path:
            eval_mode = 'chat'
        else:
            eval_mode = 'base'
    
    # 为不同模式设置不同的结果目录，避免结果混淆
    base_results_root = eval_config.get('evalplus_root', 'evalplus_results')
    results_root = f"{base_results_root}_{eval_mode}"
    
    print(f"📊 当前评测模式: {eval_mode} ({'微调模型' if eval_mode == 'chat' else '基础模型'})")
    print(f"📁 结果存储路径: {results_root}")
    
    # 检查是否已有生成的结果文件
    results_file = find_results(results_root)
    if results_file:
        print(f"✓ 找到已有的{eval_mode}模型结果文件: {results_file}")
        print("使用已有结果文件生成报告...")
        generate_report(results_file, output_dir, eval_mode)
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
        print(f"✓ 找到{eval_mode}模型结果文件: {results_file}")
        generate_report(results_file, output_dir, eval_mode)
    else:
        print(f"⚠ 未找到{eval_mode}模型结果文件")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())