#!/usr/bin/env python3
"""
精确转换15个HuggingFace数据集为LLaMA-Factory格式
基于实际数据集格式进行精确转换
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
from tqdm import tqdm
import glob


def convert_apps(input_files: List[Path], output_file: Path, max_samples: Optional[int] = None):
    """转换 APPS 数据集 - JSONL格式，包含question和solutions"""
    print(f"  转换 APPS 数据集...")
    converted = []
    
    for file_path in input_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, desc=f"  处理 {file_path.name}")):
                if max_samples and len(converted) >= max_samples:
                    break
                try:
                    data = json.loads(line)
                    question = data.get("question", "")
                    solutions = json.loads(data.get("solutions", "[]"))
                    
                    if question and solutions and len(solutions) > 0:
                        converted.append({
                            "messages": [
                                {"role": "user", "content": f"Solve this programming problem:\n\n{question}"},
                                {"role": "assistant", "content": f"```python\n{solutions[0]}\n```"}
                            ]
                        })
                except Exception as e:
                    continue
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in converted:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    return len(converted)


def convert_tiny_codes(input_files: List[Path], output_file: Path, max_samples: Optional[int] = None):
    """转换 tiny-codes 数据集 - Parquet格式"""
    print(f"  转换 tiny-codes 数据集...")
    converted = []
    
    for file_path in input_files:
        if max_samples and len(converted) >= max_samples:
            break
        df = pd.read_parquet(file_path)
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  处理 {file_path.name}"):
            if max_samples and len(converted) >= max_samples:
                break
            try:
                prompt = row.get('prompt', row.get('instruction', ''))
                response = row.get('response', row.get('output', row.get('code', '')))
                if prompt and response:
                    converted.append({
                        "messages": [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": response}
                        ]
                    })
            except:
                continue
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in converted:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    return len(converted)


def convert_commitpackft(input_files: List[Path], output_file: Path, max_samples: Optional[int] = None):
    """转换 commitpackft 数据集 - JSONL格式在多个语言目录下"""
    print(f"  转换 commitpackft 数据集...")
    converted = []
    
    for file_path in input_files:
        if max_samples and len(converted) >= max_samples:
            break
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if max_samples and len(converted) >= max_samples:
                    break
                try:
                    data = json.loads(line)
                    commit_msg = data.get("subject", data.get("message", ""))
                    new_code = data.get("new_contents", "")
                    old_code = data.get("old_contents", "")
                    
                    if commit_msg and new_code:
                        if old_code:
                            prompt = f"Refactor the code based on: {commit_msg}\n\nOld code:\n```\n{old_code}\n```"
                        else:
                            prompt = f"Implement: {commit_msg}"
                        
                        converted.append({
                            "messages": [
                                {"role": "user", "content": prompt},
                                {"role": "assistant", "content": f"```\n{new_code}\n```"}
                            ]
                        })
                except:
                    continue
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in converted:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    return len(converted)


def convert_stackexchange_codereview(input_files: List[Path], output_file: Path, max_samples: Optional[int] = None):
    """转换 stackexchange_codereview - Parquet格式，有conversations字段（numpy.ndarray类型）"""
    print(f"  转换 stackexchange_codereview 数据集...")
    converted = []
    
    for file_path in input_files:
        if max_samples and len(converted) >= max_samples:
            break
        df = pd.read_parquet(file_path)
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  处理 {file_path.name}"):
            if max_samples and len(converted) >= max_samples:
                break
            try:
                # conversations字段是numpy.ndarray类型
                if 'conversations' in row:
                    conversations = row['conversations']
                    # numpy.ndarray转list
                    if hasattr(conversations, 'tolist'):
                        conversations = conversations.tolist()
                    
                    if conversations and len(conversations) > 0:
                        messages = []
                        for msg in conversations:
                            role = "user" if msg.get('from') == 'human' else "assistant"
                            messages.append({"role": role, "content": msg.get('value', '')})
                        if messages and len(messages) >= 2:
                            converted.append({"messages": messages})
                # 备用：使用instruction和completion
                elif 'instruction' in row and 'completion' in row and row['instruction'] and row['completion']:
                    converted.append({
                        "messages": [
                            {"role": "user", "content": str(row['instruction'])},
                            {"role": "assistant", "content": str(row['completion'])}
                        ]
                    })
            except Exception as e:
                continue
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in converted:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    return len(converted)


def convert_code_contests(input_files: List[Path], output_file: Path, max_samples: Optional[int] = None):
    """转换 code_contests - Parquet格式，solutions是dict类型"""
    print(f"  转换 code_contests 数据集...")
    converted = []
    
    for file_path in input_files:
        if max_samples and len(converted) >= max_samples:
            break
        df = pd.read_parquet(file_path)
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  处理 {file_path.name}"):
            if max_samples and len(converted) >= max_samples:
                break
            try:
                description = row.get('description', '')
                solutions = row.get('solutions')
                
                if description and solutions:
                    # solutions是dict格式，包含language和solution数组
                    if isinstance(solutions, dict):
                        solution_arr = solutions.get('solution', [])
                        # 转换为list（可能是numpy.ndarray）
                        if hasattr(solution_arr, 'tolist'):
                            solution_arr = solution_arr.tolist()
                        if solution_arr and len(solution_arr) > 0:
                            solution = solution_arr[0]
                            if solution:
                                converted.append({
                                    "messages": [
                                        {"role": "user", "content": f"Solve this competitive programming problem:\n\n{description}"},
                                        {"role": "assistant", "content": solution}
                                    ]
                                })
            except Exception as e:
                continue
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in converted:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    return len(converted)


def convert_reflection_seq_gpt(input_files: List[Path], output_file: Path, max_samples: Optional[int] = None):
    """转换 ReflectionSeq-GPT - JSONL格式，包含messages字段"""
    print(f"  转换 ReflectionSeq-GPT 数据集...")
    converted = []
    
    for file_path in input_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"  处理 {file_path.name}"):
                if max_samples and len(converted) >= max_samples:
                    break
                try:
                    data = json.loads(line)
                    if 'messages' in data:
                        messages_raw = data['messages']
                        if isinstance(messages_raw, str):
                            messages_raw = json.loads(messages_raw)
                        
                        messages = []
                        for msg in messages_raw:
                            role = msg.get('role', 'user')
                            content = msg.get('content', '')
                            if isinstance(content, list):
                                # 提取text类型的content
                                text_content = ' '.join([c.get('content', '') for c in content if c.get('type') == 'text'])
                                content = text_content
                            if content:
                                messages.append({"role": role, "content": content})
                        
                        if messages:
                            converted.append({"messages": messages})
                except:
                    continue
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in converted:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    return len(converted)


def convert_codeforces(input_files: List[Path], output_file: Path, max_samples: Optional[int] = None):
    """转换 Codeforces-Python-Submissions - Parquet格式"""
    print(f"  转换 Codeforces-Python-Submissions 数据集...")
    converted = []
    
    for file_path in input_files:
        if max_samples and len(converted) >= max_samples:
            break
        df = pd.read_parquet(file_path)
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  处理 {file_path.name}"):
            if max_samples and len(converted) >= max_samples:
                break
            try:
                # 优先使用prompt和response字段（已经格式化好的）
                if 'prompt' in row and 'response' in row and row['prompt'] and row['response']:
                    converted.append({
                        "messages": [
                            {"role": "user", "content": row['prompt']},
                            {"role": "assistant", "content": row['response']}
                        ]
                    })
                # 备用：使用problem-description和code
                elif 'code' in row and row['code']:
                    problem = row.get('problem-description', row.get('title', ''))
                    code = row['code']
                    if problem:
                        prompt = f"Solve this Codeforces problem:\n\n{problem}"
                    else:
                        prompt = "Write a Python solution for this Codeforces problem."
                    
                    converted.append({
                        "messages": [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": f"```python\n{code}\n```"}
                        ]
                    })
            except:
                continue
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in converted:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    return len(converted)


def convert_self_oss_instruct(input_files: List[Path], output_file: Path, max_samples: Optional[int] = None):
    """转换 self-oss-instruct-sc2-exec-filter-50k - Parquet格式，instruction/response字段"""
    print(f"  转换 self-oss-instruct 数据集...")
    converted = []
    
    for file_path in input_files:
        if max_samples and len(converted) >= max_samples:
            break
        df = pd.read_parquet(file_path)
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  处理 {file_path.name}"):
            if max_samples and len(converted) >= max_samples:
                break
            try:
                instruction = row.get('instruction', row.get('prompt', ''))
                response = row.get('response', row.get('output', ''))
                
                if instruction and response:
                    converted.append({
                        "messages": [
                            {"role": "user", "content": instruction},
                            {"role": "assistant", "content": response}
                        ]
                    })
            except:
                continue
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in converted:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    return len(converted)


def convert_swe_problems(input_files: List[Path], output_file: Path, max_samples: Optional[int] = None):
    """转换 real-world-swe-problems - Parquet格式"""
    print(f"  转换 real-world-swe-problems 数据集...")
    converted = []
    
    for file_path in input_files:
        if max_samples and len(converted) >= max_samples:
            break
        df = pd.read_parquet(file_path)
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  处理 {file_path.name}"):
            if max_samples and len(converted) >= max_samples:
                break
            try:
                prompt = row.get('prompt', '')
                solution = row.get('gold_standard_solution', '')
                
                if prompt and solution:
                    converted.append({
                        "messages": [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": solution}
                        ]
                    })
            except:
                continue
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in converted:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    return len(converted)


def convert_stack_exchange_paired(input_files: List[Path], output_file: Path, max_samples: Optional[int] = None):
    """转换 stack-exchange-paired - Parquet格式"""
    print(f"  转换 stack-exchange-paired 数据集...")
    converted = []
    
    for file_path in input_files:
        if max_samples and len(converted) >= max_samples:
            break
        df = pd.read_parquet(file_path)
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  处理 {file_path.name}"):
            if max_samples and len(converted) >= max_samples:
                break
            try:
                question = row.get('question', '')
                response_j = row.get('response_j', '')
                response_k = row.get('response_k', '')
                
                response = response_j if response_j else response_k
                if question and response:
                    converted.append({
                        "messages": [
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": response}
                        ]
                    })
            except:
                continue
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in converted:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    return len(converted)


def convert_react_code_instructions(input_files: List[Path], output_file: Path, max_samples: Optional[int] = None):
    """转换 react-code-instructions - JSONL格式，包含messages字段"""
    print(f"  转换 react-code-instructions 数据集...")
    converted = []
    
    for file_path in input_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"  处理 {file_path.name}"):
                if max_samples and len(converted) >= max_samples:
                    break
                try:
                    data = json.loads(line)
                    if 'messages' in data:
                        messages = data['messages']
                        standardized = []
                        for msg in messages:
                            role = msg.get('role', 'user')
                            content = msg.get('content', '')
                            if role and content:
                                standardized.append({"role": role, "content": content})
                        if standardized:
                            converted.append({"messages": standardized})
                except:
                    continue
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in converted:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    return len(converted)


def convert_stackexchange_qa(input_files: List[Path], output_file: Path, max_samples: Optional[int] = None):
    """转换 stackexchange-question-answering - Parquet格式"""
    print(f"  转换 stackexchange-question-answering 数据集...")
    converted = []
    
    for file_path in input_files:
        if max_samples and len(converted) >= max_samples:
            break
        df = pd.read_parquet(file_path)
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  处理 {file_path.name}"):
            if max_samples and len(converted) >= max_samples:
                break
            try:
                prompt = row.get('prompt', '')
                answer = row.get('gold_standard_solution', '')
                
                if prompt and answer:
                    converted.append({
                        "messages": [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": answer}
                        ]
                    })
            except:
                continue
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in converted:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    return len(converted)


def convert_synthetic_2_sft(input_files: List[Path], output_file: Path, max_samples: Optional[int] = None):
    """转换 SYNTHETIC-2-SFT-verified - Parquet格式，包含messages字段（numpy.ndarray类型）"""
    print(f"  转换 SYNTHETIC-2-SFT-verified 数据集...")
    converted = []
    
    for file_path in input_files:
        if max_samples and len(converted) >= max_samples:
            break
        df = pd.read_parquet(file_path)
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  处理 {file_path.name}"):
            if max_samples and len(converted) >= max_samples:
                break
            try:
                if 'messages' in row:
                    messages = row['messages']
                    # numpy.ndarray转list
                    if hasattr(messages, 'tolist'):
                        messages = messages.tolist()
                    
                    if messages and len(messages) > 0:
                        standardized = []
                        for msg in messages:
                            role = msg.get('role', 'user')
                            content = msg.get('content', '')
                            if role and content:
                                standardized.append({"role": role, "content": content})
                        if standardized and len(standardized) >= 2:
                            converted.append({"messages": standardized})
            except Exception as e:
                continue
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in converted:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    return len(converted)


def convert_sql_context(input_files: List[Path], output_file: Path, max_samples: Optional[int] = None):
    """转换 sql-create-context-instruction - Parquet格式，text字段使用[INST]...[/INST]格式"""
    print(f"  转换 sql-create-context-instruction 数据集...")
    converted = []
    
    for file_path in input_files:
        if max_samples and len(converted) >= max_samples:
            break
        df = pd.read_parquet(file_path)
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  处理 {file_path.name}"):
            if max_samples and len(converted) >= max_samples:
                break
            try:
                text = row.get('text', '')
                # text格式: [INST] instruction [/INST] response
                if text and '[INST]' in text and '[/INST]' in text:
                    # 解析 [INST]...[/INST] 格式
                    inst_start = text.find('[INST]')
                    inst_end = text.find('[/INST]')
                    if inst_start != -1 and inst_end != -1 and inst_end > inst_start:
                        instruction = text[inst_start + 6:inst_end].strip()
                        response = text[inst_end + 7:].strip()
                        
                        if instruction and response:
                            converted.append({
                                "messages": [
                                    {"role": "user", "content": instruction},
                                    {"role": "assistant", "content": response}
                                ]
                            })
            except Exception as e:
                continue
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in converted:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    return len(converted)


def convert_magpie_qwen(input_files: List[Path], output_file: Path, max_samples: Optional[int] = None):
    """转换 Magpie-Qwen2.5-Coder-Pro-300K - Parquet格式，conversations字段（numpy.ndarray类型）"""
    print(f"  转换 Magpie-Qwen2.5-Coder-Pro-300K 数据集...")
    converted = []
    
    for file_path in input_files:
        if max_samples and len(converted) >= max_samples:
            break
        df = pd.read_parquet(file_path)
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  处理 {file_path.name}"):
            if max_samples and len(converted) >= max_samples:
                break
            try:
                # conversations字段是numpy.ndarray类型
                if 'conversations' in row:
                    conversations = row['conversations']
                    # numpy.ndarray转list
                    if hasattr(conversations, 'tolist'):
                        conversations = conversations.tolist()
                    
                    if conversations and len(conversations) > 0:
                        messages = []
                        for msg in conversations:
                            role_map = {'human': 'user', 'gpt': 'assistant', 'user': 'user', 'assistant': 'assistant'}
                            role = role_map.get(msg.get('from', 'user'), 'user')
                            content = msg.get('value', '')
                            if content:
                                messages.append({"role": role, "content": content})
                        if messages and len(messages) >= 2:
                            converted.append({"messages": messages})
                # 备用：使用instruction和response
                elif 'instruction' in row and 'response' in row and row['instruction'] and row['response']:
                    converted.append({
                        "messages": [
                            {"role": "user", "content": str(row['instruction'])},
                            {"role": "assistant", "content": str(row['response'])}
                        ]
                    })
            except Exception as e:
                continue
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in converted:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    return len(converted)


# 数据集配置
DATASETS_CONFIG = {
    "apps": {
        "converter": convert_apps,
        "pattern": "*.jsonl"
    },
    "tiny-codes": {
        "converter": convert_tiny_codes,
        "pattern": "*.parquet"
    },
    "commitpackft": {
        "converter": convert_commitpackft,
        "pattern": "data/**/data.jsonl"
    },
    "stackexchange_codereview": {
        "converter": convert_stackexchange_codereview,
        "pattern": "data/*.parquet"
    },
    "code_contests": {
        "converter": convert_code_contests,
        "pattern": "data/*.parquet"
    },
    "ReflectionSeq-GPT": {
        "converter": convert_reflection_seq_gpt,
        "pattern": "*.jsonl"
    },
    "Codeforces-Python-Submissions": {
        "converter": convert_codeforces,
        "pattern": "data/*.parquet"
    },
    "self-oss-instruct-sc2-exec-filter-50k": {
        "converter": convert_self_oss_instruct,
        "pattern": "data/*.parquet"
    },
    "real-world-swe-problems": {
        "converter": convert_swe_problems,
        "pattern": "data/*.parquet"
    },
    "stack-exchange-paired": {
        "converter": convert_stack_exchange_paired,
        "pattern": "data/**/*.parquet"
    },
    "react-code-instructions": {
        "converter": convert_react_code_instructions,
        "pattern": "data/*.jsonl"
    },
    "stackexchange-question-answering": {
        "converter": convert_stackexchange_qa,
        "pattern": "data/*.parquet"
    },
    "SYNTHETIC-2-SFT-verified": {
        "converter": convert_synthetic_2_sft,
        "pattern": "data/*.parquet"
    },
    "sql-create-context-instruction": {
        "converter": convert_sql_context,
        "pattern": "data/*.parquet"
    },
    "Magpie-Qwen2.5-Coder-Pro-300K-v0.1": {
        "converter": convert_magpie_qwen,
        "pattern": "data/*.parquet"
    }
}


def main():
    parser = argparse.ArgumentParser(description="转换15个代码数据集为LLaMA-Factory格式")
    
    # 使用相对路径，从 scripts/survey-sft/ 到项目根目录
    project_root = Path(__file__).parent.parent.parent
    default_data_dir = project_root / "data"
    default_output_dir = project_root / "data" / "llamafactory"
    
    parser.add_argument("--data-dir", type=str, default=str(default_data_dir), help="数据集根目录")
    parser.add_argument("--output-dir", type=str, default=str(default_output_dir), help="输出目录")
    parser.add_argument("--max-samples", type=int, default=None, help="每个数据集最多转换的样本数")
    parser.add_argument("--datasets", nargs="+", help="指定要转换的数据集")
    
    args = parser.parse_args()
    
    data_root = Path(args.data_dir)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("🔄 转换15个代码数据集为 LLaMA-Factory 格式")
    print("="*80)
    print(f"📂 数据根目录: {data_root}")
    print(f"📂 输出目录: {output_root}")
    if args.max_samples:
        print(f"📊 每个数据集最多: {args.max_samples} 条")
    print()
    
    # 确定要转换的数据集
    if args.datasets:
        datasets = {k: v for k, v in DATASETS_CONFIG.items() if k in args.datasets}
    else:
        datasets = DATASETS_CONFIG
    
    results = {}
    dataset_info = {}
    
    for dataset_name, config in datasets.items():
        print(f"\n{'='*80}")
        print(f"📦 处理数据集: {dataset_name}")
        print(f"{'='*80}")
        
        dataset_dir = data_root / dataset_name
        if not dataset_dir.exists():
            print(f"  ⚠️  目录不存在: {dataset_dir}")
            continue
        
        # 查找输入文件
        pattern = config["pattern"]
        input_files = list(dataset_dir.glob(pattern))
        
        if not input_files:
            print(f"  ⚠️  未找到匹配文件: {pattern}")
            continue
        
        print(f"  📁 找到 {len(input_files)} 个文件")
        
        # 转换数据集
        output_file = output_root / f"{dataset_name}.jsonl"
        try:
            count = config["converter"](input_files, output_file, args.max_samples)
            results[dataset_name] = count
            
            if count > 0:
                print(f"  ✅ 转换成功: {count} 条数据")
                print(f"  📄 输出文件: {output_file}")
                print(f"  📊 文件大小: {output_file.stat().st_size / (1024**2):.2f} MB")
                
                # 添加到dataset_info
                dataset_key = dataset_name.replace('-', '_').replace('.', '_').lower()
                dataset_info[dataset_key] = {
                    "file_name": f"llamafactory/{dataset_name}.jsonl",
                    "formatting": "sharegpt",
                    "columns": {
                        "messages": "messages"
                    },
                    "tags": {
                        "role_tag": "role",
                        "content_tag": "content",
                        "user_tag": "user",
                        "assistant_tag": "assistant"
                    }
                }
            else:
                print(f"  ⚠️  转换失败: 0 条数据")
        except Exception as e:
            print(f"  ❌ 转换失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # 更新dataset_info.json
    dataset_info_path = data_root / "dataset_info.json"
    
    # 读取现有配置
    if dataset_info_path.exists():
        with open(dataset_info_path, 'r', encoding='utf-8') as f:
            existing_info = json.load(f)
    else:
        existing_info = {}
    
    # 合并配置
    existing_info.update(dataset_info)
    
    # 保存配置
    with open(dataset_info_path, 'w', encoding='utf-8') as f:
        json.dump(existing_info, f, ensure_ascii=False, indent=2)
    
    print(f"\n\n{'='*80}")
    print("📊 转换总结")
    print(f"{'='*80}")
    
    total_samples = sum(results.values())
    print(f"✅ 成功转换 {len(results)} 个数据集，共 {total_samples:,} 条数据")
    print(f"\n数据集明细:")
    for dataset_name, count in results.items():
        print(f"  - {dataset_name}: {count:,} 条")
    
    print(f"\n📄 dataset_info.json 已更新: {dataset_info_path}")
    print(f"📁 所有数据已保存到: {output_root}")
    print(f"\n💡 在 LLaMA-Factory 中使用这些数据集:")
    print(f"   dataset: " + ",".join(list(dataset_info.keys())[:3]) + ",...")
    print("="*80)


if __name__ == "__main__":
    import sys
    sys.exit(main())

