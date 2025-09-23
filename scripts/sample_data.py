#!/usr/bin/env python3
"""
数据随机抽取脚本 - 简化版本
从大规模数据集中随机抽取2万条训练样本
"""

import sys
import yaml
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.sampler import DataSampler


def load_config():
    """加载配置文件"""
    config_path = project_root / "config" / "base.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    """主函数 - 从配置文件读取参数"""
    
    # 加载配置
    config = load_config()
    
    # 从配置文件读取参数
    input_file = config['data']['input_file']
    output_dir = config['data']['output_dir']
    sample_size = config['data']['sample_size']
    seed = config['data']['seed']
    
    # 构建输出文件路径
    output_file = f"{output_dir}/sampled_data_{sample_size}.jsonl"
    
    print(f"开始数据随机抽取...")
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"采样数量: {sample_size}")
    
    # 创建输出目录
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # 初始化数据采样器
    sampler = DataSampler(seed=seed)
    
    # 执行数据采样
    result = sampler.sample_from_jsonl(
        input_file=input_file,
        output_file=output_file,
        sample_size=sample_size,
        shuffle=True
    )
    
    print(f"数据采样完成!")
    print(f"采样统计: {result}")
    
    # 验证数据
    validation = sampler.validate_sampled_data(
        data_file=output_file,
        expected_size=sample_size
    )
    
    if validation["validation_success"]:
        print(f"数据验证成功: {validation['valid_json_lines']} 条有效数据")
    else:
        print(f"数据验证失败: {validation}")
        sys.exit(1)


if __name__ == "__main__":
    main()