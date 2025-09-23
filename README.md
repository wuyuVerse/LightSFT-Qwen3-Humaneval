# LightSFT-Qwen3-Humaneval

一个用于Qwen3模型微调和HumanEval评测的完整项目，支持多种微调方式和自动化评测流程。

## 项目结构

```
LightSFT-Qwen3-Humaneval/
├── README.md                    # 项目说明文档
├── pyproject.toml              # 项目配置和依赖
├── main.py                     # 主入口文件
├── config/                     # 配置文件目录
│   ├── __init__.py
│   ├── base.yaml              # 基础配置
│   ├── qwen_sft.yaml          # Qwen SFT配置
│   ├── llamafactory.yaml      # LLaMA Factory配置
│   └── eval.yaml              # 评测配置
├── src/                        # 源代码目录
│   ├── __init__.py
│   ├── data/                   # 数据处理模块
│   │   ├── __init__.py
│   │   ├── sampler.py         # 数据采样器
│   │   └── preprocessor.py    # 数据预处理器
│   ├── training/               # 训练模块
│   │   ├── __init__.py
│   │   ├── qwen_sft.py        # Qwen SFT训练
│   │   ├── llamafactory.py    # LLaMA Factory训练
│   │   └── trainer_base.py    # 训练基类
│   ├── evaluation/             # 评测模块
│   │   ├── __init__.py
│   │   ├── humaneval.py       # HumanEval评测
│   │   └── evaluator.py       # 评测器基类
│   └── utils/                  # 工具模块
│       ├── __init__.py
│       ├── logger.py          # 日志工具
│       ├── config.py          # 配置管理
│       └── metrics.py         # 指标计算
├── scripts/                    # 脚本目录
│   ├── setup_env.sh           # 环境设置脚本
│   ├── run_qwen_sft.sh        # Qwen SFT运行脚本
│   ├── run_llamafactory.sh    # LLaMA Factory运行脚本
│   └── run_evaluation.sh      # 评测运行脚本
├── tests/                      # 测试目录
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_training.py
│   └── test_evaluation.py
├── logs/                       # 日志目录
├── outputs/                    # 输出目录
│   ├── models/                # 训练好的模型
│   ├── results/               # 评测结果
│   └── data/                  # 处理后的数据
└── requirements.txt            # 依赖列表（备用）
```

## 功能特性

- **数据采样**: 从大规模数据集中随机抽取指定数量的训练样本
- **多方式微调**: 支持Qwen SFT和LLaMA Factory两种微调方式
- **自动化评测**: 集成EvalPlus进行HumanEval评测
- **配置管理**: 使用Hydra进行灵活的配置管理
- **日志记录**: 完整的训练和评测日志记录
- **结果分析**: 自动化的结果分析和报告生成

## 快速开始

### 1. 环境设置

```bash
# 激活环境
source /volume/pt-train/users/wzhang/wjj-workspace/.zshrc

# 安装依赖
uv sync

# 或者使用pip
pip install -e .
```

### 2. 数据采样

```bash
python main.py data.sample \
    --input_file /volume/pt-train/users/wzhang/coder/coder-data/dataset/opc-sft-stage1/opc-sft-stage1_merged_chatml.jsonl \
    --output_file outputs/data/sampled_data.jsonl \
    --sample_size 20000
```

### 3. 模型微调

#### 使用Qwen SFT方式
```bash
python main.py training.qwen_sft \
    --data_file outputs/data/sampled_data.jsonl \
    --model_path /volume/pt-train/models/Qwen3-8B \
    --output_dir outputs/models/qwen_sft_model
```

#### 使用LLaMA Factory方式
```bash
python main.py training.llamafactory \
    --data_file outputs/data/sampled_data.jsonl \
    --model_path /volume/pt-train/models/Qwen3-8B \
    --output_dir outputs/models/llamafactory_model
```

### 4. HumanEval评测

```bash
python main.py evaluation.humaneval \
    --model_path outputs/models/qwen_sft_model \
    --output_file outputs/results/humaneval_results.json
```

### 5. 完整流水线

```bash
python main.py pipeline.full \
    --input_file /volume/pt-train/users/wzhang/coder/coder-data/dataset/opc-sft-stage1/opc-sft-stage1_merged_chatml.jsonl \
    --sample_size 20000 \
    --training_method qwen_sft \
    --model_path /volume/pt-train/models/Qwen3-8B
```

## 配置说明

项目使用Hydra进行配置管理，主要配置文件位于`config/`目录：

- `base.yaml`: 基础配置，包含通用参数
- `qwen_sft.yaml`: Qwen SFT训练配置
- `llamafactory.yaml`: LLaMA Factory训练配置
- `eval.yaml`: 评测配置

## 输出说明

- `outputs/models/`: 训练好的模型文件
- `outputs/results/`: 评测结果和报告
- `outputs/data/`: 处理后的数据文件
- `logs/`: 训练和评测日志

## 依赖说明

项目主要依赖：
- PyTorch: 深度学习框架
- Transformers: Hugging Face模型库
- Accelerate: 分布式训练加速
- PEFT: 参数高效微调
- EvalPlus: 代码评测工具
- Hydra: 配置管理

## 开发指南

### 代码规范
- 使用Black进行代码格式化
- 使用isort进行导入排序
- 使用flake8进行代码检查
- 使用mypy进行类型检查

### 测试
```bash
pytest tests/
```

### 贡献
1. Fork项目
2. 创建特性分支
3. 提交更改
4. 创建Pull Request

## 许可证

MIT License
