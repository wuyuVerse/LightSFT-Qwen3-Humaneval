# 脚本使用说明

## 数据采样脚本

### sample_data.py
从大规模数据集中随机抽取2万条训练样本。

```bash
python scripts/sample_data.py
```

**功能:**
- 从配置文件中读取参数
- 随机抽取指定数量的数据
- 验证数据格式
- 保存到data目录

**输出:**
- `data/sampled_data_20000.jsonl` - 采样后的数据

## 模型评测脚本

### download_dataset.py
下载HumanEvalPlus数据集（在有网络的环境下运行）。

```bash
python scripts/download_dataset.py
```

**功能:**
- 下载HumanEvalPlus数据集到本地缓存
- 支持断点续传
- 检查数据集完整性

### evaluate_model.py
使用EvalPlus进行HumanEval评测（支持离线模式）。

```bash
python scripts/evaluate_model.py
```

**功能:**
- 支持离线模式（使用本地数据集）
- 从配置文件中读取评测参数
- 使用EvalPlus框架进行评测
- 自动生成评测报告
- 支持多种推理后端

**输出:**
- `evalplus_results/` - EvalPlus原始结果目录
- `outputs/results/` - 评测结果目录
- `outputs/results/evaluation_report.md` - 评测报告

## 配置文件说明

### config/base.yaml
基础配置文件，包含：
- 数据路径配置
- 模型路径配置
- 训练参数配置
- 评测参数配置

### config/eval.yaml
评测专用配置文件，包含：
- EvalPlus评测参数
- 生成参数配置
- 输出路径配置

## 使用流程

### 在线模式（有网络连接）
1. **数据采样**
   ```bash
   python scripts/sample_data.py
   ```

2. **模型训练** (待实现)
   ```bash
   # Qwen SFT方式
   python scripts/train_qwen_sft.py
   
   # LLaMA Factory方式
   python scripts/train_llamafactory.py
   ```

3. **模型评测**
   ```bash
   python scripts/evaluate_model.py
   ```

### 离线模式（无网络连接）
1. **在有网络的环境下下载数据集**
   ```bash
   python scripts/download_dataset.py
   ```

2. **将数据集复制到GPU服务器**
   ```bash
   # 数据集位置: ~/.cache/evalplus/HumanEvalPlus.jsonl.gz
   ```

3. **在GPU服务器上运行评测**
   ```bash
   python scripts/evaluate_model.py
   ```

## EvalPlus使用说明

本项目使用[EvalPlus](https://github.com/evalplus/evalplus)进行HumanEval评测。

### 安装方式
脚本会自动安装EvalPlus，支持两种方式：
1. 从GitHub安装最新版本：`evalplus[vllm] @ git+https://github.com/evalplus/evalplus`
2. 从PyPI安装稳定版本：`evalplus[vllm]`

### 支持的推理后端
- `hf`: HuggingFace Transformers
- `vllm`: vLLM推理引擎
- `openai`: OpenAI兼容API
- `anthropic`: Anthropic API
- `google`: Google Gemini API

### 评测数据集
- `humaneval`: HumanEval数据集
- `mbpp`: MBPP数据集

## 注意事项

1. 确保已安装所有依赖：`uv sync`
2. 确保模型路径正确配置
3. 确保有足够的磁盘空间存储结果
4. 评测过程可能需要较长时间，请耐心等待
5. 首次运行会自动安装EvalPlus，可能需要一些时间
6. 如果GPU服务器无网络连接，请先在有网络的环境下下载数据集