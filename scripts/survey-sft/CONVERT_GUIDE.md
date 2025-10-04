# 数据集转换指南

## 已验证的15个数据集格式

所有数据集格式已经过详细检查，转换脚本已针对实际字段进行优化：

| # | 数据集名称 | 格式 | 关键字段 | 数据量级 |
|---|-----------|------|---------|---------|
| 1 | apps | JSONL | question, solutions | 中等 |
| 2 | tiny-codes | Parquet | prompt, response | 大 |
| 3 | commitpackft | JSONL | subject, old_contents, new_contents | 大 |
| 4 | stackexchange_codereview | Parquet | conversations / instruction+completion | 中等 |
| 5 | code_contests | Parquet | description, solutions | 中等 |
| 6 | ReflectionSeq-GPT | JSONL | messages (需解析) | 中等 |
| 7 | Codeforces-Python-Submissions | Parquet | prompt+response / code | 大 |
| 8 | self-oss-instruct-sc2-exec-filter-50k | Parquet | instruction, response | 5万 |
| 9 | real-world-swe-problems | Parquet | prompt, gold_standard_solution | 小 |
| 10 | stack-exchange-paired | Parquet | question, response_j/k | 大 |
| 11 | react-code-instructions | JSONL | messages | 中等 |
| 12 | stackexchange-question-answering | Parquet | prompt, gold_standard_solution | 中等 |
| 13 | SYNTHETIC-2-SFT-verified | Parquet | messages | 中等 |
| 14 | sql-create-context-instruction | Parquet | text (需解析) | 小 |
| 15 | Magpie-Qwen2.5-Coder-Pro-300K-v0.1 | Parquet | conversations / instruction+response | 30万 |

## 🚀 全量转换命令

### 方式1：转换所有数据集（推荐）

```bash
# 进入项目目录
cd /volume/pt-train/users/wzhang/wjj-workspace/LightSFT-Qwen3-Humaneval

# 全量转换（不限制样本数）
uv run scripts/survey-sft/convert_all_datasets.py

# 查看转换后的文件
ls -lh data/llamafactory/

# 查看dataset_info.json
cat data/dataset_info.json
```

### 方式2：限制样本数转换（测试用）

```bash
# 每个数据集最多转换10000条
uv run scripts/survey-sft/convert_all_datasets.py --max-samples 10000

# 每个数据集最多转换50000条
uv run scripts/survey-sft/convert_all_datasets.py --max-samples 50000
```

### 方式3：转换指定数据集

```bash
# 只转换几个小数据集
uv run scripts/survey-sft/convert_all_datasets.py --datasets \
  apps \
  self-oss-instruct-sc2-exec-filter-50k \
  real-world-swe-problems

# 转换Magpie大数据集
uv run scripts/survey-sft/convert_all_datasets.py --datasets Magpie-Qwen2.5-Coder-Pro-300K-v0.1
```

### 方式4：后台批量转换（推荐用于全量转换）

```bash
# 后台运行全量转换
nohup uv run scripts/survey-sft/convert_all_datasets.py > data/llamafactory/convert_all.log 2>&1 &

# 查看进度
tail -f data/llamafactory/convert_all.log

# 查看进程
ps aux | grep convert_all_datasets

# 查看已转换文件
watch -n 5 'ls -lh data/llamafactory/*.jsonl | wc -l'
```

## 📊 转换输出说明

### 输出文件位置
- **转换数据**: `/volume/pt-train/users/wzhang/wjj-workspace/LightSFT-Qwen3-Humaneval/data/llamafactory/<dataset_name>.jsonl`
- **配置文件**: `/volume/pt-train/users/wzhang/wjj-workspace/LightSFT-Qwen3-Humaneval/data/dataset_info.json`

### 输出格式
所有数据集统一转换为 LLaMA-Factory 的 ShareGPT 格式：

```jsonl
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
...
```

### dataset_info.json 示例

```json
{
  "apps": {
    "file_name": "llamafactory/apps.jsonl",
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
  },
  "tiny_codes": { ... },
  ...
}
```

## 🎯 预估转换时间和空间

| 数据集 | 预估条数 | 预估大小 | 转换时间 |
|-------|---------|---------|---------|
| apps | ~13万 | ~500MB | 5-10分钟 |
| tiny-codes | ~163万 | ~5GB | 30-60分钟 |
| commitpackft | ~100万+ | ~3GB | 30-60分钟 |
| Magpie-Qwen2.5-Coder-Pro-300K-v0.1 | 30万 | ~1GB | 10-20分钟 |
| 其他小数据集 | 各1-10万 | 各100MB-500MB | 各5-10分钟 |
| **总计** | **~400万条** | **~15-20GB** | **2-4小时** |

## 💡 使用建议

### 1. 分批转换策略

```bash
# 第一批：小数据集（快速测试）
uv run scripts/survey-sft/convert_all_datasets.py --datasets \
  apps \
  self-oss-instruct-sc2-exec-filter-50k \
  real-world-swe-problems \
  sql-create-context-instruction \
  ReflectionSeq-GPT

# 第二批：中等数据集
uv run scripts/survey-sft/convert_all_datasets.py --datasets \
  code_contests \
  stackexchange_codereview \
  Codeforces-Python-Submissions \
  stackexchange-question-answering \
  SYNTHETIC-2-SFT-verified

# 第三批：大数据集
uv run scripts/survey-sft/convert_all_datasets.py --datasets \
  Magpie-Qwen2.5-Coder-Pro-300K-v0.1 \
  tiny-codes \
  commitpackft \
  stack-exchange-paired \
  react-code-instructions
```

### 2. 监控转换进度

```bash
# 实时监控日志
tail -f data/llamafactory/convert_all.log

# 检查已转换数据集数量
ls data/llamafactory/*.jsonl | wc -l

# 检查转换后的总大小
du -sh data/llamafactory/

# 查看各数据集大小
ls -lh data/llamafactory/*.jsonl | awk '{print $9, $5}'
```

### 3. 验证转换结果

```bash
# 检查某个数据集的前几条数据
head -3 data/llamafactory/apps.jsonl | python3 -m json.tool

# 统计某个数据集的行数
wc -l data/llamafactory/apps.jsonl

# 验证JSON格式正确性
cat data/llamafactory/apps.jsonl | while read line; do echo "$line" | python3 -c "import sys, json; json.loads(sys.stdin.read())"; done | head -100
```

## 🔧 故障排除

### 问题1：内存不足
```bash
# 限制每个数据集的样本数
uv run scripts/survey-sft/convert_all_datasets.py --max-samples 100000
```

### 问题2：某个数据集转换失败
```bash
# 单独转换该数据集进行调试
uv run scripts/survey-sft/convert_all_datasets.py --datasets <dataset_name>
```

### 问题3：磁盘空间不足
```bash
# 检查磁盘空间
df -h /volume/pt-train/users/wzhang/wjj-workspace/LightSFT-Qwen3-Humaneval/data/

# 只转换部分数据集
uv run scripts/survey-sft/convert_all_datasets.py --datasets apps tiny-codes --max-samples 50000
```

## ✅ 推荐执行方案

**最佳方案**：后台全量转换

```bash
cd /volume/pt-train/users/wzhang/wjj-workspace/LightSFT-Qwen3-Humaneval

# 创建转换日志目录
mkdir -p data/llamafactory

# 后台执行全量转换
nohup uv run scripts/survey-sft/convert_all_datasets.py > data/llamafactory/convert_all.log 2>&1 &

# 记录进程ID
echo $! > data/llamafactory/convert.pid

# 监控进度（Ctrl+C退出监控，不影响后台转换）
tail -f data/llamafactory/convert_all.log

# 后续检查进度
tail -100 data/llamafactory/convert_all.log

# 转换完成后检查
cat data/dataset_info.json
ls -lh data/llamafactory/
```

预计2-4小时后，所有15个数据集将转换完成，可直接用于LLaMA-Factory训练！

