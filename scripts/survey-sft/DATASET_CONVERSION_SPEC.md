# 数据集转换详细规范文档

本文档详细说明了15个代码数据集如何转换为 LLaMA-Factory 的 ShareGPT 格式。

## 目录
- [转换格式说明](#转换格式说明)
- [数据集详细转换规则](#数据集详细转换规则)
- [特殊情况处理](#特殊情况处理)
- [质量控制](#质量控制)

---

## 转换格式说明

### 目标格式：ShareGPT

所有数据集统一转换为以下 JSONL 格式：

```json
{
  "messages": [
    {
      "role": "user",
      "content": "用户问题或指令"
    },
    {
      "role": "assistant",
      "content": "AI助手的回答"
    }
  ]
}
```

### 支持的角色
- `user`: 用户/人类的输入
- `assistant`: AI助手的输出
- `system`: 系统提示词（部分数据集包含）

---

## 数据集详细转换规则

### 1. apps (APPS 编程竞赛题目)

**源格式**: JSONL
**文件**: `data/apps/train.jsonl`

#### 原始数据结构
```json
{
  "id": 0,
  "question": "问题描述（含数学公式、输入输出示例等）",
  "solutions": "[\"solution1\", \"solution2\", ...]",  // JSON字符串
  "input_output": "...",
  "difficulty": "interview|competition|introductory",
  "url": "https://codeforces.com/...",
  "starter_code": "..."
}
```

#### 转换逻辑
1. 提取 `question` 字段作为用户问题
2. 解析 `solutions` JSON字符串为列表
3. 取第一个解决方案 `solutions[0]` 作为助手回答
4. 将代码包装在 ``` 代码块中

#### 转换后示例
```json
{
  "messages": [
    {
      "role": "user",
      "content": "Solve this programming problem:\n\nPolycarp has $n$ different binary words..."
    },
    {
      "role": "assistant",
      "content": "```python\nfor _ in range(int(input())):\n    n = int(input())\n    ...\n```"
    }
  ]
}
```

#### 数据特点
- **优点**: 题目描述详细，包含完整的输入输出示例
- **注意**: solutions可能包含多个解法，我们选择第一个
- **预估数量**: ~130,000 条

---

### 2. tiny-codes (小型代码片段)

**源格式**: Parquet
**文件**: `data/tiny-codes/part_*.parquet`

#### 原始数据结构
```python
{
  'prompt': 'Develop a Python program snippet to...',
  'response': 'Here is python code which...',
  'main_topic': '...',
  'subtopic': '...',
  'programming_language': 'Python'
}
```

#### 转换逻辑
1. 直接使用 `prompt` 字段作为用户问题
2. 直接使用 `response` 字段作为助手回答
3. 不需要额外处理

#### 转换后示例
```json
{
  "messages": [
    {
      "role": "user",
      "content": "Develop a Python program snippet to Determine High Sneezing..."
    },
    {
      "role": "assistant",
      "content": "Here is python code which determines high sneezing..."
    }
  ]
}
```

#### 数据特点
- **优点**: 格式规范，直接可用
- **覆盖**: 涵盖多种编程主题
- **预估数量**: ~1,632,520 条

---

### 3. commitpackft (Git Commit 代码变更)

**源格式**: JSONL (多语言目录)
**文件**: `data/commitpackft/data/*/data.jsonl`

#### 原始数据结构
```json
{
  "commit": "be7d2a9d457026e24fc8a2b8bba4cf67f58f40d5",
  "old_file": "path/to/old.py",
  "new_file": "path/to/new.py",
  "old_contents": "旧代码内容（可能为空）",
  "new_contents": "新代码内容",
  "subject": "Add some simple logic inductive types",
  "message": "完整的commit message",
  "lang": "Harbour",
  "license": "mit",
  "repos": "..."
}
```

#### 转换逻辑

**情况1**: 有旧代码（重构场景）
```
用户问题 = "Refactor the code based on: {subject}\n\nOld code:\n```\n{old_contents}\n```"
助手回答 = "```\n{new_contents}\n```"
```

**情况2**: 无旧代码（新增文件）
```
用户问题 = "Implement: {subject}"
助手回答 = "```\n{new_contents}\n```"
```

#### 转换后示例
```json
{
  "messages": [
    {
      "role": "user",
      "content": "Refactor the code based on: Fix bug in user authentication\n\nOld code:\n```\ndef login(user):\n    return True\n```"
    },
    {
      "role": "assistant",
      "content": "```\ndef login(user, password):\n    return verify_password(user, password)\n```"
    }
  ]
}
```

#### 数据特点
- **优点**: 包含代码演化过程，适合代码重构任务
- **语言**: 支持多种编程语言（按目录分类）
- **注意**: 部分条目 old_contents 为空
- **预估数量**: ~1,000,000+ 条

---

### 4. stackexchange_codereview (StackExchange 代码审查)

**源格式**: Parquet
**文件**: `data/stackexchange_codereview/data/train-*.parquet`

#### 原始数据结构
```python
{
  'instruction': '用户的代码审查请求',
  'completion': '审查建议和改进代码',
  'conversations': [
    {'from': 'human', 'value': '...'},
    {'from': 'gpt', 'value': '...'}
  ]
}
```

#### 转换逻辑

**优先级1**: 使用 `conversations` 字段（已格式化的对话）
```python
for msg in conversations:
    role = "user" if msg['from'] == 'human' else "assistant"
    messages.append({"role": role, "content": msg['value']})
```

**优先级2**: 使用 `instruction` + `completion`
```python
messages = [
    {"role": "user", "content": instruction},
    {"role": "assistant", "content": completion}
]
```

#### 转换后示例
```json
{
  "messages": [
    {
      "role": "user",
      "content": "Please review this Python code for performance issues:\n\ndef process_data(items):\n    result = []\n    for item in items:\n        result.append(item * 2)\n    return result"
    },
    {
      "role": "assistant",
      "content": "Here's an improved version using list comprehension:\n\ndef process_data(items):\n    return [item * 2 for item in items]"
    }
  ]
}
```

#### 数据特点
- **优点**: 真实的代码审查场景
- **格式**: 部分包含多轮对话
- **预估数量**: ~50,000 条

---

### 5. code_contests (DeepMind 竞赛编程)

**源格式**: Parquet
**文件**: `data/code_contests/data/train-*.parquet`

#### 原始数据结构
```python
{
  'name': 'Problem A',
  'description': '问题的完整描述（包含输入输出格式）',
  'public_tests': {...},
  'private_tests': {...},
  'generated_tests': {...},
  'source': 'CodeForces',
  'difficulty': 'EASY|MEDIUM|HARD',
  'solutions': {
    'language': [...],
    'solution': [...]
  } 或 ['solution1', 'solution2', ...]
}
```

#### 转换逻辑
1. 提取 `description` 作为问题描述
2. 处理 `solutions`:
   - 如果是字典：提取第一个语言的第一个solution
   - 如果是列表：取第一个solution
3. 添加问题前缀："Solve this competitive programming problem:\n\n"

#### 转换后示例
```json
{
  "messages": [
    {
      "role": "user",
      "content": "Solve this competitive programming problem:\n\nGiven an array of integers, find the maximum sum of a contiguous subarray..."
    },
    {
      "role": "assistant",
      "content": "def max_subarray_sum(arr):\n    max_sum = current_sum = arr[0]\n    for num in arr[1:]:\n        current_sum = max(num, current_sum + num)\n        max_sum = max(max_sum, current_sum)\n    return max_sum"
    }
  ]
}
```

#### 数据特点
- **优点**: 高质量竞赛题目
- **难度**: 包含难度标签
- **来源**: CodeForces, AtCoder 等
- **预估数量**: ~100,000 条

---

### 6. ReflectionSeq-GPT (反思序列对话)

**源格式**: JSONL
**文件**: `data/ReflectionSeq-GPT/train.jsonl`

#### 原始数据结构
```json
{
  "messages": "[{\"role\": \"user\", \"content\": [{\"type\": \"text\", \"content\": \"...\"}]}, ...]",
  "type": "reflection"
}
```

**注意**: `messages` 是 JSON 字符串，需要解析

#### 转换逻辑
1. 解析 `messages` JSON 字符串
2. 处理每条消息的 `content`:
   - 如果是列表：提取所有 `type="text"` 的内容并合并
   - 如果是字符串：直接使用
3. 标准化 role 字段

#### 转换后示例
```json
{
  "messages": [
    {
      "role": "user",
      "content": "Here is a programming challenge for you to tackle..."
    },
    {
      "role": "assistant",
      "content": "Let me think through this step by step..."
    }
  ]
}
```

#### 数据特点
- **优点**: 包含思考过程，适合训练推理能力
- **格式**: 嵌套JSON，需要仔细解析
- **预估数量**: ~50,000 条

---

### 7. Codeforces-Python-Submissions (Codeforces 提交记录)

**源格式**: Parquet
**文件**: `data/Codeforces-Python-Submissions/data/train-*.parquet`

#### 原始数据结构
```python
{
  'contestId': 1234,
  'name': 'Problem A',
  'problem-description': '题目描述',
  'input-specification': '输入格式',
  'output-specification': '输出格式',
  'demo-input': '示例输入',
  'demo-output': '示例输出',
  'code': 'Python代码',
  'prompt': '格式化好的问题（可选）',
  'response': '格式化好的答案（可选）',
  'verdict': 'OK|WRONG_ANSWER|...',
  'rating': 1200
}
```

#### 转换逻辑

**优先级1**: 使用 `prompt` + `response`（已格式化）
```python
if prompt and response:
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response}
    ]
```

**优先级2**: 使用 `problem-description` + `code`
```python
if problem_description and code:
    user_content = f"Solve this Codeforces problem:\n\n{problem_description}"
    assistant_content = f"```python\n{code}\n```"
```

#### 转换后示例
```json
{
  "messages": [
    {
      "role": "user",
      "content": "Solve this Codeforces problem:\n\nGiven an integer n, find the sum of all divisors of n."
    },
    {
      "role": "assistant",
      "content": "```python\nn = int(input())\nsum_divisors = sum(i for i in range(1, n+1) if n % i == 0)\nprint(sum_divisors)\n```"
    }
  ]
}
```

#### 数据特点
- **优点**: 真实的AC代码
- **过滤**: 只保留verdict=OK的提交
- **难度**: 包含rating信息
- **预估数量**: ~500,000 条

---

### 8. self-oss-instruct-sc2-exec-filter-50k (开源指令数据)

**源格式**: Parquet
**文件**: `data/self-oss-instruct-sc2-exec-filter-50k/data/train-00000-of-00001.parquet`

#### 原始数据结构
```python
{
  'fingerprint': None,
  'sha1': 'f35b008f...',
  'seed': '原始代码片段',
  'instruction': '编程任务指令',
  'response': '代码实现',
  'concepts': ['concept1', 'concept2'],
  'prompt': '完整的提示词',
  'id': 165808
}
```

#### 转换逻辑
```python
messages = [
    {"role": "user", "content": instruction},
    {"role": "assistant", "content": response}
]
```

#### 转换后示例
```json
{
  "messages": [
    {
      "role": "user",
      "content": "Write a Python function named `get_value` that takes a matrix and returns the value at a specific index."
    },
    {
      "role": "assistant",
      "content": "def get_value(matrix, indices):\n    row_index, col_index = indices\n    if row_index < 0 or row_index >= len(matrix):\n        return None\n    if col_index < 0 or col_index >= len(matrix[row_index]):\n        return None\n    return matrix[row_index][col_index]"
    }
  ]
}
```

#### 数据特点
- **优点**: 高质量过滤（sc2-exec-filter）
- **数量**: 正好 50,000 条
- **质量**: 经过执行验证

---

### 9. real-world-swe-problems (真实软件工程问题)

**源格式**: Parquet
**文件**: `data/real-world-swe-problems/data/train-00000-of-00001.parquet`

#### 原始数据结构
```python
{
  'source': 'github',
  'task_type': 'bug_fix|feature',
  'in_source_id': 'issue#123',
  'prompt': '问题描述',
  'gold_standard_solution': '标准解决方案',
  'verification_info': {...},
  'metadata': {...},
  'problem_id': 'swe_001'
}
```

#### 转换逻辑
```python
messages = [
    {"role": "user", "content": prompt},
    {"role": "assistant", "content": gold_standard_solution}
]
```

#### 转换后示例
```json
{
  "messages": [
    {
      "role": "user",
      "content": "Fix the bug in the authentication module where users can bypass login by sending empty credentials."
    },
    {
      "role": "assistant",
      "content": "def authenticate(username, password):\n    if not username or not password:\n        raise ValueError('Credentials cannot be empty')\n    return verify_credentials(username, password)"
    }
  ]
}
```

#### 数据特点
- **优点**: 真实场景，实用性强
- **来源**: GitHub issues/PRs
- **预估数量**: ~10,000 条

---

### 10. stack-exchange-paired (StackExchange 偏好对)

**源格式**: Parquet
**文件**: `data/stack-exchange-paired/data/reward/*.parquet`

#### 原始数据结构
```python
{
  'qid': 'question_id',
  'question': '用户问题',
  'date': '2023-01-01',
  'metadata': {...},
  'response_j': '更好的回答（获得更多赞）',
  'response_k': '较差的回答（获得较少赞）'
}
```

#### 转换逻辑（SFT场景）

**重要**: 这是DPO偏好数据，包含两个回答

对于SFT训练：
```python
# 只使用更好的回答 (response_j)
messages = [
    {"role": "user", "content": question},
    {"role": "assistant", "content": response_j}  # 优先使用 j
]
```

对于DPO训练（未实现）：
```python
# 需要同时保留 response_j 和 response_k
# 用于对比学习
```

#### 转换后示例
```json
{
  "messages": [
    {
      "role": "user",
      "content": "How do I reverse a string in Python?"
    },
    {
      "role": "assistant",
      "content": "You can reverse a string using slicing: `reversed_str = original[::-1]`"
    }
  ]
}
```

#### 数据特点
- **类型**: DPO偏好数据
- **用途**: SFT使用response_j, DPO使用j+k
- **质量**: 基于社区投票排序
- **预估数量**: ~300,000 条

---

### 11. react-code-instructions (React 代码指令)

**源格式**: JSONL
**文件**: `data/react-code-instructions/data/*.jsonl`

#### 原始数据结构
```json
{
  "created_at": "2024-01-01",
  "model": "claude-3.5-sonnet",
  "messages": [
    {"role": "system", "content": "You are an expert React developer..."},
    {"role": "user", "content": "Create a dashboard..."},
    {"role": "assistant", "content": "Here's the code..."}
  ],
  "recommended": true,
  "upvoted": true
}
```

#### 转换逻辑
```python
# messages 已经是标准格式，直接使用
messages = data['messages']  # 保留 system, user, assistant
```

#### 转换后示例
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an expert React, TypeScript, and TailwindCSS developer..."
    },
    {
      "role": "user",
      "content": "Create a modern dashboard with charts and tables"
    },
    {
      "role": "assistant",
      "content": "import React from 'react';\n\nconst Dashboard = () => {...}"
    }
  ]
}
```

#### 数据特点
- **优点**: 包含system prompt
- **质量**: 经过推荐和投票筛选
- **技术栈**: React + TypeScript + TailwindCSS
- **预估数量**: ~20,000 条

---

### 12. stackexchange-question-answering (StackExchange 问答)

**源格式**: Parquet
**文件**: `data/stackexchange-question-answering/data/train-*.parquet`

#### 原始数据结构
```python
{
  'source': 'stackoverflow',
  'task_type': 'qa',
  'in_source_id': 'q_12345',
  'prompt': '问题（已格式化）',
  'gold_standard_solution': '标准答案',
  'verification_info': {...},
  'metadata': {...},
  'problem_id': 'se_qa_001'
}
```

#### 转换逻辑
```python
messages = [
    {"role": "user", "content": prompt},
    {"role": "assistant", "content": gold_standard_solution}
]
```

#### 转换后示例
```json
{
  "messages": [
    {
      "role": "user",
      "content": "Below is a question asked on Stackoverflow. Provide a good response:\n\nHow do I connect to a PostgreSQL database in Python?"
    },
    {
      "role": "assistant",
      "content": "You can use the psycopg2 library:\n\nimport psycopg2\nconn = psycopg2.connect(\n    host='localhost',\n    database='mydb',\n    user='user',\n    password='password'\n)"
    }
  ]
}
```

#### 数据特点
- **来源**: StackOverflow, ServerFault 等
- **质量**: 高赞答案
- **预估数量**: ~100,000 条

---

### 13. SYNTHETIC-2-SFT-verified (合成验证数据)

**源格式**: Parquet
**文件**: `data/SYNTHETIC-2-SFT-verified/data/train-*.parquet`

#### 原始数据结构
```python
{
  'problem_id': 'syn_001',
  'task_type': 'code_generation',
  'reward': 0.95,
  'messages': [
    {'role': 'user', 'content': '...'},
    {'role': 'assistant', 'content': '...'}
  ]
}
```

#### 转换逻辑
```python
# messages 已经是标准格式
messages = row['messages']  # 直接使用
```

#### 转换后示例
```json
{
  "messages": [
    {
      "role": "user",
      "content": "Write a function to calculate factorial"
    },
    {
      "role": "assistant",
      "content": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)"
    }
  ]
}
```

#### 数据特点
- **优点**: 合成数据，经过验证
- **reward**: 包含质量评分
- **预估数量**: ~50,000 条

---

### 14. sql-create-context-instruction (SQL 指令数据)

**源格式**: Parquet
**文件**: `data/sql-create-context-instruction/data/train-*.parquet`

#### 原始数据结构
```python
{
  'text': '[INST] Write SQLite query to answer...\nSchema: CREATE TABLE...\nQuestion: ...\n[/INST] Here is the query:\nSELECT...'
}
```

**特殊格式**: 使用 `[INST]...[/INST]` 分隔

#### 转换逻辑
```python
# 分割 [INST] 和 [/INST]
parts = text.split('[/INST]')
instruction = parts[0].replace('[INST]', '').strip()
response = parts[1].strip() if len(parts) > 1 else ''

messages = [
    {"role": "user", "content": instruction},
    {"role": "assistant", "content": response}
]
```

#### 转换后示例
```json
{
  "messages": [
    {
      "role": "user",
      "content": "Write SQLite query to answer the following question:\nSchema: CREATE TABLE employees (id INT, name TEXT, age INT)\nQuestion: How many employees are older than 30?"
    },
    {
      "role": "assistant",
      "content": "Here is the SQLite query:\nSELECT COUNT(*) FROM employees WHERE age > 30"
    }
  ]
}
```

#### 数据特点
- **格式**: 特殊的 [INST] 标记
- **内容**: SQL查询生成
- **预估数量**: ~5,000 条

---

### 15. Magpie-Qwen2.5-Coder-Pro-300K-v0.1 (Magpie 合成数据)

**源格式**: Parquet
**文件**: `data/Magpie-Qwen2.5-Coder-Pro-300K-v0.1/data/train-*.parquet`

#### 原始数据结构
```python
{
  'uuid': '6e94d8f4-...',
  'model': 'Qwen/Qwen2.5-Coder-32B-Instruct',
  'instruction': '问题',
  'response': '回答',
  'conversations': [
    {'from': 'human', 'value': '...'},
    {'from': 'gpt', 'value': '...'}
  ],
  'task_category': 'Coding & Debugging',
  'difficulty': 'easy|medium|hard',
  'quality_explanation': '...',
  'reward_model': 0.85,
  'language': 'EN'
}
```

#### 转换逻辑

**优先级1**: 使用 `conversations`（完整对话）
```python
messages = []
for msg in conversations:
    role = 'user' if msg['from'] == 'human' else 'assistant'
    messages.append({"role": role, "content": msg['value']})
```

**优先级2**: 使用 `instruction` + `response`
```python
messages = [
    {"role": "user", "content": instruction},
    {"role": "assistant", "content": response}
]
```

#### 转换后示例
```json
{
  "messages": [
    {
      "role": "user",
      "content": "What is the cause of the problem with the `readLine()` function?"
    },
    {
      "role": "assistant",
      "content": "The `readLine()` function is not a standard Python function. You might be referring to `readline()`, which is used with file objects..."
    }
  ]
}
```

#### 数据特点
- **优点**: 高质量合成数据
- **模型**: 由 Qwen2.5-Coder-32B 生成
- **质量**: 包含难度和质量评分
- **预估数量**: 300,000 条

---

## 特殊情况处理

### 1. 空值处理
```python
# 检查必要字段是否存在且非空
if not prompt or not response:
    return None  # 跳过此条数据
```

### 2. 数据类型转换
```python
# solutions 可能是字符串或列表
if isinstance(solutions, str):
    solutions = json.loads(solutions)
```

### 3. 多解决方案选择
```python
# 总是选择第一个解决方案
if isinstance(solutions, list) and len(solutions) > 0:
    solution = solutions[0]
```

### 4. 代码块包装
```python
# 为代码添加语法高亮
if is_code:
    content = f"```python\n{code}\n```"
```

### 5. 角色映射
```python
# 标准化不同的角色名称
ROLE_MAP = {
    'human': 'user',
    'gpt': 'assistant',
    'from': {'human': 'user', 'gpt': 'assistant'}
}
```

---

## 质量控制

### 数据过滤规则

1. **必要字段检查**
   - 用户问题和助手回答都必须非空
   - 跳过缺少关键字段的数据

2. **长度限制**（可选）
   - 问题长度: 10 - 10,000 字符
   - 回答长度: 10 - 50,000 字符

3. **格式验证**
   - JSON 解析必须成功
   - messages 列表必须包含至少一对问答

4. **去重**（可选）
   - 基于内容哈希去除完全重复的数据

### 数据统计

每个转换后的文件应包含以下统计信息：
- 总条数
- 平均问题长度
- 平均回答长度
- 包含代码块的比例

---

## 使用建议

### 1. 数据集选择

**适合初学者训练**:
- tiny-codes (简单代码片段)
- apps (基础编程题)

**适合进阶训练**:
- code_contests (竞赛题)
- Codeforces (真实提交)

**适合实战训练**:
- real-world-swe-problems
- stackexchange_codereview

**适合多轮对话**:
- react-code-instructions
- ReflectionSeq-GPT

### 2. 数据混合策略

```yaml
# LLaMA-Factory 配置示例
dataset: apps,tiny_codes,code_contests,Magpie_qwen2_5_coder_pro_300k_v0_1
dataset_dir: data
cutoff_len: 2048
```

### 3. 质量分层

**高质量** (优先使用):
- Magpie-Qwen2.5-Coder-Pro-300K (有质量评分)
- self-oss-instruct (经过执行验证)
- SYNTHETIC-2-SFT-verified (有reward评分)

**中等质量**:
- apps, code_contests, Codeforces
- stackexchange 系列

**实验性**:
- commitpackft (可能包含非标准格式)

---

## 附录

### 完整的转换命令

```bash
# 全量转换所有15个数据集
uv run scripts/survey-sft/convert_all_datasets.py

# 限制每个数据集样本数
uv run scripts/survey-sft/convert_all_datasets.py --max-samples 50000

# 只转换指定数据集
uv run scripts/survey-sft/convert_all_datasets.py --datasets apps tiny-codes Magpie-Qwen2.5-Coder-Pro-300K-v0.1
```

### 验证转换结果

```bash
# 检查某个数据集格式
head -1 data/llamafactory/apps.jsonl | python3 -m json.tool

# 统计行数
wc -l data/llamafactory/*.jsonl

# 检查所有数据集
for file in data/llamafactory/*.jsonl; do
    echo "=== $(basename $file) ==="
    echo "Lines: $(wc -l < $file)"
    echo "Size: $(du -h $file | cut -f1)"
    head -1 $file | python3 -c "import sys, json; d=json.load(sys.stdin); print('Keys:', list(d.keys())); print('Roles:', [m['role'] for m in d.get('messages', [])])"
    echo
done
```

---

**文档版本**: 1.0
**最后更新**: 2025-10-03
**维护者**: LightSFT-Qwen3-Humaneval Project

