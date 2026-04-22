# KV-Tool Cache Co-design

> **PhD Research Project** — Intelligent Co-design of KV Cache & Tool Cache for Large Language Model Inference  
> Unified Memory Management · Adaptive Prefetching · Task-Aware Scheduling

[![Python](https://img.shields.io/badge/Python-3.13-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10-orange)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/HuggingFace-4.57-yellow)](https://huggingface.co)
[![Model](https://img.shields.io/badge/Model-DistilGPT--2-green)](https://huggingface.co/distilgpt2)
[![Platform](https://img.shields.io/badge/Platform-CPU%20%2F%20macOS-lightgrey)](.)

---

## 项目简介

本项目是一个**博士级别**的实验研究工程，探索在大语言模型（LLM）推理过程中，将 **KV Cache（注意力键值缓存）** 与 **Tool Cache（工具调用结果缓存）** 进行智能协同设计，以降低推理延迟、提升缓存命中率，同时不损失任何生成质量。

### 核心问题

当前大模型推理系统中，KV Cache 和 Tool Cache 是**完全独立**的两套系统：

```
传统方案（Independent）：
  ┌─────────────────┐    ┌─────────────────┐
  │   KV Cache      │    │   Tool Cache    │
  │ (LRU, 独立管理)  │    │ (LFU, 独立管理)  │
  └─────────────────┘    └─────────────────┘
         ↑                        ↑
         │ 完全割裂，无语义关联     │
```

本项目提出协同设计方案：

```
本文方案（Co-design）：
  ┌────────────────────────────────────────────┐
  │          Unified Scheduler                  │
  │  ┌──────────────────┐  KV-Tool Linking      │
  │  │  Tool Cache(LFU) │ ←──────────────────► │
  │  │  Tool事件触发     │    KV Prefetch        │
  │  └──────────────────┘                       │
  │  ┌──────────────────┐                       │
  │  │  KV Cache (LRU)  │  past_key_values      │
  │  │  真实HF API复用  │                       │
  │  └──────────────────┘                       │
  └────────────────────────────────────────────┘
```

---

## 实验结果摘要

| 指标 | 数值 | 意义 |
|------|------|------|
| 重复工具调用占比 | **53.3%** | 工具层冗余严重，缓存收益显著 |
| KV Token 冗余率 | **54.7%** | 超过一半 KV 存储可复用 |
| 同类工具 KV 相似度 | **0.207**（跨类 0.098） | 语义关联是协同设计的数学基础 |
| Vanilla → Co-design 延迟降低 | **55.1%** | 0.258s → 0.116s |
| KV 复用率：Co-design vs Independent | **100% vs 53.3%** | 预取机制带来结构性提升 |
| PPL Delta（质量验证） | **0.00** ✓ | 无任何精度损失 |
| 工具调用准确率 | **100% = 100%** | 质量完全保真 |
| 消融：去除 KV-Tool Linking | 联合命中率 **0%**，延迟 **+34.6%** | 核心模块不可缺少 |

---

## 项目结构

```
kv-tool-cache/
├── main.py                      # 主实验代码（7个实验全部在此）
├── instrument.md                # 实验设计文档（课题设计思路与实验规范）
├── console_output.md            # 真实运行输出记录
├── experiment_cache_codesign.py # 早期实验版本（参考）
├── experiment_analysis.md       # 实验分析记录
├── diss.md                      # 博士答辩级实验汇报手册（问答式）
├── skills.md                    # 本项目技术能力清单
└── README.md                    # 本文件
```

---

## 快速开始

### 环境要求

```bash
Python >= 3.11
PyTorch >= 2.0
transformers >= 4.50
numpy >= 2.0
```

### 安装依赖

```bash
pip install torch transformers numpy
```

> ⚠️ 注意：如你的环境中 `tokenizers` 版本较旧，请先升级：
> ```bash
> pip install tokenizers --upgrade
> ```

### 运行全部实验

```bash
cd kv-tool-cache
python main.py
```

预计运行时间：**3~8 分钟**（CPU，DistilGPT-2）

首次运行会自动下载 DistilGPT-2 模型（~350MB）。

---

## 实验体系

本项目共 **7 个实验**，构成完整的博士论文实验体系：

### Exp 1 — 缓存冗余量化（Motivation 核心）
**验证**：KV Cache 与 Tool Cache 之间存在高度语义关联，独立设计造成大量冗余。  
**Workload**：15 轮多工具对话（search/calculator/retrieval）  
**关键指标**：KV 余弦相似度、Jaccard 重叠率、KV Token 冗余率  
**结论**：同类工具 KV 相似度 2.1× 高于跨类，KV 冗余率 54.7%

### Exp 2 — 端到端推理延迟对比
**验证**：协同设计相比独立方案显著降低推理延迟，高重复场景收益更大。  
**Baseline**：Vanilla / KV-only / Independent / Co-design  
**结论**：Co-design 整体降低 55% 延迟，高重复场景额外降低 6.3%

### Exp 3 — 缓存命中率拆解
**验证**：协同设计提升的是关键的联合命中率（Tool AND KV 同时命中）。  
**关键指标**：Tool HR / KV HR / Joint HR / Invalid HR  
**结论**：KV 复用率从 53.3% 提升至 **100%**（+88%）

### Exp 4 — 生成质量验证
**验证**：缓存机制不引入任何精度损失，排除"精度换速度"质疑。  
**指标**：PPL Delta、工具调用准确率  
**结论**：PPL Delta = **0.00**，工具准确率 **100% = 100%**

### Exp 5 — 长上下文可扩展性
**验证**：协同设计在更长上下文下延迟增长更平缓。  
**Workload**：32 / 64 / 128 / 256 / 512 tokens  
**结论**：Co-design 在所有上下文长度下绝对延迟均更低

### Exp 6 — 消融研究
**验证**：每个设计模块都是必要的，去掉任何一个都导致性能下降。  
**方案**：5 组消融（w/o Linking / Eviction / Prefetch / Scheduler）  
**结论**：KV-Tool Linking 最关键，去除后延迟 +34.6%，命中率归零

### Exp 7 — 可视化分析
**内容**：工具频率热力图、延迟对比柱状图、可扩展性曲线、消融瀑布图  
**形式**：ASCII 可视化（可扩展为 matplotlib 图表）

---

## 核心技术亮点

### 1. 真实 KV Cache 复用（非模拟）

```python
# 编码 prefix，获取 past_key_values 和首 token logits
_, past_kv, first_logits, _ = _encode_prefix(prompt)
kv_store.put(key, (past_kv, first_logits[:, 0, :]))

# 命中时：直接使用缓存的 KV，跳过 prefix 重新计算
past_kv, current_logits = kv_store.get(key)
# 直接进入自回归解码 → 完全等价于无缓存路径（PPL Delta = 0）
```

### 2. Tool 事件驱动 KV 预取

```python
# Tool Cache Miss → 立即预取 KV（不等到下次 KV miss 再计算）
if not tool_hit:
    tool_result = TOOL_REGISTRY[tool_name](**params)
    tool_cache.put(tool_key, tool_result)
    # ★ 预取：同步写入 KV，保证下次 Tool 命中时 KV 必命中
    _, pkv, first_logits, _ = _encode_prefix(follow_up_prompt)
    kv_store.put(_prompt_key(follow_up_prompt), (pkv, first_logits[:, 0, :]))
```

### 3. LFU + LRU 协同淘汰策略

```python
# Tool Cache: LFU（保护高频工具，适合工具调用局部性）
tool_cache = LFUCache(capacity=128)

# KV Cache: LRU（保护最近上下文，适合对话连续性）
kv_store = LRUCache(capacity=128)
```

---

## 对标论文

| 论文 | 关联 |
|------|------|
| [LMCache (arXiv:2510.09665)](https://arxiv.org/abs/2510.09665) | KV Cache 统一管理框架，本项目 Tool Cache 协同设计是其扩展 |
| vLLM [SOSP 2023] | PagedAttention KV 分页缓存，是 KV Cache Only 基线的工业实现 |
| ToolBench [ICLR 2024] | 工具调用数据集与评测基准，提供 Tool Cache 的现实 workload |
| GPTCache | 语义缓存（输入-输出级），与本系统互补（计算过程级 vs 结果级） |
| SGLang [arXiv:2312.07104] | 前缀 KV 共享，与本项目 KV 复用机制同源，本项目额外引入 Tool 维度 |

---

## 研究贡献

1. **首次量化** KV Cache 与 Tool Cache 之间的语义相关性（余弦相似度分析）
2. **提出并验证** Tool 事件驱动 KV 预取机制，KV 复用率从 53.3% 提升至 100%
3. **严格证明** 协同缓存在 Greedy Decoding 下 PPL Delta = 0（数学等价性）
4. **构建完整** 7 实验体系，覆盖 Motivation / System / Quality / Scalability / Ablation

---

## 未来工作

- [ ] 在 Llama-2-7B / Qwen-7B + A100 GPU 上重现实验（大模型场景验证）
- [ ] 引入语义模糊匹配（Sentence Embedding）扩展 Tool Cache 命中范围
- [ ] 多用户并发场景的跨请求 KV 共享与一致性协议
- [ ] 基于强化学习的自适应缓存预取策略
- [ ] 集成进 vLLM PagedAttention 框架的工程实现

---

## 答辩资料

详见 [`diss.md`](./diss.md) — 包含 20 个导师最常追问的问题及标准回答，按博士答辩严格程度整理。

---

## 作者

**PhD Candidate — LLM System Research**  
Research Direction: Efficient LLM Inference, Cache System Co-design, Tool-Augmented Language Models  

---

*Last Updated: 2026-04-22*

