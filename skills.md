# Skills & Techniques Demonstrated
## Project: Intelligent Co-design of KV Cache & Tool Cache for LLM Inference

> 本文件列出本项目中涉及的全部技术能力，适用于简历、开题报告技术能力说明、答辩能力展示。

---

## 一、大模型推理系统（LLM Inference Systems）

| 技能 | 具体体现 |
|------|----------|
| Transformer 注意力机制 | 理解并正确使用 `past_key_values`（KV Cache）API，实现 prefix 前向计算的跳过与复用 |
| 自回归解码（Autoregressive Decoding） | 手工实现 Greedy Argmax 逐 token 生成，无依赖 `model.generate()` 黑盒 |
| KV Cache 原理 | 清楚 KV Cache 的层级结构（每层 key/value tensor 的形状：`[batch, heads, seq_len, head_dim]`）及显存开销估算 |
| 因果语言模型（Causal LM） | 正确处理 `use_cache=True/False`、`past_key_values` 传入方式、logits 提取位置 |
| 困惑度（Perplexity）计算 | 通过 `model(ids, labels=ids).loss` 精确计算 PPL，并用于生成质量验证 |
| 模型加载与配置 | HuggingFace `AutoTokenizer` / `AutoModelForCausalLM`，`torch_dtype`，设备映射 |

---

## 二、缓存系统设计（Cache System Design）

| 技能 | 具体体现 |
|------|----------|
| LRU Cache 实现 | 基于 `collections.OrderedDict` 实现 O(1) get/put，含命中率统计 |
| LFU Cache 实现 | 基于频率字典实现 Least Frequently Used 淘汰，适合工具调用高频重复场景 |
| 缓存 Key 设计 | 使用 `MD5(prompt)` 和 `MD5(tool_name + JSON(params))` 设计确定性唯一键 |
| 联合缓存调度 | 设计 KV Cache 与 Tool Cache 的协同索引（Tool Key → KV Key）机制 |
| 缓存预取（Prefetching） | 工具 Cache Miss 时主动触发 KV Prefetch，保证下次必命中 |
| 缓存命中率分析 | 拆解 Tool HR / KV HR / Joint HR / Invalid HR 四维度命中率体系 |
| 缓存冗余量化 | 设计 KV Token 冗余率、Jaccard 重叠率、KV 余弦相似度三种量化方法 |

---

## 三、实验设计能力（Experimental Design）

| 技能 | 具体体现 |
|------|----------|
| Ablation Study 设计 | 设计 5 组消融方案，每次关闭一个模块，定量分析每个模块贡献 |
| Baseline 体系设计 | 设计 4 种对比基线：Vanilla / KV-only / Independent / Co-design |
| Workload 分层 | 按工具重复率分低/高重复场景，分析不同负载下的收益差异 |
| 正确性验证 | 通过统一解码策略（Greedy Argmax）保证 PPL Delta = 0，排除实现缺陷 |
| 可扩展性测试 | 设计 5 种上下文长度（32~512 tokens），分析延迟增长曲线 |
| 统计指标体系 | 延迟（per-turn latency）、命中率（hit rate）、PPL、工具准确率、加速比（speedup ratio）全覆盖 |

---

## 四、Python 工程能力（Python Engineering）

| 技能 | 具体体现 |
|------|----------|
| 类型注解（Type Hints） | 全量使用 `Dict`, `List`, `Optional`, `Tuple`, `Any` 类型注解 |
| 数据类（Dataclass） | 使用 `@dataclass` 定义消融配置结构体 `AblationConfig` |
| 装饰器 | `@torch.no_grad()` 正确用于推理函数，避免梯度计算开销 |
| 高精度计时 | `time.perf_counter()` 替代 `time.time()` 实现毫秒级精度计时 |
| 函数式 API 设计 | 模块化函数设计（encode / generate / call_tool），低耦合高内聚 |
| 随机数控制 | `random.seed(42) + np.random.seed(42) + torch.manual_seed(42)` 保证可复现性 |
| collections 模块 | `OrderedDict`（LRU）、`Counter`（频率统计）、`defaultdict`（分组） |
| 哈希与 JSON | `hashlib.md5`、`json.dumps(sort_keys=True)` 实现稳定的缓存键生成 |

---

## 五、机器学习数学基础（ML Math Foundation）

| 技能 | 具体体现 |
|------|----------|
| 余弦相似度 | 使用 `F.cosine_similarity` 对 KV 张量做层级相似度分析 |
| Jaccard 相似度 | Token 集合交并比，量化文本重叠率 |
| 困惑度（PPL） | `exp(cross_entropy_loss)`，用于语言模型质量评估 |
| Transformer 注意力数学 | 理解 KV Cache 为注意力矩阵计算的等价加速，推导 PPL Delta = 0 的理论依据 |
| 延迟建模 | 分析工具延迟（120ms）+ KV 编码延迟 + 解码延迟的组合模型 |

---

## 六、论文写作与学术表达能力（Academic Writing）

| 技能 | 具体体现 |
|------|----------|
| Motivation 构建 | 从量化冗余数据（54.7% KV redundancy）出发，建立协同设计的必要性论证 |
| 实验结论撰写 | 每个实验配备假设（Hypothesis）、方法（Method）、结果（Result）、结论（Conclusion）四段式写作 |
| 图表设计能力 | 为每个实验设计对应图表（横轴/纵轴/类型/对比对象），符合顶会 NeurIPS/OSDI 规范 |
| 消融实验写作 | 正确区分"模块贡献分析"与"超参数敏感性分析"，选择关闭而非弱化每个模块 |
| 答辩 Q&A 准备 | 预判 20+ 导师追问，精准回答机制原理、实验局限性与未来工作 |

---

## 七、系统架构设计能力（System Architecture）

| 技能 | 具体体现 |
|------|----------|
| 三层架构设计 | 应用层 / 协同缓存层 / 存储层清晰分离 |
| 缓存协议设计 | Tool Cache Miss → KV Prefetch 的事件驱动协议 |
| 统一调度器 | 跨 KV / Tool 两种缓存的容量分配与淘汰优先级统一管理 |
| 性能瓶颈分析 | 识别"Tool 命中但 KV 未命中"为独立方案的核心结构性缺陷 |
| 可扩展性设计 | 抽象 `LRUCache` / `LFUCache` 接口，支持容量配置与策略切换 |

---

## 八、工具链与开发环境（Toolchain）

| 工具 | 用途 |
|------|------|
| PyTorch 2.10 | 模型推理、张量操作、梯度控制 |
| HuggingFace Transformers 4.57 | 模型加载、tokenizer、past_key_values API |
| NumPy 2.4 | 统计计算（均值、百分比、增长率） |
| Python 3.13 | 主语言，dataclass、collections、hashlib、json |
| macOS CPU 环境 | 无 GPU 依赖，降低实验门槛，证明轻量可复现性 |
| JetBrains PyCharm | 开发 IDE，调试与运行 |
| Markdown | 实验文档、答辩材料、README 撰写 |

---

*Skills document v1.0 | 2026-04-22*

