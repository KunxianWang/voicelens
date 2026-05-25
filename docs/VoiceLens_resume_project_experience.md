# 项目经历

---

<table style="width:100%; border-collapse:collapse;">
  <tr>
    <td style="width:70%;"><strong>VoiceLens — 跨境电商 VoC 数据工程与智能分析平台</strong></td>
    <td style="width:30%; text-align:right;"><strong>2026 年 1 月 — 2026 年 5 月</strong></td>
  </tr>
</table>

**项目技术栈：** Python 3.11、Prefect 3、PostgreSQL、SQLAlchemy、Qdrant、Pydantic、Claude API、Sentence-Transformers、BM25、RRF Hybrid Search、scikit-learn、TF-IDF + KMeans、LangGraph、Streamlit、Docker、pytest、ruff。

**项目描述：** VoiceLens 是一个面向跨境电商消费电子品牌的客户之声分析平台，围绕 Amazon Reviews 2023 Electronics 数据集，构建从原始评论采集、数据质量治理、结构化入库、ABSA 细粒度情感抽取、向量索引、混合检索、负面问题聚类、异常检测到 Dashboard 与 Agent 问答的端到端系统。项目以 PostgreSQL 作为结构化事实库，以 Qdrant 作为检索索引，结合 LLM 结构化抽取、BM25 + Dense Hybrid Search、TF-IDF + KMeans 聚类和 EWMA z-score 异常检测，实现对用户负面反馈、产品质量问题和新兴风险事件的可追溯分析。

**项目亮点：**

- **负责端到端数据工程链路设计与实现：** 基于 Python + Prefect 构建 `ingest -> DQ -> normalize -> ABSA -> embed -> cluster -> anomaly` 离线处理流程，完成 Amazon Reviews JSONL/GZ 文件流式解析、metadata brand map 构建、品牌别名归一、字段类型转换、文本清洗、质量校验与 Postgres 入库；累计扫描 **43.9M** review rows 与 **1.6M** metadata rows，构建 **253.5k** 条确定性 MVP 评论子集，最终加载 **245,961** 条有效评论，DQ 通过率约 **97.03%**。

- **设计结构化数据模型与可重跑处理状态机制：** 使用 SQLAlchemy 建模 `Brand / Sku / Review / IngestRun / DQEvent / AspectOntology / AspectMention / ABSAReviewStatus / Cluster / ReviewCluster / Incident` 等核心表，将原始评论事实、模型抽取事实、处理状态和分析结果分层存储；通过 unique constraint、provider / model / aspect_version scope 和 review-level status 支持幂等重跑、模型对比与 LLM 成本控制。

- **构建可治理的 ABSA 细粒度情感抽取模块：** 设计版本化 Aspect Ontology，将评论抽取为 `aspect / sentiment / severity / evidence_quote` 结构化事实；基于 Pydantic schema 与 validator 对 LLM 输出进行强校验，限制 aspect code、sentiment、severity 取值，并要求 evidence quote 必须为原文片段，避免模型生成不可追溯证据；完成 **1,000** 条真实 Claude ABSA batch 验证，holdout eval 达到 **macro-F1 0.7074、precision 0.8409、recall 0.8043、Cohen's κ 0.9336、evidence-verbatim rate 1.0**。

- **实现 Qdrant 向量索引与混合检索系统：** 将 review 原文与 ABSA 上下文拼接为 embedding text，构建包含 brand、asin、rating、posted_at、aspect、sentiment、severity、evidence quote 的 Qdrant payload；支持 Dense Search、BM25 Lexical Search 与 RRF Hybrid Search，兼顾语义相似召回和 SKU / brand / aspect 等关键词精确匹配，在 refined golden set 上实现 **Hit@5 0.959、Recall@20 0.886、MRR@10 0.849、filter precision@10 1.00**。

- **设计负面问题聚类与新兴异常检测能力：** 基于 negative aspect mentions 构建 `ClusterRecord`，按 aspect 分组后使用 TF-IDF + KMeans 聚类，将零散负面评论沉淀为可命名 issue cluster，并结合 severity weight 计算问题影响度，最终生成 **44** 个 issue clusters；进一步基于 weekly volume、EWMA baseline、rolling std 与 z-score 检测异常增长，生成 **23** 个 emerging incidents，用于定位 SKU 或问题维度上的潜在质量风险。

- **构建 Citation-grounded RAG 问答链路：** 实现 `retrieve -> build citations -> generate answer` 的证据约束型问答流程，将检索命中的 review evidence 转换为 citation object，并在答案生成后过滤 unsupported citation marker，避免模型引用不存在的证据；当检索结果不足时返回 insufficient evidence，提高业务问答的可审计性和可信度。

- **实现受控 LangGraph Agent 工作流：** 基于 LangGraph 构建 deterministic router，将用户问题路由到 `retrieval_answer`、`analytics_summary`、`incident_summary` 或 `insufficient_scope` 四类工具节点，形成 `route -> one tool -> format -> END` 的单轮工作流；相比开放式 autonomous agent，该设计更容易测试、解释和控制，适合 MVP 阶段的业务分析问答场景。

- **开发 Streamlit 可视化 Dashboard：** 搭建 Overview、ABSA Distribution、Issue Clusters、Emerging Incidents、Retrieval Search、Agent Q&A、Data Quality 等页面，支持品牌、aspect、sentiment、severity、rating、date 等维度筛选；通过 `ui/db.py` 封装 SQL 聚合查询，将 Postgres 事实表、Qdrant 检索结果、cluster 与 incident 分析结果统一展示给业务用户。

- **建立评测与工程质量保障体系：** 针对 ingest、DQ、ABSA、retrieval、clustering、anomaly、RAG、agent、UI database helpers 等模块编写自动化测试，使用 pytest + SQLite fixture + mock provider 覆盖核心行为；同时提供 ABSA eval、retrieval eval、agent eval、golden set 构建与调参脚本，并通过 ruff 约束代码风格，保证离线 pipeline 和用户侧功能的稳定性。
