# VoiceLens 五层函数数据模拟执行

> 配套阅读文档：`docs/VoiceLens_five_layer_code_walkthrough.md`
>
> 上一份文档解释“每一层代码在做什么”。这份文档用一组具体数据，按上一份文档的讲解顺序，把关键函数从输入到输出模拟跑一遍。阅读目标是看清：
>
> 1. 一个函数收到的数据长什么样。
> 2. 它改变了什么。
> 3. 它的输出为什么能成为下一个函数的输入。

---

## 0. 模拟约定

## 0.1 这不是测试日志

本文是按当前代码行为写出的手工执行轨迹，不是某次终端日志逐字拷贝。下面几类值会用“示意值”：

- SHA1 摘要。
- UUIDv5 point id。
- embedding vector 的浮点数。
- SQLAlchemy 自动生成的主键。
- cluster id、incident id。

这些值的具体数字每次环境可能不同，但数据形状和函数间关系不变。

## 0.2 为什么不能只用一条评论

本文有一个主角 review：

```text
R501: "Anker power bank stopped working after two weeks. It became too hot to touch, so I returned it."
```

但以下函数本来就需要一组数据：

- DQ 的 duplicate 与 rejected 分支。
- BM25 与 hybrid retrieval 的排序比较。
- KMeans clustering 的最小样本量。
- anomaly detection 的历史周窗口。

所以本文会围绕主角 review 加少量配套数据。主角始终是 `R501`，其他 review 只是为了让函数分支能看得见。

## 0.3 本文使用的原始数据

### 0.3.1 Metadata 原始行

Amazon metadata 文件里有一行：

```json
{
  "parent_asin": "B0ANKPB001",
  "store": "Anker Innovations",
  "title": "Anker 20K Power Bank",
  "details": {
    "Brand": "Anker"
  }
}
```

### 0.3.2 主 review 原始行

Amazon review 文件里有一行：

```json
{
  "review_id": "raw-r501",
  "parent_asin": "B0ANKPB001",
  "rating": 1.0,
  "verified_purchase": "true",
  "timestamp": 1714521600000,
  "helpful_vote": 8,
  "title": "Stopped working",
  "text": "Anker power bank stopped working after two weeks. It became too hot to touch, so I returned it."
}
```

### 0.3.3 用于 DQ 的配套原始行

| 行 | 目的 | 数据特点 |
|---|---|---|
| `raw-r501` | 主样例 | 合法。 |
| `raw-r501-dup` | duplicate 分支 | 与 `raw-r501` 在 DQ fingerprint 上相同。 |
| `raw-r-short` | short text 分支 | 文本只有 `"bad"`。 |
| `raw-r-rating` | invalid rating 分支 | rating 是 `6`。 |

### 0.3.4 用于 retrieval 的配套 review

后续假设 Postgres 和 Qdrant 中还有两条已处理 review：

| review id | text | ABSA 重点 |
|---|---|---|
| `R502` | `Battery lasted all weekend and the power bank stayed cool.` | `battery positive` |
| `R503` | `This Bose speaker keeps disconnecting over bluetooth.` | `bluetooth negative` |

### 0.3.5 用于 clustering 的配套 negative mentions

为了让 `min_cluster_size=5` 的 reliability cluster 成立，假设同批 ABSA 还有这些 negative mentions：

| review id | aspect | severity | evidence quote |
|---|---|---|---|
| `R501` | `reliability` | `high` | `stopped working after two weeks` |
| `R511` | `reliability` | `medium` | `died after a month` |
| `R512` | `reliability` | `medium` | `stopped working in week three` |
| `R513` | `reliability` | `high` | `completely dead after two uses` |
| `R514` | `reliability` | `low` | `not reliable after a short time` |
| `R515` | `reliability` | `medium` | `failed after fourteen days` |

### 0.3.6 用于 anomaly 的周序列

假设这些 reliability clustered mentions 按周聚合后形成一条 series：

| week start | observed volume | 说明 |
|---|---:|---|
| `2024-04-01` | 2 | 历史周 1 |
| `2024-04-08` | 2 | 历史周 2 |
| `2024-04-15` | 3 | 历史周 3 |
| `2024-04-22` | 9 | 当前 spike week，包含 `R501` |

---

# 1. 公共代码和数据库模型先落地

## 1.1 `config.py` 如何给函数准备默认值

假设 `.env` 没有覆盖，当前运行路径中会得到类似配置：

| 配置 | 示例值 | 谁会用 |
|---|---|---|
| `DATABASE_URL` | `postgresql+psycopg2://.../voicelens` | `get_session()` |
| `QDRANT_URL` | `http://localhost:6333` | retrieval 与 dashboard |
| `QDRANT_COLLECTION` | `reviews_v2` | embed/search |
| `BRAND_ALLOWLIST` | `("Anker", "Soundcore", "Bose", ...)` | Amazon adapter、agent brand filter |
| `DQ_MIN_TEXT_LEN` | `20` | `_check_row()` |
| `DQ_MIN_LANG_CONFIDENCE` | `0.7` | `_check_row()` |

配置层不改变 review 数据，但它决定很多函数的默认行为。例如 `_check_row()` 会使用 `DQ_MIN_TEXT_LEN=20` 判断 `"bad"` 太短。

## 1.2 ORM 模型在模拟数据里长什么样

当主 review 完成 ingest 后，数据库会逐步出现这些结构化对象：

```text
Brand(id=1, name="Anker")
Sku(id=11, brand_id=1, asin="B0ANKPB001")
IngestRun(id=90, source="amazon_reviews_2023_mvp_subset", status="completed")
Review(
    id=501,
    source="amazon_reviews_2023",
    source_id="raw-r501",
    sku_id=11,
    rating=1,
    text_raw="Stopped working. Anker power bank stopped working after two weeks. It became too hot to touch, so I returned it.",
    ingest_run_id=90
)
```

后续层继续加：

```text
AspectMention(...)
ABSAReviewStatus(...)
Cluster(...)
ReviewCluster(...)
Incident(...)
```

你可以把 `models.py` 理解成“函数模拟的总账本”。每层函数不是在空气里运行，而是在这些表之间转数据。

---

# 2. 采集与数据质量层模拟

本节顺序对齐上一份导读的第 1 层。

## 2.1 `amazon_reviews_2023.py`

## 2.1.1 `_open_text(path)`

### 输入

```python
path = Path("data/meta_Electronics.jsonl.gz")
```

### 模拟执行

函数看后缀：

```text
path endswith ".gz" -> use gzip.open(..., "rt", encoding="utf-8")
```

### 输出

返回一个文本流对象，下一步 `_stream_jsonl()` 从这个流一行一行读 JSON。

---

## 2.1.2 `_stream_jsonl(path)`

### 输入

文件中的一行：

```json
{"parent_asin":"B0ANKPB001","store":"Anker Innovations","details":{"Brand":"Anker"}}
```

### 模拟执行

```python
line = line.strip()
row = json.loads(line)
yield row
```

### 输出

```python
{
    "parent_asin": "B0ANKPB001",
    "store": "Anker Innovations",
    "details": {"Brand": "Anker"}
}
```

如果某行 JSON 坏了，函数会 skip 并写 warning，后面的合法行仍继续 yield。

---

## 2.1.3 `canonicalize_brand(raw_brand, allowlist)`

### 输入

```python
raw_brand = "Anker Innovations"
allowlist = ("Anker", "Soundcore", "Bose", "JBL", "UGREEN")
```

### 模拟执行

函数将 `raw_brand` 转成 lower-case，再与 alias regex 匹配：

```text
"anker innovations" matches alias for "Anker"
```

### 输出

```python
"Anker"
```

这一步把来源侧的品牌别名归一到项目内部维度值。

---

## 2.1.4 `load_asin_brand_map(metadata_path, allowlist)`

### 输入

- metadata generator 产出的 metadata row。
- allowlist。

### 内部调用轨迹

```text
_stream_jsonl(metadata_path)
  -> metadata row
match_candidate_brands(...)
choose_best_brand(...)
  -> "Anker"
for key in ("parent_asin", "asin")
  -> take "B0ANKPB001"
```

### 输出

```python
{
    "B0ANKPB001": "Anker"
}
```

后续 `_normalize_row()` 会用这个 map 给 review 补品牌。

---

## 2.1.5 `_normalize_row(raw, asin_to_brand, allowlist)`

### 输入

```python
raw = {
    "review_id": "raw-r501",
    "parent_asin": "B0ANKPB001",
    "rating": 1.0,
    "verified_purchase": "true",
    "timestamp": 1714521600000,
    "helpful_vote": 8,
    "title": "Stopped working",
    "text": "Anker power bank stopped working after two weeks. It became too hot to touch, so I returned it."
}
asin_to_brand = {"B0ANKPB001": "Anker"}
allowlist = ("Anker", "Bose", "JBL")
```

### 关键变化

| 原始字段 | 函数处理 | 内部字段 |
|---|---|---|
| `parent_asin` | 作为 fallback ASIN | `asin="B0ANKPB001"` |
| `rating=1.0` | float -> rounded int | `rating=1` |
| `verified_purchase="true"` | string -> bool | `verified=True` |
| unix ms timestamp | 转 ISO datetime string | `posted_at="2024-05-01T00:00:00"` |
| `title` + `text` | 拼接 | `text_raw="Stopped working. ..."` |
| ASIN map | 查品牌 | `brand="Anker"` |

### 输出

```python
normalized_r501 = {
    "source": "amazon_reviews_2023",
    "source_id": "raw-r501",
    "asin": "B0ANKPB001",
    "brand": "Anker",
    "model_number": None,
    "category": None,
    "rating": 1,
    "verified": True,
    "posted_at": "2024-05-01T00:00:00",
    "helpful_count": 8,
    "text_raw": (
        "Stopped working. Anker power bank stopped working after two weeks. "
        "It became too hot to touch, so I returned it."
    ),
    "language": "en",
    "lang_confidence": 0.95,
    "locale": "en-US",
}
```

注意：这里还没有判定文本是否质量合格，质量门在 `run_dq()`。

---

## 2.1.6 `iter_amazon_reviews(path, asin_to_brand, brand_allowlist, limit)`

### 输入

```python
iter_amazon_reviews(
    Path("data/Electronics.jsonl.gz"),
    asin_to_brand={"B0ANKPB001": "Anker"},
    brand_allowlist=("Anker",),
    limit=4,
)
```

### 模拟 yield

| 原始行 | `_normalize_row()` 输出 | allowlist 结果 |
|---|---|---|
| `raw-r501` | `normalized_r501` | yield |
| `raw-r501-dup` | normalized duplicate | yield，DQ 稍后识别重复 |
| `raw-r-short` | short-text row | yield，DQ 稍后识别短文本 |
| unknown brand row | `brand=None` | 这里就 skip |

### 输出

一个 generator，调用方把它转成 list 后得到：

```python
matched = [normalized_r501, normalized_r501_dup, normalized_short, normalized_bad_rating]
```

adapter 只做来源解析与品牌范围过滤，ASIN/rating/text 是否合格交给 DQ。

---

## 2.2 `dq.py`

## 2.2.1 `DQOutcome`

先看最终 shape。假设 `run_dq(matched)` 完成后，会构造：

```python
DQOutcome(
    valid_rows=[normalized_r501],
    rejected={
        "duplicate_review_id": [normalized_r501_dup],
        "text_too_short": [normalized_short],
        "invalid_rating": [normalized_bad_rating],
    },
    per_check_failed=Counter({
        "duplicate_review_id": 1,
        "text_too_short": 1,
        "invalid_rating": 1,
    }),
    total_rows=4,
)
```

属性调用效果：

```python
outcome.passed_rows  # 1
outcome.failed_rows  # 3
outcome.pass_rate    # 0.25
```

---

## 2.2.2 `_row_dedup_key(row)`

### 输入

```python
row = normalized_r501
```

### 模拟执行

函数拼 fingerprint：

```text
amazon_reviews_2023
| B0ANKPB001
| raw-r501
| 2024-05-01T00:00:00
| sha1(text_raw)
```

### 输出

```python
"sha1:9c1e...-示意值"
```

如果 duplicate row 的这些字段一致，它的 dedup key 也一致，于是 `run_dq()` 会把第二条放入 `rejected["duplicate_review_id"]`。

---

## 2.2.3 `_check_row(row)`

### 合法输入

```python
_check_row(normalized_r501)
```

### 输出

```python
None
```

含义是这一行通过 row-local checks。

### 短文本输入

```python
normalized_short["text_raw"] = "bad"
_check_row(normalized_short)
```

### 输出

```python
"text_too_short"
```

### rating 非法输入

```python
normalized_bad_rating["rating"] = 6
_check_row(normalized_bad_rating)
```

### 输出

```python
"invalid_rating"
```

---

## 2.2.4 `run_dq(rows)`

### 输入

```python
rows = [
    normalized_r501,
    normalized_r501_dup,
    normalized_short,
    normalized_bad_rating,
]
```

### 执行轨迹

```text
row 1 -> dedup key unseen -> _check_row -> None -> valid_rows
row 2 -> dedup key already seen -> rejected["duplicate_review_id"]
row 3 -> dedup key unseen -> _check_row -> "text_too_short"
row 4 -> dedup key unseen -> _check_row -> "invalid_rating"
```

### 输出

就是前面展示的 `DQOutcome`。

`valid_rows=[normalized_r501]` 会进入数据库写入；rejected rows 会被 `write_dq_events()` 汇总为质量指标。

---

## 2.3 `normalize.py`

## 2.3.1 `clean_text(text)`

### 输入

```python
text = "Stopped working.\n\nAnker power bank   stopped working after two weeks."
```

### 输出

```python
"Stopped working. Anker power bank stopped working after two weeks."
```

它让入库文本更稳定，也让后续 embedding text 与 UI snippet 少一些格式噪音。

---

## 2.3.2 `parse_posted_at(value)`

### 输入

```python
"2024-05-01T00:00:00"
```

### 输出

```python
datetime(2024, 5, 1, 0, 0, 0)
```

字符串在入 `Review.posted_at` 前被转成 Python datetime。

---

## 2.3.3 `get_or_create_brand(session, "Anker")`

### 第一次输入

数据库还没有 `Anker`。

### 输出

```text
Brand(id=1, name="Anker")
```

### 第二次输入

数据库已有 `Anker`。

### 输出

返回已有 brand，不新增一行。

---

## 2.3.4 `get_or_create_sku(session, brand, asin, ...)`

### 输入

```python
brand = Brand(id=1, name="Anker")
asin = "B0ANKPB001"
```

### 输出

```text
Sku(id=11, brand_id=1, asin="B0ANKPB001")
```

`Review` 之后存 `sku_id=11`，不会在每条 review 里重复存一大份 SKU 信息。

---

## 2.3.5 `load_reviews(session, rows, ingest_run_id)`

### 输入

```python
rows = [normalized_r501]
ingest_run_id = 90
```

### 内部执行

```text
get_or_create_brand -> Brand(id=1)
get_or_create_sku -> Sku(id=11)
select Review by (source, source_id)
  -> not found
clean_text(text_raw)
parse_posted_at(posted_at)
insert Review
```

### 输出

```python
inserted = 1
```

数据库新增：

```text
Review(id=501, source_id="raw-r501", sku_id=11, rating=1, ingest_run_id=90)
```

如果下一次仍传同一 `source` 和 `source_id`，查询会命中已有 review，输出 `inserted=0`。

---

## 2.4 `ingest_flow.py`

## 2.4.1 `read_jsonl(path)`

在 sample path 上，函数会把 JSONL 变成 list：

```python
[
    normalized_r501,
    normalized_short,
]
```

这个函数用于 sample ingest。真实 Amazon 数据 path 则走 `iter_amazon_reviews()`。

---

## 2.4.2 `open_ingest_run(session, source)`

### 输入

```python
source = "amazon_reviews_2023_mvp_subset"
```

### 输出

```text
IngestRun(id=90, source="amazon_reviews_2023_mvp_subset", status="running")
```

---

## 2.4.3 `write_dq_events(session, run_id, outcome)`

### 输入

```python
run_id = 90
outcome.per_check_failed = {
    "duplicate_review_id": 1,
    "text_too_short": 1,
    "invalid_rating": 1,
}
outcome.total_rows = 4
```

### 输出到数据库

它会对 `ALL_CHECKS` 中每个 check 写一行 `DQEvent`。其中几行示意：

```text
DQEvent(run_id=90, check_name="duplicate_review_id", n_rows=4, n_failed=1)
DQEvent(run_id=90, check_name="text_too_short", n_rows=4, n_failed=1)
DQEvent(run_id=90, check_name="invalid_rating", n_rows=4, n_failed=1)
DQEvent(run_id=90, check_name="missing_asin", n_rows=4, n_failed=0)
```

---

## 2.4.4 `close_ingest_run(session, run, n_rows, status)`

### 输入

```python
run = IngestRun(id=90, status="running")
n_rows = 1
status = "completed"
```

### 输出到数据库

```text
IngestRun(id=90, n_rows=1, completed_at=<utc time>, status="completed")
```

---

## 2.4.5 `ingest_flow(...)`

sample flow 将以上函数串成：

```text
read_jsonl
-> run_dq
-> open_ingest_run
-> load_reviews
-> write_dq_events
-> close_ingest_run
```

### 模拟 summary

```python
{
    "total_rows": 4,
    "passed_rows": 1,
    "failed_rows": 3,
    "dq_pass_rate": 0.25,
    "loaded_reviews": 1,
    "per_check_failed": {
        "duplicate_review_id": 1,
        "text_too_short": 1,
        "invalid_rating": 1
    }
}
```

---

## 2.5 `amazon_ingest_flow.py`

## 2.5.1 `load_brands_file(path)`

### 输入

`data/resolved_brand_allowlist.json`：

```json
{"brands": ["Anker", "Bose", "JBL"]}
```

### 输出

```python
("Anker", "Bose", "JBL")
```

---

## 2.5.2 `_count_lines_cheap(path)`

### 输入

review 文件含 4 条非空 JSONL。

### 输出

```python
4
```

这是 raw input rows 统计，不等于最后 matched rows 或 loaded reviews。

---

## 2.5.3 `amazon_ingest_flow(...)`

### 模拟输入

```python
amazon_ingest_flow(
    input_path="data/Electronics.jsonl.gz",
    metadata_path="data/meta_Electronics.jsonl.gz",
    brands=("Anker",),
    limit=4,
    source="amazon_reviews_2023_mvp_subset",
)
```

### 执行轨迹

```text
load_asin_brand_map
-> iter_amazon_reviews
-> run_dq
-> open_ingest_run
-> load_reviews
-> write_dq_events
-> close_ingest_run
-> count valid brand and ASIN distribution
```

### 输出

```python
{
    "input_rows": 4,
    "matched_rows": 4,
    "passed_rows": 1,
    "failed_rows": 3,
    "loaded_reviews": 1,
    "dq_pass_rate": 0.25,
    "per_check_failed": {
        "duplicate_review_id": 1,
        "text_too_short": 1,
        "invalid_rating": 1
    },
    "brand_counts": {"Anker": 1},
    "top_asin_counts": {"B0ANKPB001": 1}
}
```

到这里，主 review 已从 Amazon raw row 变成了数据库中的 `Review(id=501)`。

---

# 3. ABSA 层模拟

## 3.1 `schema.py`

## 3.1.1 `ontology_codes(version)`

### 输入

```python
ontology_codes("v2")
```

### 输出

```python
(
    "battery",
    "charging",
    "overheating",
    "sound_quality",
    "bluetooth",
    "delivery",
    "price",
    "reliability",
)
```

这组 code 会传给 validator，决定 provider 输出里哪些 aspect 合法。

---

## 3.1.2 `AspectMentionOut`

### 合法输入

```python
AspectMentionOut(
    aspect_code="reliability",
    sentiment="negative",
    severity="high",
    evidence_quote="stopped working after two weeks",
)
```

### 输出

一个通过 Pydantic 校验的 mention object。

### 非法输入

```python
AspectMentionOut(
    aspect_code="reliability",
    sentiment="negative",
    severity=None,
    evidence_quote="stopped working after two weeks",
)
```

### 输出

Pydantic validation error，因为 negative sentiment 必须带 severity。

---

## 3.1.3 `ABSAOutput`

### 输入

```python
ABSAOutput(
    aspects=[
        AspectMentionOut(... reliability ...),
        AspectMentionOut(... overheating ...),
    ]
)
```

### 输出

一个顶层 ABSA result object，后续 validator 会遍历 `raw.aspects`。

---

## 3.2 `providers.py`

## 3.2.1 `provider.extract(review_text)`

为了演示多 aspect，本文模拟真实 provider 对 `R501` 返回：

```python
raw_absa = ABSAOutput(
    aspects=[
        AspectMentionOut(
            aspect_code="reliability",
            sentiment="negative",
            severity="high",
            evidence_quote="stopped working after two weeks",
        ),
        AspectMentionOut(
            aspect_code="overheating",
            sentiment="negative",
            severity="high",
            evidence_quote="too hot to touch",
        ),
    ]
)
```

如果用 `MockABSAProvider`，输出仍是同一 contract：`ABSAOutput`。Mock 与真实 LLM 的差别在“如何得到这些 mentions”，不是 flow 后面的数据 shape。

---

## 3.2.2 `get_provider(name, model=...)`

### 输入

```python
get_provider("mock")
```

### 输出

```text
MockABSAProvider(provider="mock", model_name="mock")
```

真实跑 LLM 时则可能是：

```python
get_provider("anthropic", model="claude-opus-4.6")
```

后续 `absa_flow()` 只需要 provider contract，不需要知道底层是 mock 还是 HTTP API。

---

## 3.3 `validators.py`

## 3.3.1 `validate_absa_output(review_text, output, ontology_codes)`

### 输入

```python
review_text = Review(id=501).text_raw
output = raw_absa
allowed = ontology_codes("v2")
```

### 执行检查

| mention | aspect 合法 | quote 是 substring | duplicate | 结果 |
|---|---|---|---|---|
| `reliability` | 是 | 是 | 否 | valid |
| `overheating` | 是 | 是 | 否 | valid |

### 输出

```python
ValidationResult(
    valid_mentions=[
        <reliability negative high>,
        <overheating negative high>,
    ],
    errors=[],
)
```

### 再加一个坏 mention

如果 provider 多返回：

```python
{
    "aspect_code": "safety",
    "sentiment": "negative",
    "severity": "high",
    "evidence_quote": "dangerous fire risk not in original text"
}
```

validator 会至少拒绝：

- aspect code 不在 ontology。
- quote 不是原文 substring。

它不会让这个坏 mention 进入 `AspectMention` 表。

---

## 3.4 `extractor.py`

## 3.4.1 `extract_for_review(provider, review_id, review_text, ontology_codes)`

### 输入

```python
extract_for_review(
    provider=mock_or_llm_provider,
    review_id=501,
    review_text=Review(id=501).text_raw,
    ontology_codes=ontology_codes("v2"),
)
```

### 内部调用

```text
provider.extract(review_text)
-> raw_absa
validate_absa_output(review_text, raw_absa, ontology_codes)
-> ValidationResult
```

### 输出

```python
ExtractionOutcome(
    review_id=501,
    result=ValidationResult(
        valid_mentions=[reliability_mention, overheating_mention],
        errors=[],
    ),
    raw_aspect_count=2,
)
```

flow 后面只消费已经验证过的 `valid_mentions`。

---

## 3.5 `absa_flow.py`

## 3.5.1 `_load_ontology(session, "v2")`

### 数据库输入

`aspect_ontology` 已 seeded：

```text
AspectOntology(id=201, version="v2", code="reliability")
AspectOntology(id=202, version="v2", code="overheating")
...
```

### 输出

```python
{
    "reliability": 201,
    "overheating": 202,
    "...": ...
}
```

这个 map 让 flow 能把 code 转成 `AspectMention.aspect_id` foreign key。

---

## 3.5.2 `_select_review_ids(...)`

### 输入

```python
_select_review_ids(
    session,
    source="amazon_reviews_2023_mvp_subset",
    aspect_version="v2",
    provider_name="mock",
    model_name="mock",
    limit=100,
    force=False,
)
```

### 当前数据库

`Review(id=501)` 存在，但还没有：

```text
ABSAReviewStatus(review_id=501, aspect_version="v2", provider="mock", model_name="mock")
```

### 输出

```python
[501]
```

如果 status 已存在且 `force=False`，输出中会跳过 `501`。

---

## 3.5.3 `_delete_existing(...)`

### 使用场景

只有 `force=True` 才走。

### 假设已有数据

```text
AspectMention(review_id=501, aspect_version="v2", model_name="mock")
ABSAReviewStatus(review_id=501, aspect_version="v2", provider="mock", model_name="mock")
```

### 输出

```python
(n_status_deleted=1, n_mentions_deleted=2)
```

然后 flow 才重新调用 provider。

---

## 3.5.4 `_summary_skeleton()`

### 输出初始 summary

```python
{
    "input_reviews": 0,
    "processed_reviews": 0,
    "reviews_with_mentions": 0,
    "reviews_no_mentions": 0,
    "invalid_reviews": 0,
    "failed_reviews": 0,
    "inserted_mentions": 0,
    "validation_errors": {...},
    "guardrail_triggered": False,
    "guardrail_reason": None,
}
```

后续每处理一条 review，summary 被累加。

---

## 3.5.5 `_classify_review_status(...)`

### 输入

```python
n_valid = 2
n_errors = 0
raw_aspect_count = 2
```

### 输出

```python
"success"
```

对比：

| 输入状态 | 输出 |
|---|---|
| `n_valid=0, n_errors=0, raw_aspect_count=0` | `no_mentions` |
| `n_valid=0, n_errors>0, raw_aspect_count>0` | `invalid` |

---

## 3.5.6 `_record_status(...)`

### 输入

```python
_record_status(
    session,
    review_id=501,
    aspect_version="v2",
    provider_name="mock",
    model_name="mock",
    status="success",
    n_mentions=2,
    n_errors=0,
    error_codes=None,
)
```

### 数据库输出

```text
ABSAReviewStatus(
    review_id=501,
    aspect_version="v2",
    provider="mock",
    model_name="mock",
    status="success",
    n_mentions=2,
    n_errors=0
)
```

---

## 3.5.7 `_guardrail_reason()` 和 `_check_guardrails()`

### 正常输入

```python
summary["processed_reviews"] = 1
estimated_cost_usd = 0.0004
max_cost_usd = 1.0
min_processed_for_rate_guardrail = 50
```

### 输出

```python
None
```

rate guardrail 还没到最小样本量，cost 也没超。

### 超预算输入

```python
estimated_cost_usd = 1.25
max_cost_usd = 1.0
```

### 输出

```python
"estimated_cost_usd 1.2500 exceeded max_cost_usd 1.0000"
```

`_check_guardrails()` 会把这个 reason 写入 summary，并在 `stop_on_guardrail=True` 时停止 batch。

---

## 3.5.8 `absa_flow(...)`

### 输入

```python
absa_flow(
    source="amazon_reviews_2023_mvp_subset",
    limit=100,
    provider="mock",
    aspect_version="v2",
)
```

### 主执行轨迹

```text
get_provider
-> _load_ontology
-> _select_review_ids
-> extract_for_review(R501)
-> insert AspectMention rows
-> _classify_review_status
-> _record_status
-> _check_guardrails
```

### 数据库新增

```text
AspectMention(
    review_id=501,
    aspect_id=201,
    sentiment="negative",
    severity="high",
    evidence_quote="stopped working after two weeks",
    aspect_version="v2",
    model_name="mock"
)

AspectMention(
    review_id=501,
    aspect_id=202,
    sentiment="negative",
    severity="high",
    evidence_quote="too hot to touch",
    aspect_version="v2",
    model_name="mock"
)
```

### summary

```python
{
    "input_reviews": 1,
    "processed_reviews": 1,
    "reviews_with_mentions": 1,
    "inserted_mentions": 2,
    "aspect_counts": {
        "reliability": 1,
        "overheating": 1
    },
    "sentiment_counts": {"negative": 2},
    "severity_counts": {"high": 2},
    "provider": "mock",
    "model_name": "mock",
    "aspect_version": "v2",
    "guardrail_triggered": False
}
```

---

# 4. 索引与检索层模拟

## 4.1 `embeddings.py`

## 4.1.1 `get_embedding_provider("mock")`

### 输出

```text
MockEmbeddingProvider(name="mock", dimension=32)
```

---

## 4.1.2 `MockEmbeddingProvider.embed_batch(texts)`

### 输入

```python
texts = [
    "Review: ... R501 text ...\nAspects: reliability negative high; overheating negative high"
]
```

### 输出

```python
[
    [0.11, -0.38, 0.04, "...", 0.07]  # 32-d normalized vector, 示意
]
```

### 对比 `LocalEmbeddingProvider`

如果 provider 是 `local`，同样的输入会通过 sentence-transformers 输出模型维度向量。例如默认 working index 使用的本地模型会输出 384 维向量。

---

## 4.2 `qdrant_index.py`

## 4.2.1 `build_point_id(review_id, aspect_version, provider, model_name)`

### 输入

```python
build_point_id(501, "v2", "mock", "mock")
```

### 输出

```python
"<uuid-v5-for-501-v2-mock-mock>"
```

同一组输入会得到同一个 point id；换成 `provider="anthropic"` 就会变。

---

## 4.2.2 `build_embedding_text(text_raw, mentions)`

### 输入

```python
text_raw = Review(id=501).text_raw
mentions = [
    {
        "aspect_code": "reliability",
        "sentiment": "negative",
        "severity": "high",
        "evidence_quote": "stopped working after two weeks",
    },
    {
        "aspect_code": "overheating",
        "sentiment": "negative",
        "severity": "high",
        "evidence_quote": "too hot to touch",
    },
]
```

### 输出

```text
Review: Stopped working. Anker power bank stopped working after two weeks. It became too hot to touch, so I returned it.
Aspects: reliability negative high; overheating negative high
```

这段字符串才是 embedding provider 真正看到的 document text。

---

## 4.2.3 `build_payload(...)`

### 输入

```python
review = {
    "review_id": 501,
    "source_id": "raw-r501",
    "source": "amazon_reviews_2023",
    "sku_id": 11,
    "asin": "B0ANKPB001",
    "brand": "Anker",
    "rating": 1,
    "verified": True,
    "posted_at": "2024-05-01T00:00:00",
    "text_raw": Review(id=501).text_raw,
}
mentions = [reliability_dict, overheating_dict]
```

### 输出

```python
payload_r501 = {
    "review_id": 501,
    "source_id": "raw-r501",
    "brand": "Anker",
    "asin": "B0ANKPB001",
    "rating": 1,
    "text_raw": "Stopped working. ...",
    "aspect_version": "v2",
    "provider": "mock",
    "model_name": "mock",
    "absa_status": "success",
    "aspect_codes": ["reliability", "overheating"],
    "sentiments": ["negative", "negative"],
    "severities": ["high", "high"],
    "evidence_quotes": [
        "stopped working after two weeks",
        "too hot to touch",
    ],
    "mention_count": 2,
}
```

---

## 4.2.4 `ensure_collection(client, name, vector_size, ...)`

### 第一次输入

```python
ensure_collection(client, "reviews_v2_mock", vector_size=32)
```

### 输出

```python
True
```

含义是 collection 新建成功。

### 再次输入

```python
ensure_collection(client, "reviews_v2_mock", vector_size=32)
```

### 输出

```python
False
```

collection 已存在且维度兼容，不需要新建。

### 维度错误输入

```python
ensure_collection(client, "reviews_v2_mock", vector_size=384)
```

当已有 collection 是 32 维时，这里抛错，避免混写不兼容向量。

---

## 4.2.5 `upsert_points(client, collection, points)`

### 输入

```python
points = [
    PointStruct(
        id="<uuid-v5-for-501-v2-mock-mock>",
        vector=[... 32 floats ...],
        payload=payload_r501,
    )
]
```

### 输出

```python
1
```

Qdrant 里现在有一个可检索 point。

---

## 4.3 `embed_flow.py`

## 4.3.1 `_resolve_review_rows(...)`

### 输入 scope

```python
aspect_version = "v2"
provider = "mock"
model_name = "mock"
statuses = ("success", "no_mentions")
```

### 数据库状态

```text
Review(id=501)
ABSAReviewStatus(review_id=501, status="success")
AspectMention(review_id=501, reliability...)
AspectMention(review_id=501, overheating...)
```

### 输出

```python
reviews = [
    {
        "review_id": 501,
        "source_id": "raw-r501",
        "brand": "Anker",
        "asin": "B0ANKPB001",
        "rating": 1,
        "text_raw": "Stopped working. ...",
        "absa_status": "success",
    }
]

mentions_by_review = {
    501: [
        {"aspect_code": "reliability", ...},
        {"aspect_code": "overheating", ...},
    ]
}

skipped = 0
```

---

## 4.3.2 `_build_points(...)`

### 输入

- `reviews`。
- `mentions_by_review`。
- embedder。

### 内部函数链

```text
build_embedding_text
-> embedder.embed_batch
-> build_payload
-> build_point_id
-> PointStruct
```

### 输出

一个 generator，yield `PointStruct`。

---

## 4.3.3 `embed_flow(...)`

### 输入

```python
embed_flow(
    aspect_version="v2",
    provider="mock",
    model="mock",
    embedding_provider="mock",
    collection="reviews_v2_mock",
    limit=1000,
)
```

### summary

```python
{
    "selected_reviews": 1,
    "embedded_reviews": 1,
    "upserted_points": 1,
    "skipped_failed_or_invalid": 0,
    "collection": "reviews_v2_mock",
    "embedding_provider": "mock",
    "vector_size": 32,
    "aspect_version": "v2",
    "absa_provider": "mock",
    "absa_model": "mock",
}
```

---

## 4.4 `filters.py` 与 dense search

## 4.4.1 `build_search_filter(...)`

### 输入

```python
build_search_filter(
    brand="Anker",
    aspect="reliability",
    sentiment="negative",
    rating_max=2,
)
```

### 输出概念

```text
Filter.must = [
    brand == "Anker",
    aspect_codes contains "reliability",
    sentiments contains "negative",
    rating <= 2,
]
```

它会传入 Qdrant dense query。

---

## 4.4.2 `retrieve(...)`

### 输入

```python
retrieve(
    client=client,
    collection="reviews_v2_mock",
    embedder=MockEmbeddingProvider(),
    query="Anker stopped working too hot",
    limit=3,
    filter_=anker_reliability_negative_filter,
)
```

### 内部执行

```text
embed query
-> Qdrant query_points
-> normalize points to SearchHit
```

### 输出

```python
[
    SearchHit(
        score=0.91,  # 示意
        review_id=501,
        brand="Anker",
        asin="B0ANKPB001",
        rating=1,
        aspect_codes=["reliability", "overheating"],
        sentiments=["negative", "negative"],
        evidence_quotes=[
            "stopped working after two weeks",
            "too hot to touch",
        ],
        text_raw="Stopped working. ...",
    )
]
```

---

## 4.5 `lexical.py`

## 4.5.1 `tokenize(text)`

### 输入

```python
tokenize("Anker stopped working after two weeks!")
```

### 输出

```python
["anker", "stopped", "working", "after", "two", "weeks"]
```

---

## 4.5.2 `_reconstruct_mentions(payload)`

### 输入

```python
payload_r501["aspect_codes"] = ["reliability", "overheating"]
payload_r501["sentiments"] = ["negative", "negative"]
payload_r501["severities"] = ["high", "high"]
```

### 输出

```python
[
    {"aspect_code": "reliability", "sentiment": "negative", "severity": "high"},
    {"aspect_code": "overheating", "sentiment": "negative", "severity": "high"},
]
```

BM25 corpus 重建时会把这些 mention 再送回 `build_embedding_text()`，保证 lexical 与 dense 基于相同 document text。

---

## 4.5.3 `BM25LexicalRetriever.from_qdrant(...)`

### 输入 corpus

Qdrant payloads：

| review id | corpus text 重点 |
|---|---|
| `501` | `stopped working`, `too hot`, `reliability` |
| `502` | `battery lasted`, `stayed cool` |
| `503` | `bluetooth disconnecting`, `Bose` |

### 输出

```text
BM25LexicalRetriever(corpus_size=3)
```

---

## 4.5.4 `rank(query, limit)`

### 输入

```python
bm25.rank("stopped working power bank", limit=3)
```

### 输出示意

```python
[
    LexicalHit(review_id=501, score=4.83, payload=payload_r501),
    LexicalHit(review_id=502, score=0.32, payload=payload_r502),
    LexicalHit(review_id=503, score=0.00, payload=payload_r503),
]
```

---

## 4.5.5 `filter_payloads(...)`

### 输入

```python
filter_payloads(
    [payload_r501, payload_r502, payload_r503],
    brand="Anker",
    aspect="reliability",
    sentiment="negative",
)
```

### 输出

```python
[payload_r501]
```

---

## 4.6 `hybrid.py`

## 4.6.1 `lexical_search(...)`

### 输入

```python
lexical_search(
    bm25=bm25,
    query="stopped working power bank",
    limit=3,
    brand="Anker",
)
```

### 输出

```python
[SearchHit(review_id=501, ...)]
```

BM25 排名先发生，brand/aspect/sentiment 等 predicate 再过滤。

---

## 4.6.2 `hybrid_search(...)`

### 输入

```python
hybrid_search(
    client=client,
    collection="reviews_v2_mock",
    embedder=embedder,
    bm25=bm25,
    query="Anker power bank stopped working and got hot",
    limit=3,
    dense_filter=anker_filter,
    lexical_brand="Anker",
    fusion="rrf_equal",
)
```

### 中间 ranking

```python
dense_ranking   = [501, 502]
lexical_ranking = [501]
```

### RRF 后输出

```python
[
    SearchHit(review_id=501, score=1.0, ...),
    SearchHit(review_id=502, score=0.5, ...),
]
```

`501` 同时被 dense 与 lexical 召回，因此融合后稳定排第一。

---

## 4.7 `rag/retrieve.py`

## 4.7.1 `open_client(url)`

### 输入

```python
open_client(":memory:")
```

### 输出

in-memory Qdrant client，测试路径可用。

### 真实输入

```python
open_client("http://localhost:6333")
```

### 输出

连接本地 Qdrant server 的 client。

---

## 4.7.2 `retrieve_reviews(...)`

### 输入

```python
retrieve_reviews(
    "What are the main reliability complaints for Anker?",
    mode="hybrid",
    brand="Anker",
    aspect="reliability",
    sentiment="negative",
    top_k=3,
)
```

### 内部调用

```text
open_client
-> get_embedding_provider
-> BM25LexicalRetriever.from_qdrant
-> build_search_filter
-> hybrid_search
```

### 输出

```python
[
    SearchHit(review_id=501, ...),
    SearchHit(review_id=511, ...),
    SearchHit(review_id=512, ...),
]
```

---

# 5. 分析层模拟

## 5.1 `clustering.py`

## 5.1.1 `build_clustering_records(...)`

### 数据库输入

范围：

```python
aspect_version="v2"
provider="mock"
model_name="mock"
```

数据：

- `ABSAReviewStatus.status == "success"`。
- `AspectMention.sentiment == "negative"`。
- reliability mentions 有 `R501`、`R511` 到 `R515`。

### 输出

```python
[
    ClusterRecord(
        review_id=501,
        aspect_code="reliability",
        severity="high",
        evidence_quote="stopped working after two weeks",
        text_raw="Stopped working. ...",
        brand="Anker",
        asin="B0ANKPB001",
        rating=1,
        posted_at="2024-04-22T...",
    ),
    ... five more ClusterRecord rows ...
]
```

---

## 5.1.2 `ClusterRecord.cluster_text`

### 输入 record

```python
record.evidence_quote = "stopped working after two weeks"
record.text_raw = "Stopped working. Anker power bank ..."
```

### 输出

```python
"stopped working after two weeks"
```

如果 evidence quote 为空，才 fallback 到 raw review text。

---

## 5.1.3 `severity_weight(severity)`

### 输入输出

| 输入 | 输出 |
|---|---:|
| `"high"` | `3.0` |
| `"medium"` | `2.0` |
| `"low"` | `1.0` |
| `None` | `1.0` |

---

## 5.1.4 `cluster_records(records, min_cluster_size=5, ...)`

### 输入

6 条 reliability records。

### 内部过程

```text
group by aspect_code
-> reliability group of 6 records
vectorize cluster_text with TF-IDF
choose k based on sample size and min_cluster_size
KMeans or single-cluster fallback
extract top keywords
pick representative review ids and quotes
```

### 输出示意

```python
[
    ClusterResult(
        aspect_code="reliability",
        label="stopped working after short use",
        algorithm="tfidf_kmeans",
        keywords=["stopped working", "after", "failed", "dead"],
        members=[R501, R511, R512, R513, R514, R515],
        representative_review_ids=[501, 513, 511, 512, 515],
        representative_quotes=[
            "stopped working after two weeks",
            "completely dead after two uses",
            "died after a month",
        ],
    )
]
```

### 派生属性

```python
result.size  # 6
result.severity_weighted_size
# R501 high 3 + R511 medium 2 + R512 medium 2
# + R513 high 3 + R514 low 1 + R515 medium 2 = 13
```

---

## 5.2 `cluster_flow.py`

## 5.2.1 `_clear_previous_clusters(...)`

### 输入

假设同 scope 之前已有旧 cluster：

```text
Cluster(id=39, aspect_version="v2", provider="mock", model_name="mock")
```

### 输出

```python
1
```

同时删掉旧 `ReviewCluster` memberships。

---

## 5.2.2 `persist_clusters(session, results, ...)`

### 输入

```python
results = [reliability_cluster_result]
run_id = "20240501T000000Z"
```

### 数据库输出

```text
Cluster(
    id=41,
    aspect_code="reliability",
    label="stopped working after short use",
    size=6,
    severity_weighted_size=13.0
)

ReviewCluster(cluster_id=41, review_id=501, aspect_code="reliability", severity="high")
...
ReviewCluster(cluster_id=41, review_id=515, aspect_code="reliability", severity="medium")
```

### 函数输出

```python
{
    "removed_prior_clusters": 1,
    "clusters_written": 1,
    "memberships_written": 6,
}
```

---

## 5.2.3 `cluster_flow(...)`

### 输入

```python
cluster_flow(
    aspect_version="v2",
    provider="mock",
    model="mock",
    min_cluster_size=5,
)
```

### 输出

```python
{
    "negative_mentions": 6,
    "clusters": 1,
    "clustered_mentions": 6,
    "clusters_by_aspect": {"reliability": 1},
    "algorithm": "tfidf_kmeans",
}
```

---

## 5.3 `anomaly.py`

## 5.3.1 `MentionEvent`

`R501` 的 cluster membership 变成事件：

```python
MentionEvent(
    review_id=501,
    cluster_id=41,
    aspect_code="reliability",
    severity="high",
    posted_at=date(2024, 4, 24),
    rating=1,
    evidence_quote="stopped working after two weeks",
)
```

---

## 5.3.2 `week_start(value)`

### 输入

```python
date(2024, 4, 24)  # Wednesday
```

### 输出

```python
date(2024, 4, 22)  # Monday
```

---

## 5.3.3 `aggregate_weekly(events, granularity="cluster")`

### 输入

事件聚合后希望得到这条 series：

```text
cluster:41
  2024-04-01 -> 2 events
  2024-04-08 -> 2 events
  2024-04-15 -> 3 events
  2024-04-22 -> 9 events
```

### 输出

```python
{
    "cluster:41": [
        WeeklyBucket(observed_volume=2, week_start=date(2024, 4, 1), ...),
        WeeklyBucket(observed_volume=2, week_start=date(2024, 4, 8), ...),
        WeeklyBucket(observed_volume=3, week_start=date(2024, 4, 15), ...),
        WeeklyBucket(
            observed_volume=9,
            severity_weighted_volume=18.0,  # 示意
            unique_review_count=9,
            week_start=date(2024, 4, 22),
            example_quotes=["stopped working after two weeks", ...],
        ),
    ]
}
```

---

## 5.3.4 `ewma_baseline(prev_volumes, span=4)`

### 输入

```python
prev_volumes = [2.0, 2.0, 3.0]
span = 4
```

### 计算示意

```text
alpha = 2 / (4 + 1) = 0.4
baseline after first 2 = 2.0
baseline after second 2 = 2.0
baseline after 3 = 0.4 * 3 + 0.6 * 2 = 2.4
```

### 输出

```python
2.4
```

---

## 5.3.5 `rolling_std(prev_volumes, window=8)`

### 输入

```python
[2.0, 2.0, 3.0]
```

### 输出示意

```python
0.5774
```

`z_score()` 里还会和 `min_std=1.0` 比较，因此有效 std 会被 floor 到 `1.0`。

---

## 5.3.6 `z_score(observed, baseline, std, min_std)`

### 输入

```python
observed = 9
baseline = 2.4
std = 0.5774
min_std = 1.0
```

### 输出

```python
(9 - 2.4) / 1.0 = 6.6
```

---

## 5.3.7 `detect_anomalies(buckets_by_series, ...)`

### 输入条件

```python
min_history_weeks = 3
z_threshold = 2.0
min_volume = 3
min_severity_score = 5.0
```

### 当前周判断

| 条件 | R501 所在 spike week |
|---|---|
| 历史周数 >= 3 | 是 |
| observed volume >= 3 | `9`，是 |
| severity score >= 5 | `18`，是 |
| z-score >= 2.0 | `6.6`，是 |

### 输出

```python
[
    AnomalyResult(
        series_key="cluster:41",
        cluster_id=41,
        aspect_code="reliability",
        week_start=date(2024, 4, 22),
        observed_volume=9,
        baseline_volume=2.4,
        z_score=6.6,
        severity_score=18.0,
        ...
    )
]
```

---

## 5.3.8 `build_incident_summary(result, cluster_label)`

### 输入

```python
cluster_label = "stopped working after short use"
result = reliability_spike_result
```

### 输出示意

```text
Reliability complaints in cluster 'stopped working after short use'
spiked to 9 mentions during week 2024-04-22, above EWMA baseline 2.4
(z=6.6, severity score 18). Representative quote:
'stopped working after two weeks'
```

---

## 5.4 `anomaly_flow.py`

## 5.4.1 `_quote_map(session, ...)`

### 输入

数据库里有：

```text
AspectMention(review_id=501, aspect_code="reliability", evidence_quote="stopped working after two weeks")
```

### 输出

```python
{
    (501, "reliability"): "stopped working after two weeks",
    ...
}
```

---

## 5.4.2 `build_mention_events(session, ...)`

### 输入

`ReviewCluster`、`Cluster`、`Review` 与 quote map。

### 输出

```python
events = [
    MentionEvent(review_id=501, cluster_id=41, aspect_code="reliability", ...),
    ...
]
cluster_labels = {
    41: "stopped working after short use"
}
```

---

## 5.4.3 `_clear_incidents(...)`

同一 scope、同一 granularity 的旧 incidents 先删掉，避免重复展示上次结果。

---

## 5.4.4 `upsert_incident(...)`

### 输入

```python
upsert_incident(
    session,
    run_id="20240501T010000Z",
    granularity="cluster",
    aspect_version="v2",
    provider="mock",
    model_name="mock",
    cluster_id=41,
    aspect_code="reliability",
    week_start=date(2024, 4, 22),
    observed_volume=9,
    baseline_volume=2.4,
    z_score=6.6,
    severity_score=18.0,
    summary="Reliability complaints ...",
    ...
)
```

### 输出

```text
Incident(
    id=71,
    cluster_id=41,
    aspect_code="reliability",
    week_start=2024-04-22,
    observed_volume=9,
    z_score=6.6,
    severity_score=18.0
)
```

如果 natural key 已存在，则更新该 incident。

---

## 5.4.5 `anomaly_flow(...)`

### 输入

```python
anomaly_flow(
    aspect_version="v2",
    provider="mock",
    model="mock",
    granularity="cluster",
)
```

### summary

```python
{
    "granularity": "cluster",
    "total_series": 1,
    "total_weeks": 4,
    "incidents_detected": 1,
    "incidents_by_aspect": {"reliability": 1},
    "top_incidents": [
        {
            "aspect_code": "reliability",
            "cluster_id": 41,
            "week_start": "2024-04-22",
            "observed_volume": 9,
            "baseline_volume": 2.4,
            "z_score": 6.6,
            "severity_score": 18.0,
        }
    ]
}
```

---

# 6. RAG、Agent 与 UI 层模拟

## 6.1 `rag/citations.py`

## 6.1.1 `build_citations(hits)`

### 输入

```python
hits = [
    SearchHit(review_id=501, score=1.0, evidence_quotes=["stopped working after two weeks"], ...),
    SearchHit(review_id=511, score=0.5, evidence_quotes=["died after a month"], ...),
]
```

### 输出

```python
[
    Citation(
        citation_id=1,
        review_id=501,
        brand="Anker",
        asin="B0ANKPB001",
        evidence_quote="stopped working after two weeks",
        text_snippet="Stopped working. Anker power bank ...",
        retrieval_score=1.0,
    ),
    Citation(
        citation_id=2,
        review_id=511,
        evidence_quote="died after a month",
        retrieval_score=0.5,
    ),
]
```

---

## 6.2 `rag/providers.py` 与 `rag/answer.py`

## 6.2.1 `MockAnswerProvider.generate(question, citations)`

### 输入

```python
question = "What are the main reliability complaints for Anker?"
citations = [Citation(1, review_id=501, ...), Citation(2, review_id=511, ...)]
```

### 输出示意

```text
Based on 2 retrieved review(s), customers raise points relevant to this
question [1][2]. For example, one review states:
"stopped working after two weeks" [1].
```

---

## 6.2.2 `generate_answer(question, hits, provider="mock")`

### 内部调用

```text
build_citations
-> provider.generate
-> parse citation markers
-> remove unsupported markers
-> build AnswerResult
```

### 输出

```python
AnswerResult(
    question="What are the main reliability complaints for Anker?",
    answer="Based on 2 retrieved review(s) ... [1][2] ... [1].",
    citations=[Citation(1, ...), Citation(2, ...)],
    retrieved_review_ids=[501, 511],
    insufficient_evidence=False,
    guardrail_flags={
        "cited_ids": [1, 2],
        "unsupported_citations": [],
        "uncited_answer": False,
    },
)
```

### unsupported citation 分支

如果某个 provider 原始 answer 写成：

```text
Customers report failures [1] and fires [99].
```

当前 citations 只有 `[1]` 与 `[2]`，则输出 answer 会变成：

```text
Customers report failures [1] and fires .
```

并记录：

```python
guardrail_flags["unsupported_citations"] == [99]
```

---

## 6.3 `agent/router.py`

## 6.3.1 `classify_route(question)`

### 输入与输出

| question | 输出 route |
|---|---|
| `What are the main reliability complaints for Anker?` | `retrieval_answer` |
| `Which issues spiked recently?` | `incident_summary` |
| `Which aspect has the most negative mentions?` | `analytics_summary` |
| `Should I buy Apple stock today?` | `insufficient_scope` |

---

## 6.3.2 `extract_filters(question)`

### 输入

```python
"What are the main negative reliability complaints for Anker?"
```

### 输出

```python
{
    "aspect": "reliability",
    "sentiment": "negative",
    "brand": "Anker",
}
```

---

## 6.4 `agent/state.py`

## 6.4.1 `new_state(question, provider, model)`

### 输入

```python
new_state(
    "What are the main negative reliability complaints for Anker?",
    provider="mock",
    model=None,
)
```

### 输出

```python
{
    "question": "What are the main negative reliability complaints for Anker?",
    "route": "",
    "filters": {},
    "warnings": [],
    "provider": "mock",
    "model_name": "",
}
```

---

## 6.4.2 `result_summary(state)`

LangGraph 跑完后 state 里可能有：

```python
{
    "route": "retrieval_answer",
    "filters": {"brand": "Anker", "aspect": "reliability", "sentiment": "negative"},
    "answer": "Based on ... [1]",
    "citations": [Citation(1, ...)],
    "retrieved_review_ids": [501, 511],
}
```

### 输出

一个 JSON-friendly dict，给 CLI 与 Streamlit 展示。

---

## 6.5 `agent/tools.py`

## 6.5.1 `route_question_node(state)`

### 输入

刚由 `new_state()` 建好的 state。

### 输出 partial update

```python
{
    "route": "retrieval_answer",
    "filters": {
        "aspect": "reliability",
        "sentiment": "negative",
        "brand": "Anker",
    }
}
```

---

## 6.5.2 `_default_retrieve(question, filters, top_k)`

### 输入

```python
question = "What are the main negative reliability complaints for Anker?"
filters = {"aspect": "reliability", "sentiment": "negative", "brand": "Anker"}
top_k = 8
```

### 输出

它调用 `retrieve_reviews(...)`，返回：

```python
[SearchHit(review_id=501, ...), SearchHit(review_id=511, ...), ...]
```

---

## 6.5.3 `retrieval_answer_node(state, config)`

### 输入

state 已有 route 与 filters。

### 内部调用

```text
retrieve_fn or _default_retrieve
-> generate_answer
-> convert AnswerResult into state update
```

### 输出 partial update

```python
{
    "answer": "Based on retrieved review(s) ... [1][2] ...",
    "citations": [Citation(1, ...), Citation(2, ...)],
    "retrieved_results": [SearchHit(501, ...), SearchHit(511, ...)],
    "retrieved_review_ids": [501, 511],
    "warnings": [],
}
```

---

## 6.5.4 `incident_summary_node(state, config)`

### 输入 question

```text
Which issues spiked recently?
```

### 内部数据

`db.get_incidents()` 返回 `Incident(id=71, reliability spike...)`。

### 输出

```python
{
    "incidents_result": [incident_71_dict],
    "answer": (
        "Top 1 emerging-issue incident(s) by severity score:\n"
        "1. [cluster] stopped working after short use - week 2024-04-22: "
        "observed 9 vs baseline 2.4 ..."
    ),
    "warnings": [],
}
```

---

## 6.5.5 `analytics_summary_node(state, config)`

### 输入 question

```text
Which aspect has the most negative mentions?
```

### 模拟 DB 聚合

```python
db.get_aspect_distribution()
# [
#   {"aspect_code": "reliability", "mentions": 6},
#   {"aspect_code": "overheating", "mentions": 1},
# ]
```

### 输出

```python
{
    "analytics_result": {
        "kind": "aspect_distribution",
        "total_mentions": 7,
        ...
    },
    "answer": (
        "Aspect distribution across 7 mentions: reliability=6, overheating=1. "
        "Largest aspect: reliability (6 mentions). "
        "Most negative aspect: reliability (6 negative mentions)."
    ),
    "warnings": [],
}
```

---

## 6.5.6 `insufficient_scope_node(state, config)`

### 输入 question

```text
Should I buy Apple stock today?
```

### 输出

```python
{
    "answer": "<safe out-of-scope message>",
    "citations": [],
    "warnings": ["question routed to insufficient_scope"],
}
```

---

## 6.5.7 `format_response_node(state)`

### 输入

route node 执行后的 state。

### 输出

- 如果已有 answer，则保留。
- 如果没有 answer，则补 `"(no answer was produced)"`。
- 如果 warnings 缺失，则补空 list。

---

## 6.6 `agent/graph.py`

## 6.6.1 `_route_selector(state)`

### 输入

```python
state["route"] = "retrieval_answer"
```

### 输出

```python
"retrieval_answer"
```

LangGraph 用它决定 conditional edge 走哪个 tool node。

---

## 6.6.2 `build_agent_graph(config)`

### 输出结构

```text
START
-> route_question
-> retrieval_answer
-> format_response
-> END
```

对于其他 route，第二步节点会换成 incident、analytics 或 insufficient。

---

## 6.6.3 `run_agent(question, ...)`

### 输入

```python
run_agent(
    "What are the main negative reliability complaints for Anker?",
    provider="mock",
    top_k=3,
)
```

### 输出 final state

```python
{
    "question": "...",
    "route": "retrieval_answer",
    "filters": {
        "aspect": "reliability",
        "sentiment": "negative",
        "brand": "Anker",
    },
    "answer": "Based on ... [1][2] ...",
    "retrieved_review_ids": [501, 511, 512],
    "citations": [Citation(1, ...), Citation(2, ...), Citation(3, ...)],
    "warnings": [],
}
```

---

## 6.7 `ui/db.py`

UI query helper 数量多。下面用当前模拟数据库状态把每个上一份导读列出的函数跑一遍。

假设当前 DB 总体状态：

- brands：`Anker`、`Bose`。
- reviews：`R501`、`R502`、`R503` 加上 reliability cluster 配套 reviews。
- aspects：`reliability`、`overheating`、`battery`、`bluetooth`。
- clusters：`Cluster(id=41)`。
- incidents：`Incident(id=71)`。

## 6.7.1 Filter option helpers

| 函数调用 | 模拟输出 | 用途 |
|---|---|---|
| `list_brands()` | `["Anker", "Bose"]` | UI sidebar brand selector。 |
| `list_aspects()` | `["battery", "bluetooth", "overheating", "reliability"]` | UI sidebar aspect selector。 |

## 6.7.2 Overview helpers

| 函数调用 | 模拟输出 |
|---|---|
| `get_overview_stats()` | `{"total_reviews": 9, "total_brands": 2, "total_skus": 3, "absa_processed_reviews": 9, "total_aspect_mentions": 10, "total_clusters": 1, "total_incidents": 1, "total_ingest_runs": 1}` |
| `get_reviews_by_brand()` | `[{"brand": "Anker", "reviews": 8}, {"brand": "Bose", "reviews": 1}]` |
| `get_absa_status_distribution()` | `{"success": 9, "no_mentions": 0, "invalid": 0, "failed": 0}` |
| `get_aspect_distribution()` | `[{"aspect_code": "reliability", "mentions": 6}, {"aspect_code": "overheating", "mentions": 1}, ...]` |
| `get_sentiment_distribution()` | `{"negative": 8, "positive": 2}` |

## 6.7.3 ABSA page helpers

### `get_aspect_sentiment_matrix(...)`

输入：

```python
get_aspect_sentiment_matrix(
    brand="Anker",
    sentiment="negative",
    rating_min=1,
    rating_max=2,
)
```

输出：

```python
[
    {"aspect_code": "reliability", "sentiment": "negative", "count": 6},
    {"aspect_code": "overheating", "sentiment": "negative", "count": 1},
]
```

### `get_severity_distribution(...)`

```python
get_severity_distribution(brand="Anker", aspect="reliability")
```

输出：

```python
{"low": 1, "medium": 3, "high": 2}
```

### `get_negative_examples(...)`

```python
get_negative_examples(aspect="reliability", limit=2)
```

输出：

```python
[
    {
        "review_id": 501,
        "brand": "Anker",
        "aspect_code": "reliability",
        "severity": "high",
        "evidence_quote": "stopped working after two weeks",
        "text_snippet": "Stopped working. Anker power bank ...",
    },
    {
        "review_id": 513,
        "evidence_quote": "completely dead after two uses",
        ...
    },
]
```

## 6.7.4 Cluster page helpers

### `get_clusters(aspect="reliability")`

输出：

```python
[
    {
        "cluster_id": 41,
        "aspect_code": "reliability",
        "label": "stopped working after short use",
        "size": 6,
        "severity_weighted_size": 13.0,
        "algorithm": "tfidf_kmeans",
        "topic_keywords": ["stopped working", "failed", "dead"],
        "representative_quotes": ["stopped working after two weeks", ...],
    }
]
```

### `get_cluster_members(41, limit=2)`

输出：

```python
[
    {"review_id": 501, "severity": "high", "evidence_quote": "stopped working after two weeks", ...},
    {"review_id": 513, "severity": "high", "evidence_quote": "completely dead after two uses", ...},
]
```

## 6.7.5 Incident page helpers

| 函数调用 | 模拟输出 |
|---|---|
| `list_incident_aspects()` | `["reliability"]` |
| `list_incident_granularities()` | `["cluster"]` |
| `get_incidents(aspect="reliability")` | `[{"incident_id": 71, "granularity": "cluster", "cluster_label": "stopped working after short use", "week_start": "2024-04-22", "observed_volume": 9, "baseline_volume": 2.4, "z_score": 6.6, "severity_score": 18.0, ...}]` |

## 6.7.6 DQ page helpers

| 函数调用 | 模拟输出重点 |
|---|---|
| `get_ingest_runs(limit=10)` | `[{"run_id": 90, "source": "amazon_reviews_2023_mvp_subset", "status": "completed", "n_rows": 1, ...}]` |
| `get_dq_summary()` | `{"total_events": 7, "total_rows_checked": 28, "total_failed": 3, "failures_by_reason": {"duplicate_review_id": 1, "text_too_short": 1, "invalid_rating": 1}, ...}` |
| `get_pipeline_coverage()` | `{"total_reviews": 9, "absa_processed_reviews": 9, "reviews_with_mentions": 9, "processed_coverage_rate": 1.0, "absa_success_rate": 1.0, ...}` |

注意 `get_dq_summary()` 中 `total_rows_checked` 是按 DQ event 累加的检查行数，不等于 ingest raw rows。每个 check 都会写 event，因此它用于展示“检查覆盖统计”。

---

## 6.8 Streamlit page 函数如何消费这些结果

## 6.8.1 `overview.render()`

它会调用：

```text
db.get_overview_stats
_qdrant_point_count
db.get_reviews_by_brand
db.get_absa_status_distribution
db.get_aspect_distribution
db.get_sentiment_distribution
```

页面结果：

- 两行 metrics。
- reviews by brand 柱图。
- ABSA status 柱图。
- aspect/sentiment 分布图。

---

## 6.8.2 `absa.render()`

假设 sidebar 选：

```text
brand=Anker
aspect=reliability
sentiment=negative
severity=(any)
rating range=1..2
```

它会调用：

```text
_opt("(any)") -> None
get_aspect_sentiment_matrix
get_severity_distribution
get_negative_examples
get_negative_examples(aspect="reliability") for spotlight
```

页面结果：

- reliability negative count。
- severity distribution。
- R501 等 negative examples。

---

## 6.8.3 `clusters.render()`

它会调用：

```text
db.get_clusters
db.get_cluster_members
```

页面结果：

- cluster table 出现 `Cluster(id=41)`。
- 用户选 cluster 后看到 `R501` 的 quote。

---

## 6.8.4 `incidents.render()`

它会调用：

```text
db.list_incident_granularities
db.list_incident_aspects
db.get_incidents
```

页面结果：

- `Incident(id=71)` 出现在 severity 排序表。
- detail 展示 EWMA baseline、z-score、example quote。

---

## 6.8.5 `retrieval.render()`

用户输入：

```text
query="power bank stopped working too hot"
mode="hybrid"
brand="Anker"
aspect="reliability"
sentiment="negative"
```

函数链：

```text
_build_backends
-> run_search
-> hybrid_search
-> optionally run_answer
```

页面结果：

- raw search hits 中 R501 靠前。
- 点击 answer button 后显示 cited answer。

---

## 6.8.6 `agent.render()`

用户输入：

```text
What are the main negative reliability complaints for Anker?
```

函数链：

```text
run_agent
-> result_summary
->_render_result
```

页面结果：

- route 显示 `retrieval_answer`。
- filters 显示 `brand/aspect/sentiment`。
- answer 显示 citations。

---

# 7. 从一条 raw row 到一个用户答案的总轨迹

下面把 `R501` 的变化压缩成一张表。

| 阶段 | 函数主链 | R501 的形态 |
|---|---|---|
| 原始数据 | `_stream_jsonl` | Amazon raw JSON。 |
| 来源标准化 | `_normalize_row` | normalized row，补品牌、ASIN、timestamp、text。 |
| 质量门 | `_row_dedup_key`、`_check_row`、`run_dq` | valid row。 |
| 入库 | `load_reviews` | `Review(id=501)`。 |
| ABSA | `provider.extract`、`validate_absa_output`、`extract_for_review`、`absa_flow` | `AspectMention(reliability)`、`AspectMention(overheating)` 与 status。 |
| 索引 | `build_embedding_text`、`build_payload`、`embed_flow` | Qdrant point。 |
| 检索 | `build_search_filter`、`retrieve`、`BM25.rank`、`hybrid_search` | `SearchHit(review_id=501)`。 |
| 聚类 | `build_clustering_records`、`cluster_records`、`persist_clusters` | `ReviewCluster(cluster_id=41, review_id=501)`。 |
| 异常 | `aggregate_weekly`、`detect_anomalies`、`upsert_incident` | spike week incident 的证据成员。 |
| 回答 | `build_citations`、`generate_answer` | cited answer 中的 `[1]` evidence。 |
| Agent/UI | `classify_route`、`run_agent`、`ui/db.py` queries | 用户看到 route、answer、charts、cluster、incident。 |

---

# 8. 阅读建议

读本文时不要只看“输出长什么样”，还要反向问：

1. 这个函数为什么不直接调用下一层？
2. 它的输出为什么被设计成 dict、dataclass、ORM row 或 `SearchHit`？
3. 它把脏数据、模型不确定性、幂等性、成本或 UI 解耦问题解决在哪一层？

当你能自己用另一条 review 复写一遍 `raw row -> cited answer` 的轨迹时，VoiceLens 的代码主线就基本掌握了。
