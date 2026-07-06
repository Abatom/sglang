# Tokenizer Worker CPU 热点治理:双重编码消除 + Encode 异步化 + 前缀缓存

> 适用场景:多模态模型(VLM)+ 纯文本长输入、多轮对话流量、`--tokenizer-worker-num` 多 worker 部署。
> 变更日期:2026-07。

## 1. 问题现象

`--tokenizer-worker-num` 多 worker 部署下,tokenizer_worker 进程 CPU 严重不均(示意):

```
~100%   worker A     ← 打满
 ~70%   worker B
 ~50%   worker C
  ...
  ~0%   其余绝大多数 worker  ← 完全空闲
```

`ss` 确认:少量 established keep-alive 长连接集中在少数几个 worker 上(单 worker 持有多条),其余大部分 worker 一条连接都没有。

## 2. 诊断过程与根因

### 2.1 连接层(表象)

多 worker 模式使用 `uvicorn.run(..., workers=N)`,所有 worker 共享一个 listen socket:

- 新连接落到哪个 worker 由内核 accept 竞争决定,天然偏斜;
- HTTP keep-alive 下,连接一旦被某 worker accept,其上所有请求终生绑定该 worker;
- 客户端连接池只维持少量长连接 → 大部分 worker 分不到活。

连接级方案(SO_REUSEPORT、每 N 请求 `Connection: close` 强制轮换)被评估后放弃:治标不治本,依赖客户端重连行为。

### 2.2 请求成本层(根因,py-spy 火焰图)

对最热 worker 采样 30s:

| 占比 | 位置 |
|---|---|
| **~34.6%** | `_apply_jinja_template` → `apply_chat_template` → HF encode(第 1 次编码) |
| **~34.8%** | `generate_request` → `_tokenize_one_request` → `_tokenize_texts` → HF encode(**第 2 次编码,纯浪费**) |
| ~13% | Prometheus `/metrics` multiprocess collect(全部 worker 的 mmap 合并) |
| ~4.8% | `_validate_request` jsonschema 工具校验 |
| ~4.2% | Jinja 渲染 |

**两个根因:**

1. **双重编码**:多模态模型的 chat 请求,serving 层先把 chat template encode 成 prompt_ids,再 `decode` 回文本传给 TokenizerManager,后者又 **re-encode 一遍**。纯文本请求根本不会进多模态 processor,这一来一回(decode + re-encode)完全是浪费。纯文本模型无此问题(直接传 input_ids)。
2. **encode 同步跑在 event loop 主线程**:单核打满的同时,阻塞该 worker 上所有并发请求的处理。

结论:问题本质不是"连接分布不均",而是**单请求 tokenize 成本过高且全部压在单线程上**,连接不均只是放大器。

## 3. 改动内容

### 3.1 消除双重编码(无需配置,部署即生效)

`python/sglang/srt/entrypoints/openai/serving_chat.py`

- 新增 `_multimodal_prompt_kwargs()`:多模态模型的**纯文本请求**直接透传 `input_ids`(chat template 已编好),跳过 decode 和下游 re-encode;携带图片/音频/视频数据的请求保持原文本路径(多模态 processor 需要文本);
- `_apply_jinja_template` 仅在请求实际含多模态数据时才 `decode(prompt_ids)`;
- MossVL 等强制走 mm processor 的架构保留原行为(`_forces_mm_processor`);
- conversation 模板路径(prompt_ids 为空)自动保持原文本路径;
- `serving_responses.py`(Responses API)同步接入。

**收益:热点 worker 直接砍掉 ~35% CPU(re-encode)+ decode 开销。**

### 3.2 chat encode 异步化(开 `--enable-dynamic-batch-tokenizer` 生效)

- `_convert_to_internal_request` 调用链全面 async 化(serving_base 抽象方法 + 全部 9 个实现 + anthropic/tokenize 调用点);
- 新增 `_encode_rendered_prompt()`:启用 dynamic batch tokenizer 时,chat template 的 encode 走 `AsyncDynamicbatchTokenizer`(独立线程 + 并发请求动态攒批);未启用时行为与历史完全一致。

**收益:encode 移出 event loop,主线程不再被长输入卡死;热点 worker 可用 ~2 个核(主线程 + encode 线程)。**

约束:与 `--enable-tokenizer-batch-encode` 互斥;`skip_tokenizer_init` 下自动禁用。

### 3.3 Tokenizer 前缀缓存(开 `--enable-tokenizer-prefix-cache` 生效)

`python/sglang/srt/managers/tokenizer_prefix_cache.py`(新模块)

多轮对话每轮都对**全部历史**重新 encode,单会话总代价 O(L²)。前缀缓存记住最近 encode 过的 prompt 及其 token ids,新请求命中公共前缀时只 encode 新增后缀。

**正确性设计**(核心):BPE 在任意切分点不满足 `encode(A+B) == encode(A)+encode(B)`(merge 可跨切点)。唯一通用安全切分点是**特殊 token 边界**——HF fast tokenizer 先按 added special tokens 切分文本再做 BPE,merge 永不跨越。因此:

- 复用只截止到公共前缀内**最后一个特殊 token 结束处**,其后字符与新后缀一起重编;
- 多轮对话场景,上一轮 prompt 以 generation prompt(`<|im_start|>assistant\n`)结尾,恰是下一轮的严格前缀,切点距分叉仅 ~10 字符,**复用率接近 100%**;
- 特殊 token 取自 `added_tokens_decoder`(覆盖 Llama-3 式未注册进 `additional_special_tokens` 的 header token),排除 `rstrip=True` 的 token(会吞掉后文空白);
- insert 时对字符串中特殊 token 出现序列与 ids 中特殊 id 序列**逐对校验**,任何不一致直接放弃缓存(安全兜底);
- LRU,默认 32 条/worker;短于 1024 字符的 prompt 不缓存;
- 非 fast tokenizer(如 TikToken 系)自动降级禁用并打日志。

**收益:多轮长对话从第 2 轮起,每轮 encode 代价 ≈ O(新增内容)。** 连接亲和(同会话固定同 worker)天然保证 per-worker 命中率。

## 4. 推荐配置

```bash
python -m sglang.launch_server \
  --model <多模态模型> \
  --tokenizer-worker-num 8 \              # 无需大量 worker:少而能打,还降低 /metrics 合并成本
  --enable-dynamic-batch-tokenizer \      # encode 离开 event loop + 动态攒批
  --enable-tokenizer-prefix-cache \       # 多轮只 encode 增量
  --tokenizer-prefix-cache-size 32        # 可选,默认 32
```

## 5. 效果账(基于热点 worker 火焰图)

| 成分 | 改动前 | 改动后 |
|---|---|---|
| 第 2 次 encode + decode | ~35% | **0**(3.1,无条件) |
| 第 1 次 encode | ~35%,阻塞主线程 | 移到独立线程(3.2);多轮流量下再被前缀缓存(3.3)压到增量级 |
| 主线程占用 | ~100%(饱和,排队) | ~30%(metrics/校验/流式序列化) |

注意:本方案**不改变连接→worker 的分布**,而是把单 worker 有效容量提升数倍,使现有分布方式不再成为瓶颈。若未来单 worker 流量超过 ~2 核(主线程 + 单 encode 线程;多模态路径 `TOKENIZERS_PARALLELISM` 被强制 false,Rust 侧单线程),需再评估请求级均衡(共享 encode 池/前置代理)。

## 6. 验证

**单测**(CPU,CI suite `base-a-test-cpu`):

```bash
python -m pytest test/registered/unit/managers/test_tokenizer_prefix_cache.py -v
python -m pytest test/registered/unit/entrypoints/openai/test_serving_chat.py -v
```

**部署后验证:**

1. 正确性:同一多轮会话,开/关 `--enable-tokenizer-prefix-cache` 对比 `usage.prompt_tokens` 必须一致;
2. 效果:`py-spy record -d 30 -p <热点pid>`,确认 `_tokenize_texts` 第二次 encode 消失、`apply_chat_template` encode 移入 AsyncDynamicbatchTokenizer 线程、多轮流量下 encode 占比随轮次下降;
3. 资源:worker RSS 增量封顶于 `cache_size × 单条 prompt 体积`(默认 32 条;超长 prompt 场景按此公式估算,量级可控)。

## 7. 已知边界与后续项

- **同一长文档内部分叉**(同一文档多次提问、RAG 超长 system):分叉点在单条消息正文内部,特殊 token 切点会退化到消息开头。后续可扩展"`\n` 后接非空白"切点(需启动时用真实 tokenizer 做切分等价性自检,通过才启用);
- `/generate` 文本路径的前缀缓存接入(本期只接 chat Jinja 路径);
- `/metrics` multiprocess collect 成本(~13%,随 worker 数增长,减 worker 后缓解);
- jsonschema 工具校验(~4.8%);
- `TOKENIZERS_PARALLELISM` 在多模态路径被强制 `false`(tokenizer_manager),可改为尊重用户环境变量以启用 Rust 侧多线程 batch encode。
