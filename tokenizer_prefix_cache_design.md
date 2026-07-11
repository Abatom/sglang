# SGLang Tokenizer 前缀缓存设计方案（/v1/chat/completions 多轮对话）

> 场景：百万 token 上下文多轮对话，KV cache 前缀命中 93%，prefill 已很便宜，
> 但每轮请求都对全部渲染后的历史文本重新 `tokenizer.encode`（~4MB 文本，秒级 CPU，
> 且同步阻塞事件循环），成为主要耗时。
> 目标：缓存「渲染文本 → token ids」，每轮只 encode 新增后缀。

---

## 一、现状：tokenize 的处理路径

### 1.1 两条入口路径

1. **`/generate` 原生接口**：`tokenizer_manager.py` 的 `_tokenize_one_request()` →
   `_tokenize_texts()`（tokenizer_manager.py:711-796），默认直接在 **asyncio 事件循环线程上
   同步调用** `self.tokenizer(...)`（:784/:787）。一个百万 token 的 encode 会把整个事件循环
   卡住，阻塞所有其他请求的收发。

2. **OpenAI chat completions（本方案目标路径）**：serving 层自己完成编码
   （`serving_chat.py` 的 `_apply_jinja_template`）：
   - 渲染：`apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
     tools=tools, **extra_template_kwargs)`（serving_chat.py:868-875）
   - 编码：`prompt_ids = tokenizer.encode(rendered_prompt, **encode_kwargs)`（:876-878），
     其中 `encode_kwargs = {"add_special_tokens": False}` 当且仅当
     `self._tokenizer_auto_adds_specials`（`__init__` :228-233 探测，防止双 BOS）
   - 然后把 `input_ids` 塞进 `GenerateReqInput` 传给 tokenizer_manager，
     走 `obj.input_ids is not None` 分支（tokenizer_manager.py:819）跳过重复编码。
   - tools 回退重试路径（:888-900）会重新渲染+编码一次。

### 1.2 现有缓解手段（都不是缓存）

- `--enable-dynamic-batch-tokenizer`：把 encode 移到单线程 executor 并合并小请求
  （`async_dynamic_batch_tokenizer.py`），针对高并发小请求，对单个巨型 prompt 无益。
- `--tokenizer-worker-num > 1`：多进程 tokenizer（`multi_tokenizer_mixin.py`），
  能并行不同请求，但同一请求的百万 token 仍是全量重编码。

**结论：SGLang 目前没有任何 text→token 的结果缓存，每一轮对话都对全部历史文本从头
encode。** HF fast tokenizer 单线程吞吐约 2~10 MB/s，百万 token ≈ 3-4 MB 文本，
单次 encode 几百 ms 到秒级。KV 命中 93% 意味着 prefill 本身很便宜，tokenize 占比自然凸显。

---

## 二、核心难点与正确性依据

### 2.1 难点：BPE 不是前缀稳定的

`encode(A + B) ≠ encode(A) + encode(B)`——merge 可能跨越 A/B 边界
（如 A 以 `"hell"` 结尾、B 以 `"o"` 开头会合并成一个 token）。
不能简单拿旧结果拼接。

### 2.2 可利用的性质（正确性论证）

**special/added token 是 BPE 预切分的原子屏障，merge 永远不会跨越它们。**
因此只要在「后缀以 special token 字面量开头」的位置切分：

```
encode(text) == encode(text[:p]) + encode(text[p:])   严格成立
（前提：add_special_tokens=False，text[p:] 以完整 special 字面量开头）
```

字符串开头的 normalizer 特例（prefix space 等）只影响位置 0，永远在缓存前缀内。
多轮对话的渲染结果天然由 `<|im_start|>` 之类 special token 密集分隔，切点充足。

对违反假设的 tokenizer 家族：用**常开的拼接不变量检查 + verify 模式**兜底，
失配自动降级为全量 encode / 自动禁用缓存。

### 2.3 为什么用 LCP 匹配而不是全文 hash

上一轮的完整渲染文本（含 generation prompt 尾部 `<|im_start|>assistant\n`）
**不是**新一轮文本的严格前缀（新一轮会把 assistant 回复以不同形式渲染进历史）。
所以需要最长公共前缀（LCP）匹配，再回退到 LCP 内最后一个完整 special-token 边界。

---

## 三、总体设计

**LCP 会话级前缀缓存**，默认关闭，`--enable-tokenizer-prefix-cache` 打开：

1. 每个近期会话缓存一条：`(渲染文本, ids, special-token 边界表)`。
2. 新请求：桶索引（`hash(text[:4096])`）定位候选 → 块式 memcmp 求 LCP（4MB ~几 ms）
   → 取 LCP 内最后一个完整 special-token 边界为切点
   → 复用 `ids[:cut.token_idx]`，只 encode `text[cut.char_pos:]`，拼接。
3. 边界表构建不需要 offsets_mapping：文本正则扫 special 字面量 + ids 线性扫 special id，
   按序 1:1 配对（内容里出现字面 special 同样会编码成 special id，配对仍成立）；
   计数或 id 不匹配 → 放弃缓存该条目（安全）。
4. 拼接不变量（每次命中常开检查）：`suffix_ids[0] == 切点 special id`；
   若 tokenizer 强插了 BOS 则剥离（镜像现有 `_append_assistant_prefix_to_prompt_ids`
   的做法，serving_chat.py:286-288）；不满足 → 回退全量。
5. verify 模式：前 N 次命中同时跑全量 encode 比对（`-1` = 每次都比），
   失配告警并返回全量结果，累计 3 次后自动禁用缓存。
6. LRU 按总字节数驱逐；命中且 LCP 覆盖旧文本绝大部分时 **replace-on-hit**
   （典型多轮增长，每会话只留一条，省一半内存）。

预期收益：每会话第 2 轮起命中，单轮 encode 从「全量 4MB / 秒级」降为
「新增后缀 / 数十 ms」+ 切片拼接 ~20ms；会话级命中率 ≥ (轮数-1)/轮数。

### 3.1 special-token 边界扫描与配对校验（关键机制详解）

切点需要同时知道两个坐标：**字符位置**（后缀从哪切）和 **token 下标**（缓存 ids 复用到哪）。
常规做法要 `return_offsets_mapping`（每 token 8 字节偏移表，百万 token 多占 8MB 且并非所有
后端支持）；本方案改用「双侧扫描 + 配对校验」，只在 special token 处建坐标锚点——恰好
special token 也正是唯一合法的切点。

**① 构建匹配器（进程启动一次，`build_special_matcher`）**

1. 字面量集合 = `all_special_tokens` ∪ `added_tokens_decoder` 中 `special=True` 的条目。
   后者必须包含：Llama-3 的 `<|start_header_id|>` 等角色标记不在 `all_special_tokens` 里，
   只在 added_tokens_decoder 里。
2. 编成一个正则 alternation，**按长度降序** `re.escape` 拼接。降序是正确性条件：Python
   `re` 的 `|` 是最左优先而非最长匹配，若短字面量排前面会截断匹配长字面量的前缀。
3. 同时建 `literal → token_id` 映射（`convert_tokens_to_ids`）与 `special_id_set`。
4. tokenizer 缺这些属性（如 tiktoken 包装器）→ 返回 None，工厂禁用缓存并告警一次。
   字面量集合进程内不变（patch_tokenizer 禁止启动后新增 special），可安全地只算一次。

**② 双侧扫描（每次插入缓存条目时，`scan_boundaries`）**

```
文本侧: pattern.finditer(text)  →  [(char_pos₀, lit₀), (char_pos₁, lit₁), ...]   按出现序
ids 侧: 线性扫 ids              →  [token_idx₀, token_idx₁, ...]                 按出现序
                                    （取值 ∈ special_id_set 的下标）
```

**③ 配对校验（把「假设」变成「被检查的事实」）**

按序 zip 两个序列，要求同时满足：

1. **数量相等**：`len(文本侧) == len(ids 侧)`；
2. **逐对 id 一致**：`ids[token_idxₖ] == literal_to_id[litₖ]`。

全部通过 → 每对生成一个切点锚 `SpecialBoundary(char_pos, token_len, token_idx, token_id)`；
任一失败 → 返回 None，**该条目不入缓存**（`insert_skips` 计数）。宁可少缓存，绝不带病入库
——否则下一轮命中就是错误拼接。

例（Qwen 模板）：

```
text: <|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n
       ^char 0             ^char 19     ^char 30
ids:  [151644, 872, 198, 6023, 151645, 198, 151644, 77091, 198]
       ^idx 0                  ^idx 4       ^idx 6
配对: (0,"<|im_start|>")↔idx0✓  (19,"<|im_end|>")↔idx4✓  (30,"<|im_start|>")↔idx6✓
→ 3 个 SpecialBoundary，即 3 个合法切点
```

**④ 为什么 1:1 对应成立（以及什么情况会被拦住）**

- 正向：special token 走 added-token 预切分，文本里每个字面量——**包括用户内容里恶意
  嵌入的 `<|im_end|>`**——都会且只会编码成对应的 special id（与全量 encode 行为一致，
  所以嵌入攻击不影响正确性，只是多一个可用切点）。
- 反向：`add_special_tokens=False` 下，ids 里的 special id 只能来自文本字面量，不会凭空
  出现。顺序两侧都按出现序，天然对齐。
- 会被数量校验拦住的典型异常：tokenizer 无视 kwargs 强插 BOS（BOS ∈ special_id_set，
  ids 侧多一个 → 数量不等 → 不缓存）；自定义 normalizer 改写了字面量导致文本侧匹配数
  与 ids 侧不符。这类 tokenizer 自动退化为「永不缓存」，正确性无损。

**⑤ 命中时的增量扫描**

切点之前的文本与匹配条目逐字节相同（LCP 保证），边界锚直接从旧条目继承
（`char_pos + token_len <= cut.char_pos` 的那些）；只对新后缀跑 ②③，用
`char_offset=cut.char_pos, token_offset=cut.token_idx` 平移坐标。每轮开销 O(新增内容)。
miss 时全量扫描：正则对 4MB 文本是线性 C 代码（几十 ms），仍比秒级 encode 低两个量级。

**⑥ 切点的使用（与 `_pick_cut` 的衔接）**

存储时额外物化并行数组 `boundary_char_ends = [char_pos + token_len]`，查找时
`bisect_right(boundary_char_ends, LCP)` 取最后一个**完整落在公共前缀内**的锚——
`+ token_len` 是正确性条件：若 LCP 停在某个 special 字面量中间，该锚不可用，必须退到
更早的锚。命中后再由「拼接不变量」（后缀 encode 首 token == `token_id`）做最后一道
运行时校验。

---

## 四、改动清单

### 4.1 新模块 `python/sglang/srt/tokenizer/prefix_cache.py`

放在 `srt/tokenizer/` 而非 `entrypoints/openai/`：缓存是 tokenization 的属性，
实例挂在 TokenizerManager 上被 chat / responses / tokenize / grpc bridge 共享，
放 entrypoints 会倒置依赖方向。

结构体（**msgspec.Struct**，仓库规则禁新增 dataclass；kw_only，`SpecialBoundary` frozen）：

```python
class SpecialBoundary(msgspec.Struct, frozen=True, kw_only=True):
    char_pos: int      # special 字面量在 text 中的起始偏移
    token_len: int     # 字面量字符长度
    token_idx: int     # 对应 special id 在 ids 中的下标
    token_id: int      # special id（拼接不变量校验用）

class PrefixCacheEntry(msgspec.Struct, kw_only=True):
    text: str                          # 完整渲染文本（持引用，不拷贝）
    ids: array                         # array('i')，全量 encode 结果
    boundaries: list[SpecialBoundary]  # 按 char_pos 升序
    boundary_char_ends: list[int]      # 并行数组 [char_pos+token_len]，bisect 用
    add_specials_false: bool           # 该条目所属 encode_kwargs 命名空间
    nbytes: int

class PrefixCacheStats(msgspec.Struct, kw_only=True):
    lookups: int = 0; hits: int = 0; chars_saved: int = 0
    splice_rejects: int = 0; insert_skips: int = 0
    verify_mismatches: int = 0; entries: int = 0; total_bytes: int = 0
```

模块级纯函数（仓库规则：无状态优先）：

- `_build_special_matcher(*, tokenizer)`：字面量集合 = `all_special_tokens` ∪
  `added_tokens_decoder` 中 `special=True` 的条目（**必须包含后者**——Llama-3 的
  `<|start_header_id|>` 等角色标记不一定在 `all_special_tokens` 里）；
  正则按**长度降序** `re.escape` 拼接（Python re 是 leftmost-first，非最长匹配）；
  返回 pattern + literal→id 映射；tokenizer 缺属性（tiktoken 包装器）→ 返回 None，
  工厂据此禁用并 warning 一次。字面量集合进程内不变（`patch_tokenizer.py` 的
  patcher 禁止事后添加 special），构造时算一次。
- `_scan_boundaries(*, text, ids, matcher, literal_ids, special_id_set,
  char_offset=0, token_offset=0) -> list[SpecialBoundary] | None`：
  finditer 文本 + 线性扫 ids，按序配对并校验 `ids[token_idx] == literal_ids[literal]`；
  不匹配返回 None。offset 参数支持命中后仅增量扫后缀（O(增量)/轮）。
- `_longest_common_prefix(*, a, b) -> int`：256KB 块切片相等比较（C 级 memcmp），
  首个不等块内二分。
- `_pick_cut(*, entry, lcp) -> SpecialBoundary | None`：
  `bisect_right(boundary_char_ends, lcp)` 取最后一个满足
  **`char_pos + token_len <= lcp`** 的边界（`+ token_len` 是正确性条件：
  切点 special 字面量必须完整落在公共前缀内）；
  节省不足 `MIN_USEFUL_CUT_CHARS`(4096) 字符返回 None。

类与工厂：

```python
class TokenizerPrefixCache:
    def __init__(self, *, tokenizer, max_size_bytes, verify_first_n,
                 max_verify_mismatches=3, min_prompt_chars=8192): ...
    def encode_with_cache(self, *, text: str, add_specials_false: bool) -> list[int]: ...
    def stats(self) -> PrefixCacheStats: ...

def maybe_create_tokenizer_prefix_cache(*, tokenizer, enable, max_size_mb,
                                        verify_first_n) -> TokenizerPrefixCache | None
```

- 状态：`OrderedDict[int, PrefixCacheEntry]`（LRU，仿 `multimodal_cache.py:102-121`
  的形状）、`hash(text[:4096]) → [entry_id]` 桶索引、`_total_bytes`、`_stats`、
  `_disabled`、构造时捕获 `bos_token_id`。
- 并发：`_process_messages` 同步跑在事件循环上（serving_chat.py:560→:632），
  每个 tokenizer worker 进程内调用天然串行，**无需锁**（docstring 记录该前提）。

`encode_with_cache` 主流程（顶层读起来像伪代码，细节压进 helper）：

```
1. _disabled 或 len(text) < min_prompt_chars → 全量 encode（不插入）
2. 桶内遍历候选（跳过 add_specials_false 命名空间不同的）→ 求 LCP 取最优
3. _pick_cut；无切点 → miss：全量 encode，转 7
4. suffix_ids = tokenizer.encode(text[cut.char_pos:], **与全量路径一致的 kwargs)
5. 拼接不变量（常开）：suffix_ids[0] == cut.token_id；
   若 [bos, cut.token_id] 开头则剥 BOS；均不满足 → splice_rejects++，回退全量，转 7
6. result = entry.ids[:cut.token_idx].tolist() + suffix_ids；
   verify 模式：前 verify_first_n 次命中同时全量比对（-1=永远），
   失配 → warning（tokenizer 类名、切点、首个分歧下标）、返回全量结果，
   累计 max_verify_mismatches 次后 _disabled = True
7. 插入：命中 → 复用切点前 boundaries + 仅增量扫后缀；miss → 全量扫
   （正则 4MB 线性，几十 ms ≪ encode）；扫描 None → insert_skips++ 不缓存。
   replace-on-hit：best_lcp >= len(best.text) - 16384 时替换原条目；
   否则新增（共享长 system prompt 的两个会话不互踩）。
   LRU 按字节驱逐（popitem(last=False) + 清桶索引）
8. 每 1000 次 lookup 打一条 info 统计快照
```

### 4.2 `python/sglang/srt/managers/tokenizer_manager.py`

（仓库规则 large-class-style：`__init__` 只加一行编排，逻辑全在协作模块）

- `__init__`（:257-302）在 `self.init_tokenizer_and_processor()`（:278）后加一行：
  `self.maybe_init_tokenizer_prefix_cache()`
- 新 helper（:332 附近，只做构造接线）：

```python
def maybe_init_tokenizer_prefix_cache(self):
    self.tokenizer_prefix_cache = maybe_create_tokenizer_prefix_cache(
        tokenizer=self.tokenizer,
        enable=self.server_args.enable_tokenizer_prefix_cache,
        max_size_mb=self.server_args.tokenizer_prefix_cache_size_mb,
        verify_first_n=self.server_args.tokenizer_prefix_cache_verify_first_n,
    )
```

- 字段**恒置**（禁用时为 None）→ 满足仓库 no-getattr-defensive 规则。
- 挂 TokenizerManager 而非 serving 实例的原因：每进程有 ≥4 个
  OpenAIServingChat(-子类) 实例（http chat、responses、tokenize 内部实例、grpc
  bridge），全部已持有 tokenizer_manager，共享一份缓存避免内存翻倍与命中率碎片化。

### 4.3 `python/sglang/srt/server_args.py`

- `disable_tokenizer_batch_decode`（:2638-2641）后追加三个 `A[...]` 注解字段
  （argparse 由注解自动生成，无需手写 add_argument）：
  - `enable_tokenizer_prefix_cache: bool = False`
  - `tokenizer_prefix_cache_size_mb: int = 256`（每 tokenizer worker 进程，LRU 上限）
  - `tokenizer_prefix_cache_verify_first_n: int = 16`（-1=每次命中都校验；0=不校验）
  - help 注明：作用于 Jinja chat template 的 /v1/chat/completions 与 /v1/responses；
    `--tokenizer-worker-num > 1` 时各进程独立缓存。
- `_handle_tokenizer_batching`（:5969）的 `skip_tokenizer_init` 分支内
  （镜像 :5990-5994）：开着则 warning + 置 False（解析期变更合法，符合
  runtime-context 的 resolve-at-end 契约）。
- 与 batch_encode / dynamic_batch_tokenizer 无冲突（那是 /generate 文本路径），不加互斥。
- tokenizer 后端能力不在 arg 层校验（依赖已加载对象），由工厂运行时处理。

### 4.4 `python/sglang/srt/entrypoints/openai/serving_chat.py`

- `__init__`（:233 探测块后）：
  `self._tokenizer_prefix_cache = tokenizer_manager.tokenizer_prefix_cache`
- 新 helper（:273 附近）：

```python
def _encode_rendered_prompt(self, *, rendered_prompt: str, encode_kwargs: dict) -> List[int]:
    if self._tokenizer_prefix_cache is None:
        return self.tokenizer_manager.tokenizer.encode(rendered_prompt, **encode_kwargs)
    return self._tokenizer_prefix_cache.encode_with_cache(
        text=rendered_prompt, add_specials_false=bool(encode_kwargs))
```

- 替换两处 encode：**:876-878** 与 **:898-900**（tools 回退重试同样走缓存——
  重试渲染是确定性的，同会话后续轮次会命中）。
- 明确不动的路径：
  - `input_ids` 直传：:689-698 提前返回，天然跳过缓存；
  - `assistant_prefix`（continue_final_message）：在拼接结果之后追加（:908-911），不受影响；
  - 多模态：`decode(prompt_ids)`（:913-914）仍全量，encode 侧照常加速（已知限制）；
  - `_apply_conversation_template`（:931+）：v1 范围外；
  - dsv4/dsv32 自定义编码（:801/:806）：第三种 kwargs 命名空间，v1 排除，后续 PR
    两行即可接入（条目的 `add_specials_false` 字段已做命名空间隔离）。
- `serving_responses.py` / `serving_tokenize.py` / `grpc_bridge.py` 零改动
  （继承/共享同一 `_process_messages` 与 tokenizer_manager）。

### 4.5 ids 内存格式

- 存 `array('i')`（4B/token）。1M token 会话 ≈ 4MB ids + ~4MB text（持有
  rendered_prompt 引用，不拷贝）；默认 256MB ≈ 30 个百万级活跃会话。
- 插入时 `array('i', ids)` 一次 ~10ms；命中时 C 级切片 + `.tolist()` ~20ms，
  均比被替代的秒级 encode 低两个数量级。返回 `list[int]` 使下游
  （GenerateReqInput、usage 统计）零改动。

### 4.6 可观测性（v1 自包含）

- 统计计数器 + 每 1000 lookup 一条 info 日志：
  `TokenizerPrefixCache: hits=812/1000 (81.2%), saved=3.1e9 chars, entries=24, 212.4/256.0 MB`
- verify 失配与自动禁用打 WARNING。
- Prometheus 指标（`TokenizerMetricsCollector` 加 hits/lookups counter + bytes gauge）
  作为后续 PR。

---

## 五、测试计划

新文件 `test/registered/unit/tokenizer/test_prefix_cache.py`
（`CustomTestCase` + `register_cpu_ci`；CPU 单测允许下载真实 tokenizer——
`test_patch_tokenizer.py:177` 有先例；用未 gated 的 `Qwen/Qwen2.5-0.5B-Instruct`，
`<|im_start|>/<|im_end|>` 每条消息都有，切点密集）。

用例（每条注明使其变红的 diff，满足 unit-test-admission 规则）：

| 用例 | 守护的行为 |
|---|---|
| `test_spliced_equals_full_encode_multi_turn` | 10 轮增长逐轮断言拼接 == 全量，且后续轮次真命中；变体：assistant 内容含字面 `<|im_end|>`（specials-in-content 配对） |
| `test_cut_never_lands_inside_special_or_mid_token` | LCP 落在 special 字面量中间 → 切点退到更早边界（守护 `+ token_len` 条件） |
| `test_boundary_scan_mismatch_skips_insert` | 计数不匹配 stub → 不缓存、输出为全量 |
| `test_lru_eviction_by_bytes` | 最旧驱逐、桶索引清理、字节守恒 |
| `test_verify_mode_fallback_and_autodisable` | 失配返回全量、告警、3 次后禁用且后续直通 |
| `test_bos_stripped_from_suffix_encode` | 强插 BOS stub → 剥离后精确；剥离后仍不满足不变量 → 回退 + splice_rejects |
| `test_encode_kwargs_namespaces_do_not_mix` | `add_specials_false=True` 条目不服务 `False` 查询 |
| `test_short_prompt_bypasses_cache` | 短 prompt 直通不插入 |

接线测试：`test/registered/unit/entrypoints/openai/test_serving_chat.py` 的
`_MockTokenizerManager`（:38-80）加 `self.tokenizer_prefix_cache = None`
（no-getattr 规则下真实代码无条件读该属性；其余构造 serving 实例的 mock 一并 grep 补齐）；
新增 `test_jinja_encode_routes_through_prefix_cache`（Mock 缓存断言 :876 走 hook）。

运行：

```bash
python -m unittest test.registered.unit.tokenizer.test_prefix_cache -v
python -m unittest test.registered.unit.entrypoints.openai.test_serving_chat -v
```

---

## 六、实施顺序

1. `prefix_cache.py` + 其单测（自包含，先跑绿）。
2. `server_args.py` 字段 + `_handle_tokenizer_batching` 分支。
3. `tokenizer_manager.py` 一行编排 + helper。
4. `serving_chat.py` hook + mock 补齐 + 接线单测。

---

## 七、上线验证（用户侧）

1. 预发全量校验：`--enable-tokenizer-prefix-cache --tokenizer-prefix-cache-verify-first-n -1`
   回放生产多轮流量，要求零 `verify mismatch` 告警后再降回默认 16。
2. 容量估算：`活跃会话数 × (文本字节 + 4×token 数)`；
   4MB 文本场景 `--tokenizer-prefix-cache-size-mb 1024` ≈ 120 会话。
3. 基准：回放单会话至第 N 轮（如 50 轮 × 20k token/轮），对比 flag 开/关的逐轮 TTFT
   （流式首 chunk 延迟）；预期第 N 轮 TTFT 下降约等于原 encode 墙钟时间（百万 token 秒级）。
4. 输出无漂移：`temperature=0` + `return_prompt_token_ids=true` 开/关对比 prompt token ids。

---

## 八、已知限制

- ~~多模态路径 encode 提速但 `decode(prompt_ids)` 仍全量~~ **已解决**：开启
  `--enable-tokenizer-prefix-cache` 后，多模态模型的**纯文本请求**（无 image/video/audio
  数据）按请求级判断直传 `input_ids`，跳过 `decode(prompt_ids)` 与 tokenizer_manager
  的二次全量 encode（mm processor 在 `contains_mm_input()` 为假时本就不运行，
  tokenizer_manager.py:856-858；MossVL 架构例外，仍传文本）。带多模态文件的请求
  维持原路径（encode 侧照常被缓存加速，decode 仍全量）。
- 模板每轮改写早期历史（如剥离旧轮 reasoning content）会缩短 LCP → 优雅降级为全量 encode。
- special token 既不在 `all_special_tokens` 也不在 `added_tokens_decoder` 的 tokenizer
  → 零边界，缓存静默无效（首次插入日志打印边界数以便发现）。
- `--tokenizer-worker-num > 1` 时各 worker 进程独立缓存（会话粘性路由可提高命中）。
- miss 路径（新会话冷启动巨型 prompt）仍在事件循环上全量 encode——
  如需彻底解决可另行把 `_process_messages` 移入线程池（HF fast tokenizer 的 Rust
  核心释放 GIL），作为独立后续工作。
