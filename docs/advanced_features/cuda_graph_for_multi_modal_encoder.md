# ViT 如何使用 CUDA Graph（代码级梳理）

本文按代码路径梳理 SGLang 中 ViT（多模态视觉编码器）如何使用 CUDA Graph，重点覆盖：

- Qwen2.5-VL（窗口注意力 + 重排）
- Qwen3-VL（deepstack）
- InternVL（独立 runner，key 为 `(B, S)`）

> 代码基线：当前分支 `cursor/vit-cuda-graph-c60b`。

---

## 1. 开关和入口

统一总开关是环境变量 `SGLANG_VIT_ENABLE_CUDA_GRAPH`：

```python
# python/sglang/srt/environ.py
SGLANG_VIT_ENABLE_CUDA_GRAPH = EnvBool(False)
```

模型侧入口：

```python
# python/sglang/srt/models/qwen2_5_vl.py
self.enable_cg = _is_cuda and envs.SGLANG_VIT_ENABLE_CUDA_GRAPH.get()
if self.enable_cg:
    self.cuda_graph_runner = ViTCudaGraphRunner(self)

def forward(self, x, grid_thw):
    if self.enable_cg:
        return self.forward_with_cuda_graph(x, grid_thw)
```

```python
# python/sglang/srt/models/qwen3_vl.py
def forward(self, x, grid_thw):
    if envs.SGLANG_VIT_ENABLE_CUDA_GRAPH.get():
        return self.forward_with_cuda_graph(x, grid_thw)
```

```python
# python/sglang/srt/models/internvl.py
self.enable_cg = _is_cuda and envs.SGLANG_VIT_ENABLE_CUDA_GRAPH.get()
if self.enable_cg:
    self.cuda_graph_runner = InternViTCudaGraphRunner(self)
```

---

## 2. 执行链路（高层）

ViT CUDA Graph 的共性流程可以总结为：

1. **图外预处理（eager）**：patch embed、位置编码、`cu_seqlens` 计算、（Qwen2.5 的 window 重排）。
2. **按 shape 找 graph key**：Qwen 系列 key 是 `S`，InternVL key 是 `(B, S)`。
3. **首见 key 时 capture**：分配稳定地址缓冲区并 `torch.cuda.graph(...)` 捕获。
4. **后续 replay**：只 `copy_` 更新输入内容，调用 `graph.replay()`。

---

## 3. Qwen2.5-VL：窗口注意力 + CUDA Graph

### 3.1 图外预处理

`forward_with_cuda_graph()` 先完成 patchify、窗口索引、位置编码、`cu_seqlens` 计算，再进入 runner：

```python
# python/sglang/srt/models/qwen2_5_vl.py
x = self.patch_embed(x)
window_index, cu_window_seqlens = self.get_window_index(grid_thw)
reverse_indices = permute_inv(window_index)
...
position_embeddings = (emb.cos(), emb.sin())
...
return self.cuda_graph_runner.run(
    x=x,
    position_embeddings=position_embeddings,
    cu_seqlens=cu_seqlens,
    cu_window_seqlens=cu_window_seqlens,
    output_indices=reverse_indices,
)
```

这里 `output_indices=reverse_indices` 用于把窗口重排后的输出复原到原 token 顺序。

### 3.2 graph key 和稳定地址缓存

`ViTCudaGraphRunner` 用 `S`（序列长度）作为 key：

```python
# python/sglang/srt/multimodal/vit_cuda_graph_runner.py
def _get_graph_key(self, x_3d: torch.Tensor) -> int:
    # x_3d: [S, B, H], B=1
    return x_3d.shape[0]
```

并且按 key 缓存稳定地址张量（输入、workspace、输出、seqlens）：

```python
self.block_input: Dict[Hashable, torch.Tensor] = {}
self.block_ws: Dict[Hashable, torch.Tensor] = {}
self.block_output: Dict[Hashable, torch.Tensor] = {}
self.cu_full_len: Dict[Hashable, torch.Tensor] = {}
self.cu_window_len: Dict[Hashable, torch.Tensor] = {}
```

这是 CUDA Graph 的核心要求：**replay 时地址不能变，只能改内容**。

### 3.3 捕获内容：blocks + merger（+可选 deepstack）

capture 在 `_create_graph()` 中完成：

```python
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    for layer_num, blk in enumerate(vit.blocks):
        ...
        y = blk(..., cu_seqlens=cu_seq_len_ws, output_ws=self.block_ws[graph_key])
    main_out = vit.merger(y)
    self.block_output[graph_key] = main_out
```

其中 `cu_seq_len_ws` 的格式会跟随 mm attention backend：

```python
if override_backend == "triton_attn":
    cu_seq_len_ws = [cu_seqlens_now, cu_seqlens_kk_now, max_len]
elif override_backend == "fa3":
    cu_seq_len_ws = [cu_seqlens_now, max_len]
```

### 3.4 replay 路径

replay 时只做几件事：

```python
self.block_input[graph_key].copy_(x_3d)
self.block_graphs[graph_key].replay()
out = self.block_output[graph_key]
if output_indices is not None:
    out = out.index_select(0, output_indices)
```

---

## 4. Qwen3-VL：在同一个 runner 中加入 deepstack

Qwen3 的 `forward_with_cuda_graph()` 与 Qwen2.5 类似，但使用 `rotary_pos_emb_cos/sin`，且没有 window seqlens：

```python
# python/sglang/srt/models/qwen3_vl.py
return self.cuda_graph_runner.run(
    x=x,
    position_embeddings=None,
    rotary_pos_emb_cos=rotary_pos_emb_cos,
    rotary_pos_emb_sin=rotary_pos_emb_sin,
    cu_seqlens=cu_seqlens,
    cu_window_seqlens=None,
    output_indices=None,
)
```

deepstack 也被 capture 进图（而不是图外拼接）：

```python
# python/sglang/srt/multimodal/vit_cuda_graph_runner.py
if layer_num in self._deepstack_visual_indexes:
    deepstack_out = self._deepstack_merger_list[deepstack_capture_idx](y)
    deepstack_outs.append(deepstack_out)
...
main_out = vit.merger(y)
if deepstack_outs:
    self.block_output[graph_key] = torch.cat([main_out] + deepstack_outs, dim=1)
```

---

## 5. InternVL：`InternViTCudaGraphRunner`（key = `(B, S)`）

InternVL 的输入是 `[B, S, H]`，所以 key 不是单个 `S`，而是 `(B, S)`：

```python
# python/sglang/srt/multimodal/internvl_vit_cuda_graph_runner.py
def _graph_key(self, x: torch.Tensor) -> Tuple[int, int]:
    return (x.shape[0], x.shape[1])
```

首见 key 时，会先预热一次（触发懒初始化），再 capture：

```python
self.inp[key] = torch.empty_like(x).contiguous()
self.cu[key] = self._build_cu(B, S, device=device)
self.ws[key] = self._alloc_ws(B, S, H, device=device, dtype=dtype)
self.inp[key].copy_(x)
self._warmup_once(key)
self._capture_graph(key)
```

InternVL 在 encoder 里只对 `output_hidden_states=False` 走 graph 路径：

```python
# python/sglang/srt/models/internvl.py
if self.enable_cg and (not output_hidden_states):
    hidden_states = self.cuda_graph_runner.run(hidden_states)
    return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=None)
```

---

## 6. 与 VisionAttention backend 的耦合点

在 `SGLANG_VIT_ENABLE_CUDA_GRAPH=1` 时，视觉 attention backend 对参数格式有额外约束。

以 Triton backend 为例：

```python
# python/sglang/srt/layers/attention/vision.py
if envs.SGLANG_VIT_ENABLE_CUDA_GRAPH.get():
    if "output_ws" not in kwargs:
        raise RuntimeError("output_ws should be prepared for cuda-graph mode")
    if not isinstance(cu_seqlens, list):
        raise RuntimeError("cuda-graph mode cu_seqlens should be a list")
```

即 runner 需要提前准备稳定输出 workspace，并传入 list 结构的 `cu_seqlens` 元数据。

---

## 7. 关键约束（为什么这么做）

1. **shape 维度静态化**
   - Qwen runner 按 `S` 分图，InternVL 按 `(B,S)` 分图。
2. **地址稳定**
   - 输入、输出、workspace、seqlens、rotary 缓冲区都持久缓存，replay 只做 `copy_`。
3. **seqlens 分段模式必须匹配**
   - 即便 `S` 一样，若 `cu_seqlens` 分段语义不同，attention 分块也会错。
4. **backend 限制**
   - 当前 runner 里仅支持 `triton_attn` / `fa3`（不匹配会直接抛错）。
5. **显存换延迟**
   - 不同 key 越多，图缓存和私有内存池开销越大。

---

## 8. 启用方式

### 8.1 仅启用 ViT CUDA Graph

```bash
SGLANG_VIT_ENABLE_CUDA_GRAPH=1 \
python3 -m sglang.launch_server \
  --model Qwen/Qwen3-VL-8B-Instruct
```

### 8.2 与 Piecewise CUDA Graph 同时启用

```bash
SGLANG_VIT_ENABLE_CUDA_GRAPH=1 \
python3 -m sglang.launch_server \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --enable-piecewise-cuda-graph \
  --piecewise-cuda-graph-max-tokens 4096 \
  --piecewise-cuda-graph-compiler eager
```

---

## 9. 当前已知支持模型

- Qwen2.5-VL
- Qwen3-VL
- InternVL（本分支代码已接入 `InternViTCudaGraphRunner`）
