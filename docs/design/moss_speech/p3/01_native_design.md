# P3-01 Native 设计：参考语义、1-D 表示、KV 记账与引擎形态

> 状态：T3.1 交付（代码级核实 + BF16 reference 校准）。本文档所有参考语义均引用锁定 revision
> （HF `cff025bb` / GitHub feat/docs@`1ea408a`）的源码行号；校准数据见
> `artifacts/p3/reference/_manifest.json` 与 P3-02 协议。

## 1. 输入与范围

- 消费 P2 冻结产物：canonical grid/mask（五例 golden）、参数优先级与 effective_seed、
  `MossSpeechState` wire 契约、StageConfig（terminal_stages_fn/GPU 份额/route_fn）。
- 消费 P0：446 tensor 清单（`artifacts/p0/weight_shapes.txt`）、trace 结论（C1–C4）。
- 新基线：`artifacts/p3/reference/`（显式 BF16；P0 导出路径未指定 dtype、实际为 FP32，
  故 P0 logits 不能作为同计算精度基线——本阶段已在 manifest 记录 dtype 证据）。

## 2. 参考语义（逐条代码核实）

### 2.1 结构（modeling_moss_speech.py）

| 组件 | 事实 |
|---|---|
| 嵌入 | `embed_tokens(151680×4096)` + `audio_embed(16512×4096)`；checkpoint 中 4 个大矩阵（两 embed + 两 head）为**独立值**（实测 maxdiff 0.56/0.22），`_tied_weights_keys` 声明无效——native 保留 4 独立张量，禁止 tie |
| 干线 | `shared_block` 32 层（start_idx=0），随后 **两个 tail 各 4 层**并行作用于同一 trunk 输出 |
| 注意力 | 32 Q head / 8 KV head，head_dim 128，rope_theta 1e6；RoPE 频率在 `autocast(enabled=False)` 下以 fp32 计算（L455–462） |
| 头 | `text_lm_head`、`audio_lm_head`（无 bias），forward 返回 `logits_all=(text, audio)` 元组 |
| 轴序 | 接口 `(B, L, 2)`（channels-last），forward 内 `transpose(1,2)` 成 `(B, 2, L)`；`input_ids.dim()==2` 时 reshape `[B,2,-1]` |

### 2.2 嵌入选择（纯 token 函数，与 FSM 状态无关）

`MossSpeechModel.forward` L566–576：对每个 grid 行——

```
text_ids != modality_pad(151667) → 用 text_embed(text_token)
否则                            → 用 audio_embed(audio_channel_token)   # code/eosp/audio_pad(512) 均查 audio_embed
```

文本行的音频通道值与音频行的文本通道值**不参与**嵌入。因此行→嵌入是有损无关的纯函数，
这是 1-D 折叠表示（§3.1）无损性的依据。

### 2.3 FSM 与采样顺序（`_sample` L772–917）

每个生成步 k（1-based `generating_length`，forward 前）：

1. `prepare_inputs_for_generation`（HF 默认）；
2. **模态切换**（读上一 appended 行，forward 前）：
   `text-mode & text_ch==sosp(151646)` → audio；`audio-mode & audio_ch==eosp(16384)` → text；
3. forward → `logits_all`；
4. `next_token_logits = logits_all[:, -1, :].clone().float()`；施加音频约束：
   `audio_ch[16385:] = -inf`（恒定）；`audio_ch[16384]=-inf` 当 `generating_length < min_new_tokens`；
5. `next_token_scores = processor[i](input_ids[:, :, i], logits)`（每通道独立；
   P0/P3 greedy 配置下 processor = repetition penalty 1.1；warper 仅在 do_sample=True 时构造）；
6. 每通道独立 `argmax`（greedy）或 `softmax→multinomial`；
7. **音频模式覆写**：`current_modality==1` → 该行 text_ch 置 `modality_pad(151667)`
   （文本模式的 audio 采样值保留在 grid 中但被 §2.2 规则忽略）；
8. append 行；`stopping_criteria(input_ids[:, :, 0], scores)`：任一 MIMOStopper 命中
   （text_ch 末 token == `<|endoftext|>` 或 `im_end(151645)`）即停止，**该行保留在输出 grid 中**。

初始模态（L834–840）：prompt 末行 `text_ch==modality_pad` → audio；`audio_ch==audio_pad(512)` → text
（先判 audio 再判 text，同真时 text 生效；音频输出任务 prompt 以 `[sosp, audio_pad]` 结尾，
靠循环内 sosp 规则切到 audio）。

### 2.4 捕获点（三档，用于 parity 分层）

| 点 | 内容 | P0 文件对应 |
|---|---|---|
| raw | 两 head 最后位置原始输出（未 mask、未升精度） | 无（本阶段新增） |
| masked | raw.float() + §2.3-4 音频约束 | `logits_first_steps.pt`（"logits"） |
| scored | masked + 每通道 processor（rep penalty） | 无显式存储（"scores" 未落盘） |

P0 的 16×127 个 -inf 列即 masked 档的音频约束痕迹；与 native 比较时须对齐到同一档
（见 P3-02 协议 §2）。

### 2.5 KV 语义（P0 trace C1–C4 + 本阶段复核）

三个 `DynamicCache`：shared/text/audio，**等长**、逐位置同步增长；两个 tail 每步都执行、
都读写自己 cache 的同一位置历史。文本 tail 在音频段之后的恢复文本生成时读取的是**自己在音频
位置写入的 KV**（tail 对全序列每位置都有条目），不是跨 tail 读取。native 按 40 层实例记账
（§3.3）。

## 3. Native 映射设计

### 3.1 1-D 调度表示（D3 采用双通道语义 + 每 grid 步一个 KV 位置）

- scheduler/input_ids 一维：第 i 位 = grid 第 i 行的**选中 token**
  （`text_ch[i] != 151667` ? `text_ch[i]` : `audio_ch[i]`），配 `is_audio_row[i]` 标志。
- 无损性：嵌入是行的纯函数（§2.2）→ (selected_token, is_audio) 完全决定嵌入输入；
  KV 每行一个位置（§2.5）→ 位置计数一致；双 head 输出由模型产生，不受折叠影响。
- 不做 audio id 偏移合并词表（备选仅在逐 logits 等价证明后考虑，plan §6.2）。
- `custom_prefill_forward`：从 request payload 的双通道行列表构建
  `embed_tokens(sel[~audio]) / audio_embed(sel[audio])` 拼接的 `inputs_embeds`。
- `before_decode`：把上步**双通道**采样结果写回（audio 行：selected=audio code；text 行：
  selected=text token），文本模式被忽略的 audio 采样值保存在请求状态中用于输出 grid 重建。
- `post_decode`：双 head logits → §2.3-4 mask → per-channel processor → 采样 → §2.3-7 覆写
  → 返回 selected token 给 scheduler；完整双通道行存入请求状态。
- `post_process_outputs`：由请求状态重建 `(L, 2)` output_grid（含被忽略通道的真实采样值），
  与 reference 全 grid 逐位可比（T3.6 验收口径）。

### 3.2 FSM 状态机（runner 内，逐请求）

状态 `mode ∈ {text, audio}`，转换由上一 appended 行触发（forward 前评估，§2.3-2）：

| 当前 | 上一行条件 | 动作 |
|---|---|---|
| text | text_ch==sosp | → audio |
| audio | audio_ch==eosp | → text |
| audio | （默认） | 本行 text 覆写 modality_pad |
| text | （默认） | audio 采样值保留待重建 |

停止（text 通道）：`<|endoftext|>` 或 `im_end`（该行计入 grid）；加上 max_new_tokens。
`min_new_tokens` 只作用音频通道 eosp 掩码（§2.3-4），不额外拦停。

### 3.3 40 层 KV 记账

- 层索引：shared 0–31、text tail 32–35、audio tail 36–39；两 tail 的 K/V 物理槽不别名。
- HF `num_hidden_layers=36` 保持语义保真（hf_config 不改 checkpoint 语义）；运行时以
  ModelConfig/layer_info 提供 40 层给 allocator——具体机制（覆盖 layer 计数 vs 自定义
  layer_info hook）在 T3.2 实现并测试，不修改 checkpoint 文件。
- 理论容量：`40 × 2(K/V) × 8(KV heads) × 128 × 2B(bf16) = 163,840 B = 160 KiB/token`；
  T3.3 按实际 slot/page 对齐复验。
- V1 关闭 Radix Cache（plan §6.1），禁止多模态 prefix alias。

### 3.4 采样与参数

- per-request：temperature/top_p/top_k/repetition_penalty/seed（P2 effective_seed 链）；
  processor 顺序 = §2.3-4/5（mask→rep penalty→[采样时 warpers]）。
- 音频通道硬约束：`[16385:]` 恒禁；`eosp` 受 min_new_tokens 门控。
- 禁止从 batch 首请求读默认值（plan §6.2）；显式 seed 与 request_id 无关。
- 贪心路径无 RNG 消耗；采样路径的 RNG 派生按请求隔离（native 侧同请求单独运行与交错运行
  必须一致；跨实现 bit-exact 不承诺，见 P3-02 §4）。

### 3.5 引擎形态（D1）

- 基类：`sglang_omni/scheduling/engine_factory.py` 的公共 builder（生成生命周期 +
  `build_sglang_server_args` 路径，与 P2 工厂预检同源）；MOSS-Speech 子类仅提供：
  模型 runner 装配（`make_model_runner`）、request/result adapters（`make_adapters`）、
  生成默认值与校验（`generation_defaults`/`validate_*`）。
- 显式关闭：CUDA Graph、torch.compile、radix、量化、TP>1（V1 边界，禁止静默打开）。
- 注册：`_register_omni_model` 字典加 `MossSpeechForCausalLM → sglang_omni.models.
  moss_speech.sglang_model` 条目（加性补丁、独立 commit、T3.2 首次 GPU 加载前落地；
  注册失败不得仅 warning 跳过——新进程 registry 断言导入成功）。

### 3.6 请求/结果 adapters 与状态所有权

- request adapter：`MossSpeechState`（P2 wire）→ scheduler request；携带双通道 prompt 行、
  effective_seed、显式参数、输出模态；按输出模态分 queue（V1 不混模态 batch）。
- result adapter：输出完整 output_grid（双通道）+ 终止原因 + 长度；voice/routing 字段原样回传。
- 请求状态（owner = AR runner）：FSM mode、双通道历史（供 rep penalty 与 grid 重建）、
  seed/参数、步数、slot 映射；allocate/reset/free 协议，重复 cleanup 幂等，交错/复用后不留
  前请求数据（T3.4 CPU 测试 + T3.6 GPU 交错验证）。

## 4. 权重映射（T3.2 实现口径）

- 446 源 tensor 全量对账（`artifacts/p0/weight_shapes.txt`）；native 允许 QKV/gate-up 合并
  等表示变化，但每个源 tensor 必须有消费目标、slice/拼接/转置与 dtype 策略记录。
- 断言：`embed_tokens/text_lm_head/audio_embed/audio_lm_head` 四者加载后互不相等
  （防意外 tie，§2.1）。
- 加载器测试用 meta/小尺寸模型或 sentinel 张量验证切片与命名（不在登录节点构建 17GiB 模型）。

## 5. 已识别风险

| 风险 | 缓解 |
|---|---|
| 40 层 allocator 机制与 SGLang 版本耦合 | T3.2 定型 + 新进程 registry 断言；机制缺口另拆最小补丁（D4 升级路径） |
| 文本模式被忽略的 audio 采样值需逐位复现（全 grid 相等验收） | §3.1 post_decode 保留双通道行；greedy 下 audio 通道也是 argmax，确定性可复现 |
| rep penalty 需每通道完整历史（含覆写后的 151667） | 请求状态保存双通道历史，processor 输入与 reference 相同序列 |
| bf16 kernel 顺序差异 | P3-02 预注册容差（fp32-vs-bf16 差距为参考尺度，不依 native 结果定阈值） |

## 6. 开放项（移交 T3.2+）

- 40 层 runtime 配置的具体 hook（覆盖点）在实现时定稿并记录于此文档附录。
- `prepare_inputs_for_generation` 的 position_ids/cache_position 细节（left-pad 时位置计算）
  在 T3.5 left-padding 用例中验证。
