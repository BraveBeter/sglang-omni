# P3-02 Parity 协议（冻结版 v1）

> 状态：T3.1 冻结（2026-09-07）。阈值仅依据 reference 内部证据（bf16 重跑确定性、
> 同 dtype 核路径差异、fp32↔bf16 dtype 差距）选定，**未使用任何 native 结果**。
> 修改政策：只允许收紧；任何调整须保留旧版本与失败结果并记录评审结论。

## 1. 基线定义（artifacts/p3/reference/）

| 项 | 值 | 证据 |
|---|---|---|
| 计算/参数 dtype | **BF16**（全参数，`torch_dtype=torch.bfloat16` 显式加载） | manifest `param_dtypes=['torch.bfloat16']` |
| 注意力实现 | sdpa（transformers 4.57.1 自动选择） | manifest `attn_implementation` |
| TF32 | matmul=**False**、cudnn=True（仅影响 conv，不触及本模型 matmul/attention） | manifest env |
| 权重 | 与 P0 完全一致（4 分片 SHA256 复验全匹配） | manifest `weights_match_p0=true` |
| 输入 | canonical grid 与 P0 逐元素相等（processor 确定性） | manifest `canonical_input_matches_p0` 全 true |
| 生成 | greedy、seed 0、rep=1.1、max_new=200、min_new=0、MIMOStopper(`<|endoftext|>`=151643 / `im_end`=151645) | 脚本参数 + manifest stop_ids |
| 确定性 | 同机重跑：tokens 逐位相等、step0 raw max-abs = 0.0 | `_determinism.json` |

**P0 fp32 fixtures 的角色降级**：bf16 与 fp32 贪心网格在 t2t 第 0 行即分歧（其余案例第 2–7 行），
P0 网格与 P0 logits 仅作 dtype 差距证据与历史对照；**native 贪心/bitwise 目标一律为
`artifacts/p3/reference/<case>/tokens_grid.pt`（bf16）**。

## 2. 捕获点对齐（三档）

| 档 | 定义（p3/01 §2.4） | 用途 |
|---|---|---|
| raw | 两 head 最后位置原始输出（bf16 计算→fp32 落盘），全有限 | **主比较档**（native 同点捕获） |
| masked | raw + 音频约束（audio[16385:]=−inf；P3 基线 min_new=0 故 eosp 未掩） | 与 P0 存储档对齐时使用 |
| scored | masked + 每通道 rep penalty | 采样行为核对 |

比较规则：raw↔raw（native 未加 mask 前）；若消费 P0 masked 档，native 必须施加完全相同的
音频约束后再比。**先验非有限值集合**：reference raw 全有限；masked 的 audio[16385:] 恰为
−inf 且其余全有限。任何 NaN、或 −inf 位置/符号不匹配 → **硬失败**（禁止 nan_to_num 清洗）。

## 3. 冻结阈值（logits，分 head 分步）

参考尺度（全部来自 reference 内部，与 native 无关）：

| 证据 | 数值 |
|---|---|
| reference \|raw\| 幅值 | max 114.5、mean 9.25（t2t_short） |
| 同 dtype 核路径差异（cached vs 全前缀重算，4 探针） | **max-abs ≤ 1.125**，argmax 全部一致 |
| dtype 差距（fp32 ↔ bf16，t2t 全保留步） | max 43.5、p99.9 36.3；组合界超限占比 4.6% |

**预注册判据（native vs bf16 reference，逐步、分 head）**：

1. 非有限值模式精确匹配（§2）——硬门。
2. 组合界：`|δ| ≤ atol=1.0 + rtol=0.02·|ref|`，每步每 head 超限元素占比 ≤ **1e-4**。
3. 附加上限：每步每 head max-abs ≤ **4.5**（= 同 dtype 核路径差异上限 1.125 × 4）。
4. 判据 2/3 任一超限即失败；先定位输入/权重/KV/数值路径，**不得放宽**。
   （dtype 级差距在该组合界下超限 ~4.6%，同 dtype 差异距界约两个数量级——阈值有效区分
   「实现错误」与「合法核路径噪声」。）

## 4. 贪心序列判据（独立于 logits 容差）

- **全 grid 逐位相等**（含被忽略通道：文本模式下保留的 audio 采样值、音频模式下被覆写为
  151667 的 text 通道），对照 bf16 reference 网格。
- **tie-break 语义**：reference 采样 = `torch.argmax`（精确平局取**最低索引**）。基线统计：
  1154 通道步中 **159 步存在精确平局**（audio 通道 ~14%、text 通道偶发）——native 采样器
  必须复现「平局取最低索引」，并配 CPU 定向测试（构造平局向量断言选择）。此为全 grid
  相等的必要条件。
- 停止行为：终止行计入 grid；终止原因（im_end / `<|endoftext|>` / 长度上限）分开记录。

## 5. 随机采样判据（T3.6）

- 不承诺跨实现同 seed token bit-exact（HF 与 native RNG 算法可不同）。
- 必须验证：分布/过滤规则（mask→rep penalty→warper 顺序、[16385:] 禁采样、min_new 的
  eosp 门控）与 **native 自身确定性**：同请求单独运行 vs 交错运行结果完全一致。
- seed 派生与 batch 顺序/request_id 解耦（显式 seed 时）。

## 6. 转换覆盖用例（T3.1 交付，供 T3.3/T3.5 消费）

| 用例 | 位置 | 覆盖 |
|---|---|---|
| 自然完整转换 | `natural_transition/t2s_short_trans/`（13 步：10 音频行 → [151667, eosp] → [im_end, 3216]） | 音频段 + eosp→text + 立即终止，全步三档捕获 |
| 教师强制 text+sosp | `probe_t2t_short_text_sosp_8.pt` | sosp 切 audio 的下一步行为 |
| 教师强制 audio+eosp | `probe_t2s_cn_audio_eosp_{30,60}.pt` | eosp 切 text 的下一步行为（text argmax=im_end） |
| 教师强制 text EOS | `probe_t2t_short_text_eos_10.pt` | 停止语义 |
| cached↔fresh 等价 | 上述探针双路径 | 同 dtype 核路径差异证据（§3） |

## 7. 版本与追溯

- manifest：`artifacts/p3/reference/_manifest.json`（env/SHAs/dtype/attn/stop_ids/attempt 记录）。
- 分段落盘；step 文件命名 `step_%04d.pt`（raw/masked/scored 三键，168192 维拼接）。
- 保留策略：不覆盖 P0 原始产物；本基线为 native parity 唯一 expected 来源，
  禁止从 native 输出生成 expected。

---

## 附录 A（待评审）：G3 全 grid 判据的 audio 通道近平局问题 [ESCALATION-1]

> 状态：**2026-09-08 提请评审**；v1 正文（§4）原样保留，本附录不构成对冻结判据的修改。
> 评审通过前，T3.5/T3.6 的验收暂按 v1 执行并如实报告分通道结果。

### A.1 事实

T3.4 正式引擎路径（OmniScheduler 全链路）已达成：

| 案例 | 长度/停止 | text 通道逐位 | 全行逐位（text+audio） |
|---|---|---|---|
| t2t_short | 31/31 步，im_end 停止与 ref 一致 | **30/31**（唯一 text 差异行 28 由前期 audio 偏差经 rep penalty 传导） | 20/31 |
| t2s_short_trans | 13/13 步，自然转换+停止一致 | **13/13** | 9/13 |

全部差异行均为 **audio 通道**（t2t 10 行、t2s 4 行），且首因在 step0 即出现。

### A.2 根因

t2t step0 的 reference raw audio top2：**13375=77.0 与 7672=77.0（精确平局，fp32 存储）**。
两侧 greedy 均为 `torch.argmax`（平局取最低索引），但 native（sglang bf16 kernel 路径）
与 reference（HF bf16 kernel 路径）的 logits 存在 ULP 级差异，**平局关系在两侧不一致**
（native 侧两值非严格相等，次序翻转）→ argmax 分叉 → 该行 audio 值不同 → 经每通道
repetition-penalty 历史传导至后续步。

T3.1 基线统计：1154 通道步中 **159 步存在精确平局**（audio 通道 ~14%）。该风险在
P3-01 §2.3 已标注为「全 grid 相等的必要条件」，现实测坐实：**跨实现的 bf16 kernel
ULP 差使部分平局步不可判定一致**。

### A.3 判据影响与选项（待决策）

- **text 通道判据已达标**（t2t 30/31 中的 1 行差异亦源于 audio 传导；t2s 13/13）。
- 停止语义、长度、FSM、生成步数全部一致。

| 选项 | 内容 | 代价 |
|---|---|---|
| E1（维持 v1） | audio 通道仍要求逐位相等 | 需 bit-exact 复现 HF kernel（sglang 层不支持切换 norm/attention kernel 实现），实际不可达；T3.6 无法通过 |
| E2（分通道判据） | text 通道+停止语义保持逐位硬门；audio 通道改为「raw logits 落在 §3 冻结容差内 + argmax 在 |Δ|<τ 近平局处允许分叉（τ 按 §3 同 dtype 差异尺度定，如 0.05）」 | 修订协议 v1.1；需复核 audio 主导任务（t2s 类）的语义质量（audio 码经 P1 codec 解码后听感/码距） |
| E3（对齐 kernel） | native 侧对 logits→argmax 之前用 reference 完全相同的 fp32 重算路径 | 无法消除 forward 本身的 ULP 差异（平局在 raw 层面已断），无效 |

建议 **E2**，理由：平局处的分叉在 reference 自身的确定性重跑下是稳定的（同机 bit-exact），
但在跨实现下数学上无仲裁标准；容差判据（§3 已冻结）本就为「合法核路径噪声」设计，
近平局 argmax 分叉是该噪声的唯一可见后果；且 audio 通道在 text 模式行本就被 reference
自身忽略（其值不进入嵌入与停止判定），仅影响 rep-penalty 历史。

### A.4 若采纳 E2：v1.1 修订点

1. §4 贪心判据拆分：text 通道与终止行为=逐位硬门；audio 通道=逐步 raw logits 过 §3
   容差 + 近平局（|top1−top2|<τ 于 reference 侧）处 argmax 免逐位。
2. T3.6 验收：全 grid 逐位改为分通道报告；audio 主导任务补充 codec 解码一致性抽检。
3. 本附录升级为 v1.1 正文，v1 移入变更历史。
