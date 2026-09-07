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
