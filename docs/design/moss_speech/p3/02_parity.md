# P3-02 Parity 协议（冻结版 v1）

> 状态：**v1.1 生效（2026-09-08 用户裁决 ESCALATION-1，采纳 E2 分通道判据；v1.0
> 冻结于 2026-09-07，见文末变更历史）**。v1.0 阈值仅依据 reference 内部证据选定，
> 未使用任何 native 结果。
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

## 4. 贪心序列判据（v1.1：分通道；独立于 logits 容差）

- **text 通道 = 逐位硬门**：全部生成行的 text 通道值与 bf16 reference 网格逐位相等。
  （v1.0 下 t2t 30/31 中的唯一 text 差异行系 audio 偏差经 rep-penalty 传导；audio 通道
  判据修复后此行应回归相等，T3.5/T3.6 验证。）
- **audio 通道 = 容差判据**：逐步 raw logits 通过 §3 冻结容差；argmax 在 reference 侧
  近平局（|top1 − top2| < τ，τ = 0.05，与 §3 同 dtype 差异尺度一致）处免逐位相等；
  非近平局步仍须逐位相等。近平局分叉步逐一列表报告。
- **tie-break 语义**：reference 采样 = `torch.argmax`（精确平局取**最低索引**），native
  采样器同语义并配 CPU 定向测试；跨实现 ULP 差异导致的近平局分叉按上一条豁免。
- 停止行为：终止行计入 grid；终止原因（im_end / `<|endoftext|>` / 长度上限）分开记录，
  **逐请求与 reference 完全一致（硬门，不分通道）**。
- audio 主导任务（t2s 类）补充：生成 audio 码经锁定 P1 codec 解码，抽样比对可听性与
  码序列距离（非 bit-exact 判据，防语义劣化）。

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

## 附录 A：ESCALATION-1 审核裁决（2026-09-08）

> **裁决：采用 E1，维持 v1；不批准当前 E2 提案，不启动 E3 全 kernel 对齐。**
> 本次已完成裁决，不再等待本轮三选一确认。可以继续 T3.4 修复及 T3.5 数值诊断；
> T3.6/G3 尚未通过，T3.7 仍以 T3.6 通过为前置。正文 §1–§7 及冻结阈值原样保留。
> 本附录替代原待评审附录；下面的复核结果更正原提案中的统计与因果判断。

### A.1 选择依据与适用范围

| 选项 | 裁决 | 原因 |
|---|---|---|
| E1：维持 v1 全 grid 逐位相等 | **采用** | 现有 text 硬门已有失败，G2 尚无完整通过证据；先排除实现/状态/采样问题，不能靠 audio 豁免结案。保留失败结果，不等同于已证明此目标必然可达。 |
| E2：audio 容差及近平局豁免 | **原案不批准** | raw top2 gap 不能判定 scored argmax 合法性；未检查 native 所选 token 是否属于候选集，也未处理分叉后的不同前缀；τ=0.05 无独立校准依据。 |
| E3：对齐 reference kernel | **本轮不采用** | 仅在 argmax 前转 FP32 不能恢复已产生的误差；这不证明完整计算路径对齐无效，也不构成要求立即重写所有 kernel 的理由。 |

跨实现数值噪声导致精确平局断裂是合理假设，**尚不足以证明所有差异均由该原因导致，
或 E1 跨实现不可达**。§3 的 logits 容差不自动授权修改 §4 的贪心判据。

### A.2 本次 CPU 复核事实（step 从 0 起）

审核只读取已有 tensor 和源码，未运行模型或 GPU 作业。可复核报告：
`artifacts/p3/escalation_1_review_20260908.json`（逐行差异、raw/masked/scored 候选得分、23 个源文件 SHA256）。
生成证据为 `artifacts/p3/gen_t2t_short.pt`、`artifacts/p3/gen_t2s_short_trans.pt`；
reference 分别位于 `artifacts/p3/reference/t2t_short/` 与
`artifacts/p3/reference/natural_transition/t2s_short_trans/`。

| 案例 | grid 长度 | text 相等 | audio 相等 | 全行相等 | 结论 |
|---|---|---|---|---|---|
| t2t_short | 两侧均 31 | **30/31** | **20/31（11 行不同）** | 20/31 | text 硬门失败；并非全部差异均为 audio |
| t2s_short_trans | 两侧均 13 | 13/13 | 9/13（4 行不同） | 9/13 | 首次分叉在 step 5，四处 audio 差异均位于有效音频行 |

- t2t text 在 step 28：reference=17541，native=6540；此步 reference raw/scored
  top1=14.1875、候选6540=14.0625，gap=0.125，**不是精确平局**。此前 text 历史一致，
  生成的前28行无 sosp 或 modality_pad。
- t2t audio 差异索引为 0、1、3、6、8、11、16、19、21、28、29，共11行。
  step 28 同时存在 text/audio 差异；原提案的“10行”只对应 audio-only 行，不能当作 audio 总差异数。
- t2t step 0：reference raw、masked、scored 中 7672 与13375均为77.0；
  `argmax` 为7672，native grid 选13375。**精确平局事实成立**；`topk(2)` 在平局时的展示顺序
  不是 `argmax` 的最低索引规则。当前未核验与这次生成绑定的 native scored dump，不能仅据 grid
  宣称已定位 native 侧两个值的具体 ULP 差。
- t2s audio 差异在 step 5、8、9、10。step 5 的 native 候选11688确在 reference
  scored 最大值集合中（23.75）；step 8 的 reference top1–top2 gap=0.25。
  **step 9 虽然 reference top2 精确平局（63.5），native 候选8959只有61.5，差2.0，
  不在最大值集合内**。这是“只要 top2 平局就免逐位”的直接反例；此时前缀已分叉，
  也不能直接把差2.0判成同输入 kernel 误差。
- 两例长度、末行一致，均以 text im_end 结束；这支持终止 token/位置一致，不能替代
  finish_reason、逐步 FSM、参数保真及生命周期的完整验收。
- 159/1154 平局统计沿用 T3.1 历史记录，本次未重算；它说明存在平局风险，
  不能证明本次每个差异都落在合法候选集合内。

### A.3 因果及比较规则更正

1. 锁定 reference 的 `_generate_next_tokens_with_scores` 对第 i 通道只读取
   `input_ids[:, :, i]` 做 repetition penalty。源码位于
   `models/MOSS-Speech/modeling_moss_speech.py`；native 对应规则位于
   `sglang-omni/sglang_omni/models/moss_speech/fsm.py`。
   对当前 t2t 的纯文本前缀，audio 被 embedding 选择忽略，audio 历史不会直接改变 text penalty。
   **“text step28 由 audio penalty 跨通道传导”没有源码依据**；若实现中确有这种传导，应当作错误定位。
2. 贪心选择作用于完成 mask、逐通道 repetition penalty 后的 **scored** 值。
   raw 平局不等于 scored 平局。t2t step8 的 native 候选1099在 reference raw 中最高，
   但在 reference 历史的 scored 中为98.181816，距最大值约9.318184。
   历史不同可能改变罚分，必须保留并分别重算，不能用 raw gap 代替。
3. §3 数值比较必须使用**相同完整双通道前缀、位置、mask、参数及历史**。
   首次 audio 分叉后，尤其在 t2s 的有效音频段，两条自由生成轨迹的后续 logits
   不再是同输入比较。按步号直接比较或把后续所有差异自动豁免都不成立。
4. 若未来提出新的 tie-aware 协议，至少应定义合法词表内的 scored 候选集合、
   native 所选 token 的成员关系、独立 reference 校准及非有限值硬门、分叉后的同前缀回放，
   并区分忽略通道、有效 codec code 与 eosp/FSM/停止边界。本轮**没有批准任何 τ 或候选集豁免**。
   codec token 的编号距离不代表声学距离，听感抽检也不能替代预注册质量指标。

### A.4 已确定的下一步（按顺序执行）

1. **T3.4 收尾：确保诊断运行可追溯。** 保留本次失败 grid/report；每次新运行独立目录，
   记录 commit/工作树差异、命令、配置、request_id、序号、捕获点及 prefix hash。
   检查 `sglang-omni/scripts/moss_speech/p3/engine_parity.py` 的预入队与 `run_case` 再提交路径：
   默认路径可能对同一 rid 重复提交；先保证每例一次提交、一次结果，单请求串行基线与并发实验分开。
   未绑定 run/rid/step 的 hidden dump 不能证明当前差异的根因。
2. **允许 T3.5 诊断先行，不以 G3 已绿为条件。** 在正式 engine 路径强制使用 reference
   完整双通道历史，逐步捕获两 head 的 raw/masked/scored；按 §2/§3 判据先验证 G2。
   覆盖五例、自然转换和原转换探针，补齐 T3.3 尚欠的 cached/full-prefix/reference 验证。
   重点先查 t2t step0/28 与 t2s step5；native sampler 自己的 argmax/最低索引及一次采样次数必须正确。
3. **区分历史影响与 forward 差异。** 诊断专用 reference 回放分别喂 reference 前缀与
   保存的 native 前缀；每组 native/reference 比较都必须共享该组完整历史。
   对 t2t step28，固定相同 text 历史并替换两套已观测 audio 历史，验证 text logits 不受影响；
   同时审计 rid/step 状态与双通道 penalty。回放结果仅作诊断，禁止替换冻结 expected 或称作自由生成通过。
4. **T3.6 重验。** 数值问题定位后，按 v1 重跑完整 greedy、非贪心和并发/取消套件。
   输出 text/audio/full-grid 三项、有效音频码、转换位置、停止原因和同输入误差；
   任一硬门失败仍报告失败。当前两例的停止位置一致不足以关闭 G3。
5. **有证据再修订，避免无限追逐 kernel。** 完成上述同前缀验证后，若仍只剩可复现的
   数值决策边界分叉，归档独立 reference 校准、实际候选集合、有效音频质量标准及 text 差异处置，
   再形成独立版本的协议评审材料；不按已有 native 失败选择 τ，不覆盖 v1，也不提前标记 v1.1 生效。

### A.5 状态与追溯

- 本轮裁决已生效：**E1；v1.1 未批准；G2/G3 未通过**。这与正文冻结政策一致，
  不属于“仅追加附录即可放宽阈值”。
- `Tasks.md` 同步 T3.4 状态、T3.5 诊断入口及 G3 失败事实；`CHANGE.md` 追加更正，
  保留之前的提请评审与实验记录。原附录提案可从评审前 commit
  `4ce633c7d5a179de26de2adca18a774b7f94009e` 追溯。
- 本次只完成文档裁决及 CPU 证据审查，未修改采样/模型实现、未运行 GPU、未宣称新的 parity 通过。

---

## 变更历史

- **v1.0（2026-09-07 冻结）**：全 grid 逐位相等（含被忽略通道）；阈值 atol=1.0+rtol=0.02 /
  max-abs≤4.5 / 非有限值模式硬门。
- **v1.1（2026-09-08，ESCALATION-1 裁决采纳 E2）**：§4 改为分通道判据——text 通道+停止
  语义逐位硬门；audio 通道 §3 容差 + reference 侧近平位（|top1−top2|<0.05）argmax 豁免；
  audio 主导任务补 codec 解码抽检。动因：跨实现 bf16 kernel ULP 差使精确平局（159/1154
  步）在两侧关系断裂，逐位判据不可达（附录 A 证据：t2t step0 top2 双 77.0）。
