# Phase 3 验收与交接（2026-09-08）

P3 的 T3.1–T3.8 已完成，G1–G4 通过。采用冻结 parity v1/E1；未放宽阈值、未豁免平局、未替换 reference expected。机器可读摘要为 `sglang-omni/docs/design/moss_speech/p3/gate_summary.json`，包含原始报告路径与 SHA256。P4 未开始。实现提交：`af4f076`（数值/状态/loader）、`cfca840`（正式接管与验收驱动）；设计/报告随后独立归档。

## 验收结果

| Gate | 结果 | 主要证据（工作区根相对路径） |
|---|---|---|
| G1 权重/KV | 446 个唯一来源，缺失/未知/重复来源和错误形状硬失败；40 层 BF16、80 块 K/V 不别名，163840 bytes/token；保留旧KV并真实复用35个槽后输出无污染 | `artifacts/p3/lifecycle_3858/report.json`；CPU loader/registry测试 |
| G2 logits/转换 | 六例707行，每步双head raw最大误差0；4个转换probe的cached/reference、fresh/reference、cached/fresh均在v1阈值内 | `artifacts/p3/validation_3857/report.json`、`artifacts/p3/probes_3846/report.json` |
| G3 生成/隔离 | 完整greedy grid逐位一致；真实同模态batch及重排、batch greedy raw逐位一致；非零温度、仅换seed、GPU mask/top-k、取消存活者/恢复、padding及全部资源回收通过 | `artifacts/p3/lifecycle_3858/report.json` |
| G4 正式接管 | 正式YAML/native AR文本与音频、两请求、AR prefill事件后取消/恢复通过；waveform hash等于冻结HF codes经P1 codec；正常及启动失败均有4个PID退出证据；P2组件/stub独立回归通过 | `artifacts/p3/takeover_3855/report.json`、`artifacts/p3/takeover_3849/report.json`、`artifacts/p3/p2_regression_3854/` |

| case | 完整grid行数 | 双head raw最大误差 | 停止原因 |
|---|---:|---:|---|
| t2t_short | 31 | 0 / 0 | stop |
| t2s_cn | 200 | 0 / 0 | length |
| s2t_cn | 153 | 0 / 0 | stop |
| s2s_cn | 200 | 0 / 0 | length |
| mixed_multiturn | 110 | 0 / 0 | stop |
| t2s_short_trans | 13 | 0 / 0 | stop |

CPU：110 passed，0 skipped（含本地锁定tokenizer、五例canonical和P1音频codes资产），日志 `artifacts/p3/cpu_release.log`。本轮Python文件通过 black 24.10.0、isort 5.13.2、ruff 0.11.10；`git diff --check`通过。格式化工具安装于独立 `.venv-format`，推理环境未升级。

最终结构实验的allocator有198370可用槽，加1个padding槽，总K/V容量32501104640 bytes。正式多进程YAML的0.72 AR预算运行中分配256435可用槽（实际可用显存与codec驻留相关）。这些是A800单次启动证据，不作为P4负载预算或24G承诺。

## 实现与数值修复

- 原生 Qwen3 模块/packed 权重、RadixAttention 分页KV和公共 builder/scheduler 保留；模型专用前向复现分离投影、RMSNorm BF16舍入及CPU初始化RoPE频率。GEMM和norm按请求固定形状，attention仍是真实batch。
- 完整prefill先投影再取末位；每个grid步一个KV位置；两tail均计算，不混用其KV。定向擦除音频历史位置的text-tail K，只改变text logits（3.875）；audio-tail K只改变audio logits（1.5），另一head均0。
- 双logits以 `(rid, phase)` 绑定并只消费一次；缺失反馈硬失败。FSM按reference顺序处理同一步SOSP/EOSP，采样长度取真实输出行数；随机生成器、显式参数和cleanup按请求持有。不同输出模态公平排队，同模态允许实际batch。
- `sglang-omni/sglang_omni/models/moss_speech/stages.py` 正式工厂已替换未实现边界。CPU preflight独立保留；P2 stub只用于明确标注的组件回归。
- 原始注册/40层runtime补丁为独立commit `fec09e5`；长度explicit标记依赖为 `d8926df`。不需要额外共享scheduler修改。

## 可追溯性与旧结果

`sglang-omni/scripts/moss_speech/p3/validate_native.py` 为当前验收入口，记录run/rid/step、前缀hash、工作树差异和来源hash。`validate_lifecycle.py`、`validate_probes.py`、`takeover.py` 分别记录实际batch、转换KV及多进程证据；只有probe/forced模式使用诊断teacher forcing，native自由生成/接管不注入fixture grid。

`artifacts/p3/reference_complete_3838/manifest.json` 补齐130步长案例捕获，原grid及已有raw/masked/scored逐位不变；原P0/P1/P2资产与BF16 reference未覆盖。旧 `artifacts/p3/kv_spike.json` 的总体pass=false，且旧slot_reuse没有物理重叠；不以它证明G1关闭，最终使用3858实际重叠及3846转换证据。旧逐元素权重诊断只覆盖layer0，最终446来源结论来自严格loader覆盖/shape断言与CPU sentinel测试，不声称旧报告做过全权重逐元素审计。

升级过程中的失败均保留：3839长位置分叉、3841无效缓存布局试验、3848并发greedy分叉、3850过快stub导致取消未发生、3851并发raw分叉。附录B记录数值根因；不存在放宽v1以通过的步骤。3857之后新增loader校验/报告与类型注解，合法权重的加载数值和单请求前向不变，3858再次完成真实加载与并发/槽位验证。

## 复验命令

以下命令从工作区根运行；GPU任务全部由Slurm分配A800，产物按job号写新目录。

```bash
sbatch scripts/p3_validate_native.sbatch --supplement-dir artifacts/p3/reference_complete_3838
sbatch scripts/p3_validate_lifecycle.sbatch
sbatch scripts/p3_validate_probes.sbatch
sbatch scripts/p3_takeover.sbatch
sbatch scripts/p3_takeover.sbatch --startup-failure
sbatch scripts/p3_p2_regression.sbatch
```

对应Python驱动均位于 `sglang-omni/scripts/moss_speech/p3/`；P2独立回归位于 `sglang-omni/scripts/moss_speech/p2/`。Slurm脚本设置离线模型环境、`TMPDIR=/dev/shm`、`OMP_NUM_THREADS=1` 和指定venv。CPU全量：

```bash
env -i HOME="$HOME" PATH=/usr/bin:/bin TMPDIR=/dev/shm OMP_NUM_THREADS=1 \
 MOSS_SPEECH_MODEL_DIR="$PWD/models/MOSS-Speech" \
 MOSS_SPEECH_FIXTURES_DIR="$PWD/sglang-omni/tests/fixtures/moss_speech" \
 MOSS_P1_ALIGNMENT_DIR="$PWD/artifacts/p1/alignment" \
 .venv-omni/bin/python -m pytest sglang-omni/tests/unit_test/moss_speech/ -q
```

## P4交接边界（本轮不执行）

1. 完整HTTP四模式与实际断开/取消/错误响应链，确认能力声明与API响应契约。P3验证正式pipeline，不宣称HTTP服务已验收。
2. native AR加codec的负载、吞吐、延迟、并发上限和显存预算；现仅证明两请求正确性。按请求GEMM/norm与torch_native attention可能牺牲性能，优化需重新验收parity。
3. 当前限定单A800、TP=1、AR/KV BF16、codec FP32、eager；禁用radix、graph、compile、chunked prefill、量化及remote config。正式context上限10240，P3实际最长prompt408、最长生成200，不宣称40K上下文实测。
4. 缺省采样0.6/0.95/20/1.1，显式temperature=0保持greedy；custom stop字符串及非零min_p明确拒绝。复现reference界面案例须传显式system消息；processor默认system规则未更改。
5. P5再做24G/CI/质量与权重条款，P6流式，P7/V2性能能力；本轮到P3归档结束。

发布状态：本轮结果已本地提交；向fork的推送被自动审批拒绝，需用户明确授权后才能执行。该限制不影响上述本地验收证据。
