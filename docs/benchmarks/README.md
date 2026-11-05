# 基准测试

延迟脚本测量单请求生成流程。输入使用固定随机种子生成的 token ID 张量，输出长度固定，因此每个测试用例都有确定的输入形状和相同的生成步数。

在仓库根目录运行：

```powershell
uv run python benchmarks/latency_benchmark.py --warmup-runs 3 --repeats 10
```

默认输入长度为 `32 64 256 512 1024`，每轮生成 32 个 token。默认报告为 `docs/benchmarks/generation-benchmark.md`，相邻的 `.json` 文件保存原始数据。可根据目标 GPU 的可用显存调整最大输入长度和输出长度。

报告模板只描述测量配置和指标口径，不在标题、文件名或元数据中写死 KV Cache 等模型实现状态。可通过 `--report-path` 指定实验文件名，或在运行后重命名并补充实现状态、优化内容及分析结论；归档时保留对应的 JSON 原始数据。报告记录代码 commit 和工作区是否有未提交修改，辅助定位实验版本。

模型实现变化时仍需检查 `_run_generation()` 是否调用了预期的推理路径。报告模板与实现状态分离，并不意味着现有生成循环会自动接入新增的缓存或调度接口。

Profiler 模式与正式基准测试分开运行：

```powershell
uv run python benchmarks/latency_benchmark.py --profile --prompt-lengths 512 --trace-dir traces
```

Profiler 模式会按配置预热，为每个输入长度采集一次 trace，不写入基准测试计时结果。Chrome Trace JSON 可用 Perfetto 或 Chrome tracing 查看。每份 trace 旁边的元数据文件记录了精确的输入 token ID 和运行环境。

计时范围和指标口径：

- 分词、输入传输到设备、文本解码不计入生成阶段计时；报告的负载范围是“设备上已有输入 token ID → 固定长度生成”。
- 生成墙钟耗时包含 Python 循环开销和最后的设备同步。墙钟吞吐等于生成 token 数除以这段耗时。
- 首 token 时间间隔和 ITL 不是端到端 TTFT。CUDA 模式使用当前流上的 Event；Event 间隔可能包含 Python 提交工作期间的设备空闲时间，并非 kernel 耗时之和。CPU 模式使用 `perf_counter`。
- 逐 token 埋点也会带来开销。Profiler trace 用于定位可能的瓶颈，性能对比使用未开启 Profiler 的基准运行。
- 报告包含 PyTorch 分配器的 allocated/reserved 峰值、各轮运行前基线，以及运行后的设备级剩余显存。脚本不会在测试用例之间清空缓存分配器，因此 reserved 数据受运行顺序影响，解读时应结合起始基线。
- 汇总结果给出重复运行的中位数及观测到的最小/最大值。每轮的 ITL 分位数使用线性插值；汇总表中的分位数是各轮分位数的中位数。比较改动时应保留 JSON 原始数据。

运行成功后再保存实际基线报告，不提交占位或未经验证的性能数字。
