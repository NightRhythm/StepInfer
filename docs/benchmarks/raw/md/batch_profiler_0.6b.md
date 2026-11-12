# Profiler Breakdown — Qwen3-0.6B-Base

## Setup

| Parameter | Value |
|:---|:---|
| Model | Qwen3-0.6B-Base |
| Parameters | 596,049,920 |
| Precision | bf16 |
| Device | NVIDIA GeForce RTX 5060 Ti |
| PyTorch | 2.9.1+cu128 |
| Python | 3.12.16 |
| KV Cache | Static pre-allocated (in-place slicing) |
| Batch Size | 4 |
| Prompt Tokens | 512 |
| Generated Tokens | 1 |
| Chrome Trace | None (disabled) |

---

## Top 15 Operations (Sorted by CPU Total / Dispatch)

| Op | CPU Total | Self CPU % | CPU Time Avg | # Calls | CUDA Mem |
|:---|---:|---:|---:|---:|---:|
| `aten::is_nonzero` | 113.628ms | 0.0% | 2.029ms | 56 | 0 B |
| `aten::item` | 113.564ms | 0.1% | 2.028ms | 56 | 0 B |
| `aten::_local_scalar_dense` | 113.419ms | 84.9% | 2.025ms | 56 | 0 B |
| `aten::matmul` | 3.587ms | 0.4% | 21.225us | 169 | 3.00 GB |
| `aten::to` | 3.440ms | 0.3% | 6.100us | 564 | 2.65 GB |
| `aten::_to_copy` | 3.100ms | 0.6% | 6.873us | 451 | 2.65 GB |
| `aten::copy_` | 2.746ms | 2.1% | 3.906us | 703 | 0 B |
| `aten::mul` | 2.115ms | 1.6% | 4.180us | 506 | 4.72 GB |
| `aten::linear` | 2.107ms | 0.1% | 18.647us | 113 | 1.89 GB |
| `aten::clone` | 1.369ms | 0.1% | 6.985us | 196 | 2.19 GB |
| `aten::add` | 1.210ms | 0.9% | 5.378us | 225 | 565.70 MB |
| `aten::mm` | 1.141ms | 0.9% | 10.101us | 113 | 1.89 GB |
| `aten::masked_fill` | 952.523us | 0.1% | 34.019us | 28 | 896.00 MB |
| `aten::repeat_interleave` | 912.449us | 0.1% | 16.294us | 56 | 448.00 MB |
| `aten::reshape` | 888.578us | 0.2% | 3.498us | 254 | 672.00 MB |

---

## Profiler Detailed Table

```text
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                         Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls  
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                  aten::zeros         0.07%      87.541us         0.31%     409.191us       7.307us           0 B           0 B     224.44 MB           0 B            56  
                  aten::empty         0.33%     438.159us         0.33%     438.159us       1.554us           0 B           0 B       2.41 GB       2.41 GB           282  
                  aten::zero_         0.03%      33.395us         0.16%     215.873us       3.855us           0 B           0 B           0 B           0 B            56  
                  aten::fill_         0.23%     305.358us         0.23%     305.358us       3.635us           0 B           0 B           0 B           0 B            84  
                 aten::arange         0.02%      22.518us         0.04%      47.990us      23.995us           0 B           0 B       8.00 KB           0 B             2  
                aten::resize_         0.00%       5.481us         0.00%       5.481us       2.740us           0 B           0 B       4.00 MB       4.00 MB             2  
              aten::embedding         0.01%       9.330us         0.05%      67.958us      67.958us           0 B           0 B       4.00 MB           0 B             1  
                aten::reshape         0.20%     270.139us         0.67%     888.578us       3.498us           0 B           0 B     672.00 MB           0 B           254  
                   aten::view         0.14%     188.018us         0.14%     188.018us       0.553us           0 B           0 B           0 B           0 B           340  
           aten::index_select         0.01%      17.802us         0.04%      48.621us      48.621us           0 B           0 B       4.00 MB           0 B             1  
                 aten::expand         0.21%     281.348us         0.29%     386.407us       1.527us           0 B           0 B           0 B           0 B           253  
             aten::as_strided         0.48%     646.519us         0.48%     646.519us       0.499us           0 B           0 B           0 B           0 B          1296  
                 aten::gather         0.01%      19.644us         0.02%      20.227us      20.227us           0 B           0 B           0 B           0 B             1  
                     aten::to         0.26%     340.783us         2.58%       3.440ms       6.100us       4.01 KB           0 B       2.65 GB           0 B           564  
               aten::_to_copy         0.64%     855.768us         2.32%       3.100ms       6.873us       4.01 KB           0 B       2.65 GB           0 B           451  
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 133.539ms

```
