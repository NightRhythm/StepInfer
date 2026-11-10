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
| Prompt Tokens | 512 |
| Generated Tokens | 1 |
| Chrome Trace | None (disabled) |

---

## Top 15 Operations (Sorted by CPU Total / Dispatch)

| Op | CPU Total | Self CPU % | CPU Time Avg | # Calls | CUDA Mem |
|:---|---:|---:|---:|---:|---:|
| `aten::is_nonzero` | 17.751ms | 0.1% | 316.981us | 56 | 0 B |
| `aten::item` | 17.709ms | 0.1% | 316.231us | 56 | 0 B |
| `aten::_local_scalar_dense` | 17.660ms | 50.6% | 315.360us | 56 | 0 B |
| `aten::matmul` | 2.975ms | 1.4% | 17.603us | 169 | 764.38 MB |
| `aten::to` | 2.937ms | 0.9% | 5.207us | 564 | 771.84 MB |
| `aten::_to_copy` | 2.616ms | 2.0% | 5.801us | 451 | 771.84 MB |
| `aten::copy_` | 2.184ms | 6.3% | 3.528us | 619 | 0 B |
| `aten::mul` | 2.113ms | 6.1% | 4.176us | 506 | 1.18 GB |
| `aten::linear` | 1.870ms | 0.4% | 16.552us | 113 | 484.38 MB |
| `aten::mm` | 1.040ms | 3.0% | 9.201us | 113 | 484.38 MB |
| `aten::bmm` | 958.434us | 2.7% | 17.115us | 56 | 280.00 MB |
| `aten::add` | 937.530us | 2.7% | 4.167us | 225 | 141.42 MB |
| `aten::clone` | 838.999us | 0.3% | 7.491us | 112 | 419.29 MB |
| `aten::repeat_interleave` | 778.020us | 0.5% | 13.893us | 56 | 139.29 MB |
| `aten::masked_fill` | 696.595us | 0.4% | 24.878us | 28 | 224.00 MB |

---

## Profiler Detailed Table

```text
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                         Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls  
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                  aten::zeros         0.22%      75.772us         1.04%     361.202us       6.450us           0 B           0 B      70.15 MB           0 B            56  
                  aten::empty         0.77%     267.702us         0.77%     267.702us       1.352us           0 B           0 B     496.44 MB     496.44 MB           198  
                  aten::zero_         0.09%      31.237us         0.57%     197.465us       3.526us           0 B           0 B           0 B           0 B            56  
                  aten::fill_         0.77%     267.095us         0.77%     267.095us       3.180us           0 B           0 B           0 B           0 B            84  
                 aten::arange         0.05%      18.532us         0.11%      38.863us      19.432us           0 B           0 B       8.00 KB           0 B             2  
                aten::resize_         0.01%       4.809us         0.01%       4.809us       2.405us           0 B           0 B       1.00 MB       1.00 MB             2  
              aten::embedding         0.02%       8.701us         0.16%      55.552us      55.552us           0 B           0 B       1.00 MB           0 B             1  
                aten::reshape         0.57%     200.323us         0.91%     317.518us       1.250us           0 B           0 B           0 B           0 B           254  
                   aten::view         0.43%     148.868us         0.43%     148.868us       0.438us           0 B           0 B           0 B           0 B           340  
           aten::index_select         0.04%      14.235us         0.11%      39.004us      39.004us           0 B           0 B       1.00 MB           0 B             1  
                 aten::expand         0.64%     221.670us         0.92%     320.306us       1.266us           0 B           0 B           0 B           0 B           253  
             aten::as_strided         1.63%     569.447us         1.63%     569.447us       0.439us           0 B           0 B           0 B           0 B          1296  
                 aten::gather         0.04%      15.531us         0.05%      16.221us      16.221us           0 B           0 B           0 B           0 B             1  
                     aten::to         0.92%     320.847us         8.42%       2.937ms       5.207us       4.01 KB           0 B     771.84 MB           0 B           564  
               aten::_to_copy         2.04%     710.887us         7.50%       2.616ms       5.801us       4.01 KB           0 B     771.84 MB           0 B           451  
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 34.896ms

```
