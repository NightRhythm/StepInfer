# Profiler Breakdown — Qwen3-1.7B-Base

## Setup

| Parameter | Value |
|:---|:---|
| Model | Qwen3-1.7B-Base |
| Parameters | 1,720,574,976 |
| Precision | bf16 |
| Device | NVIDIA GeForce RTX 5060 Ti |
| PyTorch | 2.9.1+cu128 |
| Python | 3.12.16 |
| Prompt Tokens | 512 |
| Generated Tokens | 1 |
| Chrome Trace | [`../traces/baseline_trace_1.7b_prompt512.json`](../traces/baseline_trace_1.7b_prompt512.json) |

---

## Top 15 Operations (Sorted by CPU Total / Dispatch)

| Op | CPU Total | Self CPU % | CPU Time Avg | # Calls | CUDA Mem |
|:---|---:|---:|---:|---:|---:|
| `aten::is_nonzero` | 48.666ms | 0.1% | 869.030us | 56 | 0 B |
| `aten::item` | 48.608ms | 0.1% | 867.992us | 56 | 0 B |
| `aten::_local_scalar_dense` | 48.525ms | 71.6% | 866.512us | 56 | 0 B |
| `aten::matmul` | 3.507ms | 0.8% | 20.749us | 169 | 988.38 MB |
| `aten::to` | 3.434ms | 0.7% | 6.089us | 564 | 860.00 MB |
| `aten::_to_copy` | 2.938ms | 1.1% | 6.516us | 451 | 860.00 MB |
| `aten::mul` | 2.666ms | 3.9% | 5.270us | 506 | 1.48 GB |
| `aten::linear` | 2.468ms | 0.2% | 21.840us | 113 | 708.38 MB |
| `aten::copy_` | 2.206ms | 3.3% | 3.918us | 563 | 0 B |
| `aten::mm` | 1.479ms | 2.2% | 13.085us | 113 | 708.38 MB |
| `aten::clone` | 1.027ms | 0.2% | 9.169us | 112 | 392.00 MB |
| `aten::repeat_interleave` | 1.006ms | 0.2% | 17.968us | 56 | 112.00 MB |
| `aten::bmm` | 951.824us | 1.4% | 16.997us | 56 | 280.00 MB |
| `aten::add` | 909.006us | 1.3% | 4.040us | 225 | 197.42 MB |
| `aten::masked_fill` | 794.038us | 0.2% | 28.358us | 28 | 224.00 MB |

---

## Profiler Detailed Table

```text
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                         Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls  
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                 aten::arange         0.13%      87.823us         0.26%     178.582us      89.291us           0 B           0 B       8.00 KB           0 B             2  
                  aten::empty         0.32%     215.701us         0.32%     215.701us       1.519us           0 B           0 B     399.00 MB     399.00 MB           142  
                aten::resize_         0.03%      19.648us         0.03%      19.648us       9.824us           0 B           0 B       2.00 MB       2.00 MB             2  
              aten::embedding         0.01%       8.994us         0.12%      82.344us      82.344us           0 B           0 B       2.00 MB           0 B             1  
                aten::reshape         0.35%     235.387us         0.55%     373.378us       1.470us           0 B           0 B           0 B           0 B           254  
                   aten::view         0.29%     197.785us         0.29%     197.785us       0.582us           0 B           0 B           0 B           0 B           340  
           aten::index_select         0.03%      18.955us         0.09%      62.201us      62.201us           0 B           0 B       2.00 MB           0 B             1  
                 aten::expand         0.43%     290.030us         0.58%     392.975us       1.553us           0 B           0 B           0 B           0 B           253  
             aten::as_strided         0.85%     579.247us         0.85%     579.247us       0.540us           0 B           0 B           0 B           0 B          1072  
                 aten::gather         0.04%      26.985us         0.04%      27.816us      27.816us           0 B           0 B           0 B           0 B             1  
                     aten::to         0.73%     495.494us         5.07%       3.434ms       6.089us       4.01 KB           0 B     860.00 MB           0 B           564  
               aten::_to_copy         1.11%     754.705us         4.33%       2.938ms       6.516us       4.01 KB           0 B     860.00 MB     128.00 KB           451  
          aten::empty_strided         0.90%     607.328us         0.90%     607.328us       1.347us       4.01 KB       4.01 KB     859.88 MB     859.88 MB           451  
                  aten::copy_         3.25%       2.206ms         3.25%       2.206ms       3.918us           0 B           0 B           0 B           0 B           563  
                    aten::pow         0.88%     599.864us         0.96%     651.913us       5.769us           0 B           0 B     396.00 MB     396.00 MB           113  
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 67.796ms

```
