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
| Prompt Tokens | 512 |
| Generated Tokens | 1 |
| Chrome Trace | [`../traces/baseline_trace_0.6b_prompt512.json`](../traces/baseline_trace_0.6b_prompt512.json) |

---

## Top 15 Operations (Sorted by CPU Total / Dispatch)

| Op | CPU Total | Self CPU % | CPU Time Avg | # Calls | CUDA Mem |
|:---|---:|---:|---:|---:|---:|
| `aten::is_nonzero` | 20.579ms | 0.1% | 367.485us | 56 | 0 B |
| `aten::item` | 20.531ms | 0.2% | 366.618us | 56 | 0 B |
| `aten::_local_scalar_dense` | 20.471ms | 54.1% | 365.550us | 56 | 0 B |
| `aten::to` | 3.124ms | 1.1% | 5.540us | 564 | 689.00 MB |
| `aten::matmul` | 3.063ms | 1.2% | 18.125us | 169 | 764.38 MB |
| `aten::_to_copy` | 2.721ms | 1.8% | 6.034us | 451 | 689.00 MB |
| `aten::mul` | 2.447ms | 6.5% | 4.836us | 506 | 1.21 GB |
| `aten::linear` | 2.277ms | 0.3% | 20.152us | 113 | 484.38 MB |
| `aten::copy_` | 2.069ms | 5.5% | 3.675us | 563 | 0 B |
| `aten::mm` | 1.342ms | 3.5% | 11.873us | 113 | 484.38 MB |
| `aten::clone` | 923.525us | 0.3% | 8.246us | 112 | 392.00 MB |
| `aten::repeat_interleave` | 900.771us | 0.3% | 16.085us | 56 | 112.00 MB |
| `aten::add` | 887.886us | 2.3% | 3.946us | 225 | 141.42 MB |
| `aten::bmm` | 771.632us | 2.0% | 13.779us | 56 | 280.00 MB |
| `aten::masked_fill` | 731.375us | 0.4% | 26.121us | 28 | 224.00 MB |

---

## Profiler Detailed Table

```text
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                         Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls  
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                 aten::arange         0.18%      69.261us         0.37%     141.325us      70.662us           0 B           0 B       8.00 KB           0 B             2  
                  aten::empty         0.47%     177.503us         0.47%     177.503us       1.250us           0 B           0 B     399.00 MB     399.00 MB           142  
                aten::resize_         0.05%      17.512us         0.05%      17.512us       8.756us           0 B           0 B       1.00 MB       1.00 MB             2  
              aten::embedding         0.02%       7.716us         0.18%      67.101us      67.101us           0 B           0 B       1.00 MB           0 B             1  
                aten::reshape         0.55%     207.574us         0.90%     339.571us       1.337us           0 B           0 B           0 B           0 B           254  
                   aten::view         0.47%     176.604us         0.47%     176.604us       0.519us           0 B           0 B           0 B           0 B           340  
           aten::index_select         0.04%      16.035us         0.14%      51.098us      51.098us           0 B           0 B       1.00 MB           0 B             1  
                 aten::expand         0.64%     242.613us         0.88%     333.584us       1.319us           0 B           0 B           0 B           0 B           253  
             aten::as_strided         1.28%     483.107us         1.28%     483.107us       0.451us           0 B           0 B           0 B           0 B          1072  
                 aten::gather         0.06%      21.446us         0.06%      22.289us      22.289us           0 B           0 B           0 B           0 B             1  
                     aten::to         1.06%     403.007us         8.26%       3.124ms       5.540us       4.01 KB           0 B     689.00 MB           0 B           564  
               aten::_to_copy         1.78%     671.763us         7.19%       2.721ms       6.034us       4.01 KB           0 B     689.00 MB           0 B           451  
          aten::empty_strided         1.46%     552.361us         1.46%     552.361us       1.225us       4.01 KB       4.01 KB     689.00 MB     689.00 MB           451  
                  aten::copy_         5.47%       2.069ms         5.47%       2.069ms       3.675us           0 B           0 B           0 B           0 B           563  
                    aten::pow         1.61%     608.706us         1.73%     656.066us       5.806us           0 B           0 B     282.00 MB     282.00 MB           113  
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 37.842ms

```
