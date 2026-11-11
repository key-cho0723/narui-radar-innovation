# 纳睿雷达创新设计大赛

本仓库记录赛题 2（算法）的方案与研究资料。目标是设计 Ku 波段相控阵雷达的近远距兼容波形和补盲方法，并通过仿真验证低空无人机探测性能。

- [赛题原文](赛题2-算法-原文.md)
- [技术方案与试验矩阵](docs/赛题2-技术方案.md)
- [文献与创新线索](docs/赛题2-文献与创新线索.md)
- [收发覆盖示例](docs/赛题2-收发覆盖示例.png)
- [最小 Demo 实现方案、结果与输出说明](docs/赛题2-Demo技术方案.md)
- [公开文献的基线、指标和实验条件](docs/赛题2-公开基线与指标.md)
- [赛题仿真方案与同虚警率评测](docs/赛题2-仿真评测报告.md)
- [悬停、低速和 250 m 瓶颈专项文献](docs/赛题2-瓶颈专项文献.md)
- [距离–多普勒辅助白化与局部点迹验证](docs/赛题2-瓶颈改进报告.md)

最小 Demo 对比固定长脉冲、固定短脉冲与同 PRI 双脉冲的接收窗、有效能量和脉压响应。从仓库根目录使用 Python 3.13 运行以下命令，结果写入 `demo/output/`。

```powershell
python -m pip install -r demo/requirements.txt
python demo/radar_demo.py
```

完整仿真包括低空背景、双脉冲时序对照和多目标距离–速度图。以下命令将结果写入 `demo/benchmark_output/`。

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
python demo/radar_benchmark.py
python demo/radar_map.py
python -m unittest discover -s demo -v
```

瓶颈专项评测写入 `demo/benchmark_output/refinement/`。

```powershell
python demo/radar_refinement.py
python demo/radar_local_scan.py
python demo/radar_waveform_tradeoff.py
```
