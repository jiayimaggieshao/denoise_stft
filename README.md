# CardioSpecNet STFT Pipeline

频域双麦克风心音去噪流水线，基于 **CardioSpecNet**（复数 STFT U-Net + 时频 GRU）与 **复数传递抵消** 的混合融合。

## 目录结构

```
stft/
├── train_frequency.py          # 训练入口
├── src/                        # 核心模块
├── scripts/
│   ├── check_frequency_pipeline.py
│   └── run_frequency_inference.py
├── tests/
├── data/                       # 放置 npz 数据（见 data/README.md）
├── checkpoints/frequency/      # 训练输出 checkpoint
└── outputs/                    # 推理输出 WAV
```

## 环境

```powershell
cd e:\yarn\stft
pip install torch numpy scipy soundfile matplotlib tqdm
# 可选：pip install wandb
```

## 准备数据

把 windowed `.npz` 放到对应目录，格式见 [`data/README.md`](data/README.md)：

```
data/clean/step_0.1s/train/*.npz
data/clean/step_0.1s/val/*.npz
data/noise/step_0.1s/train/*.npz
data/noise/step_0.1s/val/*.npz
```

原始 PCM txt 可用 `data/scripts/` 下的工具预处理（从 `etextile-denoise` 复制）。

## 验证流水线

```powershell
python scripts/check_frequency_pipeline.py
```

## 训练

若还没有真实 npz，可先生成 demo 数据跑通 smoke：

```powershell
python scripts/bootstrap_demo_data.py
python train_frequency.py --smoke --device cpu
```

正式训练前，把真实 npz 放进 `data/clean/` 和 `data/noise/`（见 [`data/README.md`](data/README.md)），然后运行：

完整训练示例：

```powershell
python train_frequency.py `
  --step 0.1s `
  --source_stride 10 `
  --epochs 60 `
  --samples_per_epoch 20000 `
  --val_samples 2000 `
  --batch_size 16 `
  --base_channels 12 `
  --grid_blocks 2
```

Checkpoint 保存在 `checkpoints/frequency/best.pt` 和 `last.pt`。

## 推理

```powershell
python scripts/run_frequency_inference.py `
  --input data/test_real/walking/step_0.1s/heart_w6_windows.npz `
  --checkpoint checkpoints/frequency/best.pt `
  --start 0 --end 10 `
  --export_baselines
```

## 文档

- [`FREQUENCY_PIPELINE.md`](FREQUENCY_PIPELINE.md) — 架构与命令
- [`IMPLEMENTATION_SUMMARY.md`](IMPLEMENTATION_SUMMARY.md) — 交付物说明
- [`data/README.md`](data/README.md) — 数据格式与目录
