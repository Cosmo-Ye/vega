# Emulator（当前工作流）

本目录的当前工作流只保留 “shape emulator” 系列实现，用于从 21cmFAST 输出数据中训练并预测：
- `xHI(z)`（给定参数 + 任意 `z`）
- `pmxh(z,k)`（给定参数 + 任意 `z,k`）

并提供：
- 单条曲线真值 vs 预测值对比图
- 测试集全点散点图（truth vs prediction）与全局 R²

## 数据约定

### Sobol 参数
- `sobol.txt`：训练集参数（每行一组参数，列数 = 参数维度）。
- `sobol_inner.txt`：测试集参数（同上）。
- 文件名后缀 `_emu_i` 中的 `i` 对应 sobol 文件的第 `i` 行。

### xHI 文件格式
- 例：`train_data/xHI/xH_concat_emu_{i}` 或 `firsthalf_results/xHI/xH_concat_emu_{i}`
- 两列：`z`, `xHI`

### cross 文件格式（pmxh）
- 例：`train_data/cross/cross_power_concat_emu_{i}` 或 `firsthalf_results/cross/cross_power_concat_emu_{i}`
- 至少三列，脚本只读取前三列：`z`, `k`, `pmxh`（其余列忽略）
- 期望文件内部是规则网格：对每个 `z` 块有相同的 `k` 列表

## 依赖安装

```bash
python -m pip install numpy scipy scikit-learn joblib matplotlib
```

## 脚本一览

### 训练/预测
- `xhi_shape_emulator.py`：训练并预测 `xHI(z)`
- `pmxh_shape_emulator.py`：训练并预测 `pmxh(z,k)`（每个 k 一个形状参数 GP，在 `log10(k)` 上平滑连接）

### 对比/误差评估
- `compare_xhi_shape_emulator.py`：把同一条 `xHI(z)` 的真值与预测值画在同一张图（可选 ±1σ）
- `compare_pmxh_shape_emulator.py`：在给定 `k` 下画 `pmxh(z,k)` 曲线，并把真值与预测值画在同一张图（可选 ±1σ）
- `error_xHI.py`：对测试集所有样本、所有 z 点做散点图并计算全局 R²
- `error_pmxh.py`：对测试集所有样本、所有 (z,k) 点做散点图并计算全局 R²

### 辅助画图
- `train_plot/plot_train_xhi.py`：批量画训练集 `xHI(z)`（一个文件一张图）

## 用法：xhi_shape_emulator.py

### 训练
```bash
python xhi_shape_emulator.py train \
  --sobol sobol.txt \
  --xhi-dir train_data/xHI \
  --optimize-kernel \
  --out xhi_shape_emulator.joblib
```

### 单点预测（给定 params + z）
```bash
python xhi_shape_emulator.py predict \
  --model xhi_shape_emulator.joblib \
  --z 12.345 \
  --params 35.51756401546299458 93822.94044829905033
```

### 输出整条曲线（与某个文件同 z 网格）
```bash
python xhi_shape_emulator.py predict-curve \
  --model xhi_shape_emulator.joblib \
  --params 35.51756401546299458 93822.94044829905033 \
  --z-file test_data/xHI/xH_concat_emu_0 \
  --out shape_pred_emu_0.txt
```

## 用法：pmxh_shape_emulator.py

### 训练
```bash
python pmxh_shape_emulator.py train \
  --sobol sobol.txt \
  --cross-dir train_data/cross \
  --optimize-kernel \
  --out pmxh_shape_emulator.joblib
```

默认训练过程会在训练集内部随机留出一部分样本作为 holdout，用于对 `std` 做分段校准（不会使用测试集）。如需关闭该校准：

```bash
python pmxh_shape_emulator.py train \
  --sobol sobol.txt \
  --cross-dir train_data/cross \
  --optimize-kernel \
  --no-calibrate \
  --out pmxh_shape_emulator.joblib
```

也可调整 holdout 比例/最小 holdout 数量：

```bash
python pmxh_shape_emulator.py train \
  --sobol sobol.txt \
  --cross-dir train_data/cross \
  --optimize-kernel \
  --calib-holdout-frac 0.2 \
  --calib-min-holdout 8 \
  --out pmxh_shape_emulator.joblib
```

### 单点预测（给定 params + z + k）
```bash
python pmxh_shape_emulator.py predict \
  --model pmxh_shape_emulator.joblib \
  --z 7.08 \
  --k 0.07542991 \
  --params 35.51756401546299458 93822.94044829905033
```

## 用法：compare_xhi_shape_emulator.py（单图对比真值/预测）

### 单文件
```bash
python compare_xhi_shape_emulator.py \
  --model xhi_shape_emulator.joblib \
  --sobol sobol_inner.txt \
  --test-file test_data/xHI/xH_concat_emu_0 \
  --out-dir compare_plot_shape
```

### 批量（目录 + glob）
```bash
python compare_xhi_shape_emulator.py \
  --model xhi_shape_emulator.joblib \
  --sobol sobol_inner.txt \
  --test-dir test_data/xHI \
  --test-glob 'xH_concat_emu_*' \
  --out-dir test_data/compare_xHI
```

## 用法：compare_pmxh_shape_emulator.py（给定 k 的 pmxh-z 曲线对比）

### 单文件
```bash
python compare_pmxh_shape_emulator.py \
  --model pmxh_shape_emulator.joblib \
  --sobol sobol_inner.txt \
  --test-file test_data/cross/cross_power_concat_emu_0 \
  --k 0.1 \
  --out-dir test_data/compare_cross
```

### 批量（目录 + glob）
```bash
python compare_pmxh_shape_emulator.py \
  --model pmxh_shape_emulator.joblib \
  --sobol sobol_inner.txt \
  --test-dir test_data/cross \
  --test-glob 'cross_power_concat_emu_*' \
  --k 0.1 \
  --out-dir test_data/compare_cross
```

## 用法：error_xHI.py（散点图 + R²）
```bash
python error_xHI.py \
  --model xhi_shape_emulator.joblib \
  --test-dir test_data/xHI \
  --test-glob 'xH_concat_emu_*' \
  --test-sobol sobol_inner.txt \
  --out-dir error_plot
```

## 用法：error_pmxh.py（散点图 + R²）
```bash
python error_pmxh.py \
  --model pmxh_shape_emulator.joblib \
  --test-dir test_data/cross \
  --test-glob 'cross_power_concat_emu_*' \
  --test-sobol sobol_inner.txt \
  --out-dir error_plot
```

## 用法：train_plot/plot_train_xhi.py（批量画训练集）
```bash
python train_plot/plot_train_xhi.py \
  --src train_data/xHI \
  --out train_plot \
  --start 0 \
  --end 68
```
对模型进行评估
```bash
python eval_xhi.py --model xhi_shape_emulator.joblib --out eval_xhi.tsv
python eval_pmxh.py --model pmxh_shape_emulator.joblib --out eval_pmxh.tsv
```

用于下载、处理MCMC链的指令：
scp 'yifan_venus:/work/zhuxing/DESIdr1/emulator/mcmc/extlong*' /Users/zhengyifan/Desktop/cosmology/DESI_Y1/emulator/mcmc

python mcmc/check_chain.py --in mcmc/extlong1.npz mcmc/extlong2.npz mcmc/extlong3.npz mcmc/extlong4.npz

python mcmc/xhi_band_from_chain.py \
  --chains mcmc/extlong1.npz mcmc/extlong2.npz mcmc/extlong3.npz mcmc/extlong4.npz \
  --nprocs 8 --chunk-size 256 \
  --out-txt mcmc/xhi_extwhole.txt

scp 'yifan_venus:/work/zhuxing/DESIdr1/emulator/mcmc/xhi_extwhole.txt' /Users/zhengyifan/Desktop/cosmology/DESI_Y1/emulator/mcmc

python mcmc/plot_result.py \
  --kind xhi-band \
  --in mcmc/xhi_extwhole.txt \
  --out mcmc/xhi_extwhole.png