# eval_xhi.py / eval_pmxh.py 输出字段说明

这两份脚本会在训练集（默认 `train_data/...`）和测试集（默认 `test_data/...`）上计算一组统一的定量指标，并输出为 TSV（制表符分隔）表格。

## 基本用法

```bash
python eval_xhi.py
python eval_pmxh.py
```

默认会分别写出：
- `eval_xhi.tsv`
- `eval_pmxh.tsv`

可通过 `--out` 指定输出路径。

## 公共字段（两张表都会出现）

- `dataset`：数据集标签。脚本默认会尝试输出两段：`train(train_data)` 与 `test(test_data)`。
- `n_samples`：参与统计的样本文件个数（即参与评估的曲线/网格数量）。
- `n_points`：参与统计的点的总数（把所有样本的所有点拼接后的点数，按分段/分箱后会变化）。
- `z_bin`：按红移分段汇总的分箱标签。默认两段：`5<=z<=12`、`z>12`，以及总体 `all`。

## 误差与校准指标（两张表都会出现）

### wRMSE
加权均方根误差：

```
wRMSE = sqrt( sum_i w_i * (pred_i - true_i)^2  /  sum_i w_i )
```

其中权重 `w_i` 仅随红移变化：默认对 `5<=z<=12` 的点使用更大的权重（`--z-weight-factor`，默认 3），其余点权重为 1。

### Q99
绝对误差 `|pred-true|` 的 99% 分位数（越小越好）。相比 `max(|err|)` 更不容易被单个离群点支配，适合追踪“最坏情况的尾部误差”。

### Coverage68 / Coverage95
不确定度覆盖率（校准指标）：
- `Coverage68`：真值落在预测均值 ±1σ 区间内的比例
- `Coverage95`：真值落在预测均值 ±2σ 区间内的比例

脚本会对 `std` 做一个下限裁剪（`--min-std`），避免 `std≈0` 时覆盖率与 NLL 数值异常。

### NLL
高斯负对数似然（越小越好）：

```
NLL = mean_i [ 0.5*log(2*pi*std_i^2) + 0.5*(err_i^2 / std_i^2) ]
```

它同时惩罚“预测不准”和“std 报得不合理”（过小会导致 NLL 爆炸）。

## eval_xhi.py 额外字段（xHI 形状特征）

这些字段只在 `z_bin=all` 的行输出，用于量化曲线形状参数的偏差（单位为 redshift 或对数宽度）。

- `mae_z05`：`z(xHI=0.05)` 的 MAE
- `mae_z25`：`z(xHI=0.25)` 的 MAE
- `mae_z50`：`z(xHI=0.50)` 的 MAE
- `mae_z75`：`z(xHI=0.75)` 的 MAE
- `mae_log_dzl`：`log(z50 - z10)` 的 MAE（左侧宽度的对数）
- `mae_log_dzr`：`log(z90 - z50)` 的 MAE（右侧宽度的对数）
- `n_shape`：参与形状特征 MAE 统计的样本数

其中 `zXX` 的真值是从每个样本文件的曲线用插值求得（将 xHI 强制裁剪到 [0,1] 并单调化后，再做 `z(xHI=target)` 的插值）。

## eval_pmxh.py 额外字段（pmxh 形状特征 + k 分箱）

### k_bin
按 k 分箱汇总的标签。默认按 `log10(k)` 的分位数自动分为 `--k-nbins` 个箱（默认 4），并输出：
- 总体 `k_bin=all`
- 若干个 `lo<=k<=hi` 的分箱行

### mae_zmin / mae_pmin
U 形曲线极小值位置/深度的 MAE（只对 pmxh 输出）：
- `mae_zmin`：每条 `pmxh(z,k)` 曲线的 `z_min`（使 pmxh 最小的 z）之差的 MAE
- `mae_pmin`：对应极小值 `pmxh(z_min)` 之差的 MAE

在 `k_bin` 的分箱行里，这两个 MAE 也会按对应的 k 只对该分箱内的曲线进行统计。

