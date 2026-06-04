# MCMC 教程：用 `P_mpsi` 观测约束 `zeta_UV` 与 `Tvir`

本目录下的脚本把两件事串起来：

1. 从 PolyChord 链（Yukawa 参数 `yuk_A/yuk_alpha/yuk_beta`）推导 `P_mpsi(k)` 的“观测摘要”（中位数 + 1σ）。
2. 使用 `pmxh_shape_emulator.py` 训练出的 `P_mxHI(z,k)` emulator，做一个最小可用的 MCMC 框架来拟合 `zeta_UV` 与 `Tvir` 的后验分布。

`P_mxHI -> P_mpsi` 的映射已在脚本里实现为一套默认版本；如果你的物理模型不同，可以在对应函数中替换/修改映射公式。

相关脚本：

- `P_mpsi` 观测摘要脚本：[script_mcmc_obs.py](file:///Users/zhengyifan/Desktop/cosmology/DESI_Y1/emulator/mcmc/script_mcmc_obs.py)
- MCMC 主脚本：[run_mcmc.py](file:///Users/zhengyifan/Desktop/cosmology/DESI_Y1/emulator/mcmc/run_mcmc.py)
- Emulator 模型定义/训练脚本（在仓库根目录）：[pmxh_shape_emulator.py](file:///Users/zhengyifan/Desktop/cosmology/DESI_Y1/emulator/pmxh_shape_emulator.py)

---

## 运行环境与目录假设

- 推荐从仓库根目录运行（即 `.../emulator/`），因为默认路径参数都是以此为基准的。
- 默认使用的 PolyChord 链文件在：
  - `mcmc/Polychord_result/full_reio_extendprior/Polychord_fullcombined_extendprior.*`
  - `mcmc/Polychord_result/auto_reio_extendprior/Polychord_auto_extendprior.*`
- 默认 emulator 模型文件为：`pmxh_shape_emulator.joblib`（仓库根目录下已存在）。

---

## 1. 从链得到 `P_mpsi(k)` 的观测摘要

### 1.1 作为脚本运行（打印 `low/median/high`）

从仓库根目录运行：

```bash
python mcmc/script_mcmc_obs.py
```

它会对默认的 `k_bin` 计算：

- `low`：16% 分位
- `median`：50% 分位
- `high`：84% 分位

这些输出就是你可用于拟合的“观测数据摘要”。

### 1.2 作为函数调用（推荐在其他脚本中复用）

[script_mcmc_obs.py](file:///Users/zhengyifan/Desktop/cosmology/DESI_Y1/emulator/mcmc/script_mcmc_obs.py) 提供了：

`load_pmpsi_from_chain(chain_root, k_bin, ignore_rows=0.2) -> dict`

返回字段：

- `k`: `np.ndarray`，k bin（长度 `Nk`）
- `median`: `np.ndarray`，`P_mpsi(k)` 中位数（长度 `Nk`）
- `low`: `np.ndarray`，16% 分位（长度 `Nk`）
- `high`: `np.ndarray`，84% 分位（长度 `Nk`）
- `sigma`: `np.ndarray`，定义为 `(high-low)/2`（长度 `Nk`）

参数含义：

- `chain_root`：GetDist 的 root 前缀（不带后缀），例如  
  `mcmc/Polychord_result/full_reio_extendprior/Polychord_fullcombined_extendprior`
- `k_bin`：你想评估的 k 值列表（会转成数组）
- `ignore_rows`：读取链时丢弃前面比例样本（burn-in），默认 0.2

---

## 2. MCMC：用 emulator 拟合 `zeta_UV` 与 `Tvir`

主脚本：[run_mcmc.py](file:///Users/zhengyifan/Desktop/cosmology/DESI_Y1/emulator/mcmc/run_mcmc.py)

### 2.1 最小运行示例

```bash
python mcmc/run_mcmc.py --nsteps 20000 --burn 2000 --thin 10 --out mcmc/mh_zeta_tvir_chain.npz
```

输出：

- `mcmc/mh_zeta_tvir_chain.npz`：保存 thinned 后的链与元信息
- 标准输出会打印 `saved/acc_rate/n_saved`

### 2.2 输出文件格式（npz）

保存的字段包括（见脚本末尾 `np.savez_compressed`）：

- `chain`: `(Nsaved, 2)`，两列依次为 `[zeta_UV, Tvir]`
- `logp`: `(Nsaved,)`，对应的 log-posterior
- `acc_rate`: 接受率（0~1）
- `k_obs`, `pmpsi_obs`, `pmpsi_sigma`: 拟合用的数据与误差
- `pmpsi_chain_root`, `ignore_rows`, `emu_model`: 输入配置记录
- `zeta_idx`, `tvir_idx`, `tvir_in_log10`, `z_select`: 与 emulator/映射相关的配置记录（其中 `z_select` 是最终输出 `P_mpsi` 的目标红移，默认 2.33）

读取示例：

```python
import numpy as np

data = np.load("mcmc/mh_zeta_tvir_chain.npz")
chain = data["chain"]
zeta = chain[:, 0]
tvir = chain[:, 1]
print(np.mean(zeta), np.mean(tvir))
```

---

## 3. 你需要补的关键：`P_mxHI -> P_mpsi` 映射

映射函数：

- [pmpsi_from_pmxh](file:///Users/zhengyifan/Desktop/cosmology/DESI_Y1/emulator/mcmc/run_mcmc.py#L74-L116)

它当前做的事：

- 使用 5 个红移节点（2.0/2.5/3.0/3.5/4.0）定义的 `Psi` 数据表，构造分段线性样条并计算一阶导数；
- 在 emulator 输出的 `z_grid` 上做数值积分，得到 5 个节点对应的 `P_mpsi(z_i, k)`；
- 在红移方向把 `P_mpsi(z_i, k)` 插值到目标红移 `z_select`（默认 2.33），得到 `P_mpsi(z_select, k)`；
- 最后在 log-k 上插值到观测 `k_obs`。

你要做的事：

- 如果你的 `Psi` 数据表或积分核形式不同：在 `pmpsi_from_pmxh` 中替换对应的插值/积分核/外推设定。
- 如果你希望修改误差传播：在 `pmpsi_from_pmxh` 中调整 `pmpsi_by_node_var` 的构造方式（例如引入 redshift 相关性）。

MCMC 的似然是对角高斯（可自行改成协方差形式）：

- 见函数 [_log_likelihood_gaussian_diag](file:///Users/zhengyifan/Desktop/cosmology/DESI_Y1/emulator/mcmc/run_mcmc.py#L166-L177)

总误差的组合方式：

- 默认会把 emulator 误差并入总误差；如需关闭，使用 `--no-emu-std`。  
  `sigma_tot = sqrt(pmpsi_sigma^2 + pmpsi_model_std^2)`

---

## 4. 参数说明：MCMC 与 emulator 如何对接

### 4.1 emulator 的输入参数向量

`PMXHShapeEmulator.predict_grid(params=...)` 需要一个“完整的参数向量”（长度等于 `emu.train_params.shape[1]`）。

在 MCMC 里：

- `zeta_UV` 与 `Tvir` 是要采样的两个参数；
- 其余参数固定为 `base_params`。

构造方式见：

- [_build_full_params](file:///Users/zhengyifan/Desktop/cosmology/DESI_Y1/emulator/mcmc/run_mcmc.py#L139-L155)

相关参数：

- `--base-params`: 直接提供完整参数向量（长度必须等于 emulator 的参数维数）
  - 不提供时：脚本用 `train_params` 的逐维中位数作为默认基准
- `--zeta-idx`: `zeta_UV` 在参数向量中的下标（从 0 开始）
- `--tvir-idx`: `Tvir` 在参数向量中的下标（从 0 开始）
- `--tvir-in-log10`: 若 emulator 的该维参数使用 `log10(Tvir)` 编码，则打开此开关

### 4.2 先验与 MCMC 步长

脚本当前实现的是最简单的 uniform prior + 随机游走 MH：

先验范围：

- `--zeta-min/--zeta-max`
- `--tvir-min/--tvir-max`

链设置：

- `--nsteps`: MH 总步数（包含 burn-in）
- `--burn`: 丢弃前 burn 步
- `--thin`: 抽稀因子
- `--seed`: 随机种子

初始化与步长（对接受率影响很大）：

- `--zeta-init`, `--tvir-init`
- `--zeta-step`, `--tvir-step`

一般经验：

- 接受率 `acc_rate` 太高：步长通常偏小（探索太慢）
- 接受率 `acc_rate` 太低：步长通常偏大（拒绝太多）

---

## 5. 常见问题排查

### 5.1 `FileNotFoundError` / 找不到链文件

确认你从仓库根目录运行，并检查 `--pmpsi-chain-root` 是否指向正确 root 前缀（不要写成目录名）。

正确示例（默认值）：

- `mcmc/Polychord_result/full_reio_extendprior/Polychord_fullcombined_extendprior`

对应文件应存在：

- `.../Polychord_fullcombined_extendprior.txt`
- `.../Polychord_fullcombined_extendprior.paramnames`

### 5.2 `--base-params must have length ...`

说明你传入的 `--base-params` 维数与 emulator 的参数维数不一致。需要传入完整长度向量，或不传让脚本用训练集的中位数当基准。

### 5.3 占位映射导致结果“看起来不物理”

这是可能出现的情况。需要按你的模型将 `P_mxHI` 转为 `P_mpsi`（修改 `pmpsi_from_pmxh`），再重新跑 MCMC。
