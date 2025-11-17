# 工作日志（Worklog）

这个目录用于记录日常改动、测试命令、结果摘要和产物链接，帮助你在复杂迭代中快速回忆“当时做了什么、怎么做的、得到什么结果”。

## 快速开始

- 记录一条说明（不执行命令）：
  - `python3 bin/wlog.py --title "调整 num_repeats 测试" --desc "降低 num_repeats=2，比较 <nlike> 变化" --tags posterior,num_repeats`
- 记录一条说明并运行命令，保存输出：
  - `python3 bin/wlog.py --title "运行 main.ini" --cmd "python3 bin/run_vega.py -c main.ini" --tags run,vega`
 - 为条目附加当时的配置快照（例如 main.ini）：
  - `python3 bin/wlog.py --title "提交 MPI 作业" --cmd "sbatch run_vega_mpi.pbs" --attach main.ini --attach run_vega_mpi.pbs --tags run,MPI`
 - 把说明写入指定日期（例如 2025-11-14 16:00）：
   - `python3 bin/wlog.py --title "第一次 PolyChord 测试" --desc "采用默认 Polychord 设置与既定参数，在 32 cores/512GB 资源跑" --tags polychord,test --date 2025-11-14 --time 16:00`

脚本会：
- 将条目写入以日期命名的日志文件，例如：`worklog/2025-11-15.md`
- 将命令的stdout+stderr保存到：`worklog/logs/<timestamp>-<slug>.log`
- 在日志中写入命令、退出码、输出文件路径等元信息
- 如提供 `--attach`，会把指定文件复制到：`worklog/attachments/<timestamp>-<slug>/` 并在日志中列出相对路径
 - 如提供 `--date`/`--time`，会将条目写入对应的 `worklog/YYYY-MM-DD.md` 并显示指定的时间标签；命令输出与附件仍按当前时间戳归档。

## 建议写法

- 尽量包含以下要点：
  - 背景/目的（为什么改、期望什么）
  - 改动点（配置或代码的关键修改）
  - 运行命令（带参数、环境）
  - 结果摘要（关键指标、是否符合预期）
  - 后续行动（下一步要做什么）
- 用 `--tags` 标注关键词（如 `posterior`, `precision`, `num_live`），方便索引。

## 文件结构

- `worklog/YYYY-MM-DD.md`：每天的工作条目（Markdown）
- `worklog/logs/*.log`：命令执行的原始输出（可能很大，默认被 .gitignore 忽略）
- `worklog/attachments/`：可选附件（图片、表格等，默认忽略）

## 注意事项

- 日志文件（Markdown）是建议提交到版本库的；日志输出和附件通常不提交（已在 .gitignore 配置）。
- 如需仅记录不运行命令，省略 `--cmd` 或加 `--dry-run`。
- 需要自定义保存位置或格式可修改 `bin/wlog.py`。
 - `--attach` 的路径相对 `--cwd`（如未指定则相对仓库根目录）；复制的是当前文件内容的快照，适合记录易变的配置（如 `main.ini`）。