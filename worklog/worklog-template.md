# 工作日志模板

## 标题
- 日期时间：`YYYY-MM-DD HH:MM`
- 标签：`tag1, tag2`

## 背景/目的
- 简述为什么要做这次修改或运行测试。

## 改动点
- 列出主要代码或配置改动（文件、参数、范围）。

## 运行命令
- 命令：`<your command>`
- 工作目录：`<cwd>`
- 环境/依赖：`conda env / mpi / cluster`

## 结果摘要
- 关键指标：如 `ndead`, `nposterior`, `<nlike>`, `log(Z) ± err` 等。
- 是否符合预期：Yes/No，简要说明差异与原因。

## 产物与链接
- 日志输出：`worklog/logs/<timestamp>-<slug>.log`
- 附件：`worklog/attachments/...`

## 后续行动
- 下一步要做什么，要调整的参数或代码位置。