# Git 同步命令速查（Cheat Sheet）

便于快速把本地改动（含工作日志）同步到你的 GitHub 仓库。根据需要复制粘贴相应命令即可。

## 日常工作流（最常用）
- 查看状态与当前分支：
  - `git status -sb`
  - `git branch --show-current`
  - `git remote -v`
- 创建并切换到新分支（首推第一次推送时使用）：
  - `git switch -c <branch>`
  - 旧版命令：`git checkout -b <branch>`
- 选择性暂存并提交（避免把无关文件混入）：
  - `git add <文件或目录路径>`
  - 示例：`git add worklog/*.md bin/wlog.py .gitignore worklog/.gitkeep`
  - `git commit -m "<简洁的提交信息>"`
- 首次推送并建立跟踪：
  - `git push -u origin <branch>`
- 后续同一分支只需：
  - `git push`

## 远端仓库设置（Fork 与上游同步）
- 查看远端：`git remote -v`
- 将 `origin` 指向你的 GitHub Fork（推荐使用 SSH 地址）：
  - `git remote set-url origin git@github.com:<你的用户名>/vega.git`
- 添加上游官方仓库用于同步：
  - `git remote add upstream https://github.com/andreicuceu/vega.git`
- 从上游获取更新：
  - `git fetch upstream`
  - 把上游主分支变更整合到当前分支（二选一）：
    - 变更并生成合并提交：`git merge upstream/master`
    - 线性历史（更整洁）：`git rebase upstream/master`

## SSH 密钥与认证（macOS）
- 生成 GitHub 专用密钥：
  - `ssh-keygen -t ed25519 -C "github@local" -f ~/.ssh/id_ed25519_github -N ""`
- 将密钥加入代理：
  - `ssh-add ~/.ssh/id_ed25519_github`
- 把公钥添加到 GitHub：
  - 复制公钥：`cat ~/.ssh/id_ed25519_github.pub`
  - 到 GitHub → Settings → SSH and GPG keys → New SSH key 粘贴保存
- 测试认证：
  - `ssh -T git@github.com`
  - 预期输出类似：`Hi <你的用户名>! You've successfully authenticated, but GitHub does not provide shell access.`
- 可选：为 GitHub 限定密钥（防止与其它主机混用），编辑 `~/.ssh/config` 增加：
  - ```
    Host github.com
      User git
      HostName github.com
      IdentityFile ~/.ssh/id_ed25519_github
      IdentitiesOnly yes
      AddKeysToAgent yes
    ```

## 常见错误与快速修复
- `src refspec <branch> does not match any`
  - 原因：本地不存在该分支或当前是分离的 HEAD。
  - 解决：`git switch -c <branch>` 后再 `git push -u origin <branch>`；或 `git push -u origin HEAD:<branch>`。
- `Permission denied (publickey)`
  - 原因：SSH 公钥未添加到 GitHub 或未被代理使用。
  - 解决：按上面的“SSH 密钥与认证”段添加公钥，并执行 `ssh-add ~/.ssh/id_ed25519_github` 后重试。
- `ERROR: Repository not found.`
  - 原因：`origin` 指向的仓库不存在或你无权限。
  - 解决：先在 GitHub 上 fork 官方仓库到你的账号，或创建同名仓库；然后 `git remote set-url origin git@github.com:<你的用户名>/vega.git` 再推送。
- `HEAD detached at <tag/commit>`（分离的 HEAD）
  - 现象：`git branch --show-current` 无分支名，提交不在任何分支上。
  - 修复：`git switch -c <branch>`，提交并推送到该分支。
重启后加载密钥：eval "$(ssh-agent -s)"； 然后 `ssh-add --apple-use-keychain ~/.ssh/id_ed25519_github`。

## 工作日志提交示例（本仓库）
- 提交工作日志与相关脚本：
  - `git add worklog/*.md worklog/README.md worklog/worklog-template.md`
  - `git add bin/wlog.py .gitignore worklog/.gitkeep`
  - `git commit -m "Add worklog entries and logging script"`
  - `git push -u origin reproduce`

## 其他有用命令
- 查看当前分支：`git branch --show-current`
- 切换到已有分支：`git switch <branch>`
- 拉取远端更新（保持线性历史）：`git pull --rebase origin <branch>`
- 对比暂存区与工作区：`git diff`
- 对比暂存区与上一次提交：`git diff --cached`

提示：首次推送分支使用 `git push -u origin <branch>` 建立跟踪，之后可以直接用 `git push`。

## 暂存与提交（本地概念与命令）
- 工作区：你正在编辑的文件的当前状态。
- 暂存区：下一次提交要包含的改动候选集。
- 本地仓库：保存在你电脑上的 Git 历史（提交记录）。
- 提交（commit）：把暂存区的快照永久记录到本地仓库历史中。

- `git add <路径>`：把指定文件或改动加入“暂存区”。例如：
  - `git add worklog/*.md`
  - `git add bin/wlog.py .gitignore`
- `git commit -m "<说明>"`：把“暂存区”的内容写入本地仓库生成一个提交。

- 查看状态与改动：
  - 当前状态：`git status -sb`
  - 未暂存改动：`git diff`
  - 已暂存改动：`git diff --cached`

- 常见用法与修复：
  - 标准流程：编辑 → `git add` → `git commit -m "说明"` → `git push`
  - 快捷提交已跟踪文件的修改：`git commit -am "说明"`
  - 从暂存区移除文件：`git restore --staged <文件>`
  - 撤回所有暂存：`git reset`

- 本地 vs 远端：
  - `git commit` 仅在本地保存历史，不会上传到 GitHub。
  - `git push` 才会把本地分支的提交上传到远端；首次推荐 `git push -u origin <branch>`。