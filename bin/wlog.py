#!/usr/bin/env python3
import argparse
import datetime as dt
import os
import re
import shlex
import subprocess
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKLOG_DIR = ROOT / "worklog"
LOGS_DIR = WORKLOG_DIR / "logs"
ATTACH_DIR = WORKLOG_DIR / "attachments"


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"[^a-z0-9\-]", "", text)
    return text[:80] or "entry"


def ensure_dirs():
    for d in [WORKLOG_DIR, LOGS_DIR, ATTACH_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def append_entry(day_file: Path, content: str):
    with day_file.open("a", encoding="utf-8") as f:
        f.write(content)


def run_and_capture(cmd: str, cwd: Path | None, log_path: Path):
    with log_path.open("w", encoding="utf-8", buffering=1) as out:
        out.write(f"[cmd] {cmd}\n")
        if cwd:
            out.write(f"[cwd] {cwd}\n")
        out.write(f"[start] {dt.datetime.now().isoformat()}\n\n")

        proc = subprocess.run(
            cmd if isinstance(cmd, list) else shlex.split(cmd),
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        out.write(proc.stdout or "")
        out.write("\n\n")
        out.write(f"[end] {dt.datetime.now().isoformat()}\n")
        out.write(f"[exitcode] {proc.returncode}\n")
    return proc.returncode


def main():
    parser = argparse.ArgumentParser(description="Append a worklog entry and optionally run a command.")
    parser.add_argument("--title", required=True, help="Entry title")
    parser.add_argument("--desc", default="", help="Description / context")
    parser.add_argument("--tags", default="", help="Comma-separated tags")
    parser.add_argument("--cmd", default="", help="Command to run and capture output")
    parser.add_argument("--cwd", default="", help="Working directory for the command")
    parser.add_argument("--dry-run", action="store_true", help="Do not run the command, only log the entry")
    parser.add_argument("--attach", action="append", default=[], help="File(s) to snapshot into worklog/attachments")
    parser.add_argument("--date", default="", help="Write entry to specific date file (YYYY-MM-DD)")
    parser.add_argument("--time", default="", help="Record custom time label (HH:MM or HH:MM:SS)")

    args = parser.parse_args()

    ensure_dirs()

    now = dt.datetime.now()
    date_str = args.date.strip() if args.date else now.strftime("%Y-%m-%d")
    time_str = args.time.strip() if args.time else now.strftime("%H:%M:%S")
    day_file = WORKLOG_DIR / f"{date_str}.md"
    slug = slugify(args.title)

    log_rel_path = None
    exit_code = None
    if args.cmd:
        log_file_name = f"{now.strftime('%Y%m%d-%H%M%S')}-{slug}.log"
        log_path = LOGS_DIR / log_file_name
        if not args.dry_run:
            exit_code = run_and_capture(args.cmd, Path(args.cwd) if args.cwd else None, log_path)
        log_rel_path = os.path.relpath(log_path, ROOT)

    # Handle attachments snapshot (copy into a timestamped directory)
    attach_dir = ATTACH_DIR / f"{now.strftime('%Y%m%d-%H%M%S')}-{slug}"
    attachments_rel = []
    if args.attach:
        attach_dir.mkdir(parents=True, exist_ok=True)
        base_dir = Path(args.cwd) if args.cwd else ROOT
        for fp in args.attach:
            src = Path(fp)
            if not src.is_absolute():
                src = (base_dir / fp).resolve()
            if src.exists() and src.is_file():
                dst = attach_dir / src.name
                shutil.copy2(src, dst)
                attachments_rel.append(os.path.relpath(dst, ROOT))
            else:
                print(f"[warn] attachment not found: {fp}")

    entry_lines = []
    entry_lines.append(f"\n## {args.title}\n")
    entry_lines.append(f"- 时间：`{date_str} {time_str}`\n")
    if args.tags:
        entry_lines.append(f"- 标签：`{args.tags}`\n")
    if args.desc:
        entry_lines.append("\n**背景/说明**\n\n")
        entry_lines.append(args.desc + "\n")
    if args.cmd:
        entry_lines.append("\n**运行命令**\n\n")
        entry_lines.append(f"- 命令：`{args.cmd}`\n")
        if args.cwd:
            entry_lines.append(f"- 工作目录：`{args.cwd}`\n")
        if log_rel_path:
            entry_lines.append(f"- 输出日志：`{log_rel_path}`\n")
        if exit_code is not None:
            entry_lines.append(f"- 退出码：`{exit_code}`\n")

    if attachments_rel:
        entry_lines.append("\n**附加文件**\n\n")
        for rel in attachments_rel:
            entry_lines.append(f"- `{rel}`\n")

    entry_lines.append("\n**结果摘要（手动填写）**\n\n")
    entry_lines.append("- 关键指标：`ndead`, `nposterior`, `<nlike>`, `log(Z) ± err`\n")
    entry_lines.append("- 是否符合预期：Yes/No\n")

    entry_lines.append("\n**后续行动（手动填写）**\n\n")

    append_entry(day_file, "".join(entry_lines))
    print(f"Logged entry to {day_file}")
    if log_rel_path:
        print(f"Captured command output to {log_rel_path}")


if __name__ == "__main__":
    main()