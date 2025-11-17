rsync -av \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '.ipynb_checkpoints' \
  --exclude 'worklog/' \
  "$(pwd)/" yifan_sirius:/home/zhuxing/vega/