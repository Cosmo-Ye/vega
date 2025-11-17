DEST="/Users/zhengyifan/vega_result"
PRE_LIST="$(mktemp)"
find "$DEST" -type f -name "*.fits" > "$PRE_LIST"
rsync -av --exclude '.git' --exclude '__pycache__' --exclude '.ipynb_checkpoints' yifan_sirius:/work/zhuxing/vega_result/ "$DEST/"
POST_LIST="$(mktemp)"
find "$DEST" -type f -name "*.fits" > "$POST_LIST"
sort "$PRE_LIST" -o "$PRE_LIST"
sort "$POST_LIST" -o "$POST_LIST"
NEW_FILES="$(comm -13 "$PRE_LIST" "$POST_LIST")"
rm -f "$PRE_LIST" "$POST_LIST"
if [ -n "$NEW_FILES" ]; then
  while IFS= read -r f; do
    ECSV="${f%.fits}.ecsv"
    python -c 'from astropy.io import fits; from astropy.table import Table; import sys; Table(fits.open(sys.argv[1])[2].data).write(sys.argv[2], format="ascii.ecsv", overwrite=True)' "$f" "$ECSV"
  done <<< "$NEW_FILES"
fi
