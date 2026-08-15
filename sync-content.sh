#!/bin/bash
# Dong bo wiki/ (gom ca assignments/, khac essays/ an cua DBC - xem CLAUDE.md quyet dinh
# 2026-08-14: assignments/ PUBLISH cong khai) tu course root vao content/ cua Quartz.
# Chay lai script nay + `git add -A && git commit && git push` moi khi wiki duoc cap nhat
# va muon publish ban moi.
set -euo pipefail

SRC="/Volumes/DATA/.CloudStorage/Data/OneDrive2-Personal/MARYAM MIRADI/AI Solutions Mastery/wiki"
DEST="$(cd "$(dirname "$0")" && pwd)/content"

rm -rf "$DEST"
mkdir -p "$DEST"

for d in sources lectures concepts people assignments self-test synthesis; do
  if [ -d "$SRC/$d" ]; then
    cp -R "$SRC/$d" "$DEST/$d"
  fi
done

# overview.md la trang hub -> lam index.md, giu alias "overview" de wikilink [[overview]] cu van resolve
cp "$SRC/overview.md" "$DEST/index.md"
python3 - "$DEST/index.md" << 'EOF'
import sys, re
path = sys.argv[1]
text = open(path, encoding="utf-8").read()
if "aliases:" not in text:
    marker = "aliases: ['overview']\n"
    text = re.sub(r"(^---\n)", lambda m: m.group(1) + marker, text, count=1)
open(path, "w", encoding="utf-8").write(text)
EOF

echo "Synced. Files in content/:"
find "$DEST" -name "*.md" | wc -l
