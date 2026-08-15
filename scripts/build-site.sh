#!/bin/bash
# Build ca 2 ban site: VI (song ngu, content/ -> public/vi/) va EN (hoan toan tieng Anh,
# content-en/ -> public/en/), them trang redirect o root tro sang /vi/, roi chen thanh
# chuyen ngon ngu vao moi trang HTML. Dung chung cho CI (deploy.yaml) va preview local.
set -euo pipefail
cd "$(dirname "$0")/.."

rm -rf public
npx quartz build -d content -o public/vi
npx quartz build -d content-en -o public/en

# Quartz nhung san data-basepath tu cfg.baseUrl (vd "/32dbc_wiki"), thieu doan "/vi" hay
# "/en" -- can vi ca 2 cay dung chung 1 baseUrl trong quartz.config.yaml. Vá lại ngay sau
# build, truoc khi Explorer/Graph/Search doc gia tri nay luc runtime (xem fix-basepath.mjs).
node scripts/fix-basepath.mjs public/vi /vi
node scripts/fix-basepath.mjs public/en /en

# Root "/" redirect sang "/vi/" (ban mac dinh) -- giu link cu/bookmark vao root van di duoc.
cat > public/index.html << 'EOF'
<!DOCTYPE html>
<html lang="vi"><head><meta charset="utf-8"/>
<meta http-equiv="refresh" content="0; url=./vi/"/>
<link rel="canonical" href="./vi/"/>
<title>32_DBC Wiki</title>
</head><body>
<p>Đang chuyển tới <a href="./vi/">bản song ngữ</a>… / Redirecting to the <a href="./vi/">bilingual version</a>…</p>
<script>location.replace("./vi/");</script>
</body></html>
EOF

# public/vi/ va public/en/ nam long trong public/, nen 1 lan quet la du
node scripts/inject-language-switch.mjs public

echo "Build xong: public/vi/ (VI song ngu) + public/en/ (EN) + redirect o public/index.html"
