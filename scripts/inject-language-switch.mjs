#!/usr/bin/env node
// Chen thanh chuyen ngon ngu (VI song ngu <-> EN) vao moi trang HTML da build.
// Ly do dung cach nay thay vi Quartz component/plugin "chuan": quartz.ts's `layout`
// override khong duoc build pipeline cua repo nay doc (loadQuartzConfig() tu goi
// loadQuartzLayout() noi bo, khong nhan override) -- xem log.md 2026-07-28 de biet chi tiet
// dieu tra. Script nay chay sau moi lan `npx quartz build`, doc lai --basepath tu attribute
// data-basepath ma Quartz da nhung san trong <body>, nen hoat dong dung voi ca 2 cay build
// (public/vi/ va public/en/) ma khong can biet dang build cay nao.
import { readdir, readFile, writeFile } from "fs/promises"
import { join } from "path"

const SNIPPET_MARKER = "id=\"lang-switch-bar\""
// Trang redirect o public/index.html (xem build-site.sh) khong phai trang Quartz that,
// khong co data-basepath -- bo qua, khong chen switcher vao do.
const SKIP_MARKER = "http-equiv=\"refresh\""

// LƯU Ý: data-basepath đọc từ document.body.dataset.basepath ĐÃ bao gồm sẵn suffix
// "/vi" hoặc "/en" (do fix-basepath.mjs cộng thêm vào baseUrl gốc trước bước này — xem
// comment trong fix-basepath.mjs). Vì vậy script bên dưới phải tách ngôn ngữ NGAY TỪ
// basepath (không phải từ phần path còn lại sau khi trừ basepath), rồi hoán đổi suffix
// "/vi" <-> "/en" ngay trong basepath để ra basepath của cây kia. Bug cũ: script từng coi
// basepath là gốc chung (không có /vi hay /en) và tự thêm "/en" hoặc "/vi" vào SAU
// basepath — với basepath đã có sẵn "/vi", kết quả bị lồng thành ".../vi/en/" (404).
// LƯU Ý MÀU SẮC: KHÔNG dùng var(--light)/var(--dark) của Quartz cho cặp
// background/text của nút — 2 biến này đổi Ý NGHĨA giữa light/dark mode (vd --light
// = màu NỀN TRANG, nên ở dark mode nó ra màu tối, khiến nút "biến mất" vì trùng màu nền
// trang — đã tận mắt kiểm tra bằng browser trên bản deploy thật, dark mode làm nút vô
// hình dù href vẫn đúng). Dùng "color: inherit" (luôn khớp màu chữ đọc được của theme
// hiện tại) + nền/viền xám bán trong suốt (rgba trung tính) để tương phản đủ ở CẢ 2 theme
// mà không cần biết đang ở theme nào.
// data-router-ignore: co che opt-out CHUAN cua Quartz SPA router (xem
// quartz/components/scripts/spa.inline.ts dong 32: `if ("routerIgnore" in a.dataset)
// return`). Khong co attribute nay, script SPA cua Quartz se bat moi click vao <a> cung
// origin (kiem tra bang hostname, khong biet gi ve /vi hay /en la 2 cay build rieng biet)
// roi tu fetch + micromorph noi dung thay vi cho trinh duyet tai lai trang binh thuong --
// fetch cheo sang cay kia chac chan fail (khac page-graph noi bo), khien click bi "nuot"
// im lang, URL khong doi. Da tan mat kiem tra tren ban deploy that: thieu attribute nay
// thi bam nut khong dieu huong duoc.
const SNIPPET = `<div id="lang-switch-bar" style="position:sticky;top:0;z-index:1000;display:flex;justify-content:flex-end;padding:.35rem .75rem;background:rgba(128,128,128,.08);border-bottom:1px solid rgba(128,128,128,.25);">
<a id="language-switch-link" href="#" data-router-ignore style="display:inline-flex;align-items:center;gap:.35rem;font-size:.85rem;font-weight:600;text-decoration:none;color:inherit;background:rgba(128,128,128,.14);border:1px solid rgba(128,128,128,.4);border-radius:8px;padding:.2rem .7rem;">🌐</a>
</div>
<script>(function(){function setup(){var link=document.getElementById("language-switch-link");if(!link)return;var basepath=document.body.dataset.basepath||"";var path=window.location.pathname;var rel=basepath&&path.indexOf(basepath)===0?path.slice(basepath.length):path;if(!rel||rel[0]!=="/")rel="/"+rel;var isVi=/\\/vi$/.test(basepath);var isEn=/\\/en$/.test(basepath);var counterpart;if(isVi){counterpart=basepath.replace(/\\/vi$/,"/en");link.textContent="🇬🇧 English";link.title="Switch to full-English version";}else if(isEn){counterpart=basepath.replace(/\\/en$/,"/vi");link.textContent="🇻🇳 Tiếng Việt";link.title="Chuyển sang bản song ngữ (VI)";}else{counterpart=basepath+"/en";link.textContent="🇬🇧 English";link.title="Switch to full-English version";}link.href=counterpart+rel;}setup();document.addEventListener("nav",setup);document.addEventListener("render",setup);})();</script>
`

async function walk(dir) {
  const entries = await readdir(dir, { withFileTypes: true })
  const files = []
  for (const entry of entries) {
    const full = join(dir, entry.name)
    if (entry.isDirectory()) {
      files.push(...(await walk(full)))
    } else if (entry.name.endsWith(".html")) {
      files.push(full)
    }
  }
  return files
}

async function main() {
  const targetDir = process.argv[2]
  if (!targetDir) {
    console.error("Usage: node inject-language-switch.mjs <build-output-dir>")
    process.exit(1)
  }
  const files = await walk(targetDir)
  let injected = 0
  for (const file of files) {
    const html = await readFile(file, "utf8")
    if (html.includes(SNIPPET_MARKER) || html.includes(SKIP_MARKER)) continue
    const patched = html.replace(/(<body[^>]*>)/, `$1\n${SNIPPET}`)
    if (patched === html) continue
    await writeFile(file, patched, "utf8")
    injected++
  }
  console.log(`Injected language switcher into ${injected}/${files.length} HTML files in ${targetDir}`)
}

main()
