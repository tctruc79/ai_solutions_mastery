#!/usr/bin/env node
// Quartz nhung san `data-basepath="<pathname cua cfg.baseUrl>"` (vd "/32dbc_wiki") vao
// <body> moi trang, va cac plugin client-side (Explorer, Graph, Search, stacked-pages...)
// dung gia tri nay de dung link/fetch tuyet doi: `basepath + "/" + path`. Vi build-site.sh
// dung chung 1 quartz.config.yaml (1 baseUrl) cho ca 2 cay VI (/vi/) va EN (/en/), gia tri
// nhung san bi thieu doan "/vi" hoac "/en" -- khien Explorer/Graph/Search fetch/link sai
// duong dan (404) sau khi tach 2 cay ra khoi baseUrl goc. Quartz build CLI khong co flag
// override baseUrl, nen phai patch lai sau build. Script nay chay sau moi `npx quartz
// build`, cong them `suffix` ("/vi" hoac "/en") vao gia tri data-basepath da nhung san
// trong tung file HTML cua CHINH cay do.
import { readdir, readFile, writeFile } from "fs/promises"
import { join } from "path"

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
  const [targetDir, suffix] = process.argv.slice(2)
  if (!targetDir || !suffix) {
    console.error("Usage: node fix-basepath.mjs <build-output-dir> <suffix e.g. /vi>")
    process.exit(1)
  }
  const files = await walk(targetDir)
  let patched = 0
  for (const file of files) {
    const html = await readFile(file, "utf8")
    const next = html.replace(
      /data-basepath="([^"]*)"/,
      (_match, basepath) => `data-basepath="${basepath}${suffix}"`,
    )
    if (next === html) continue
    await writeFile(file, next, "utf8")
    patched++
  }
  console.log(`Patched data-basepath (+${suffix}) in ${patched}/${files.length} HTML files in ${targetDir}`)
}

main()
