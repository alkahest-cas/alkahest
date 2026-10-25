#!/usr/bin/env bun

const ROOT = `${import.meta.dir}/src`;

Bun.serve({
  port: Number(process.env.PORT) || 3000,
  async fetch(req) {
    const url = new URL(req.url);
    const path = url.pathname.endsWith("/") ? url.pathname + "index.html" : url.pathname;
    const filePath = ROOT + path;
    const file = Bun.file(filePath);
    if (await file.exists()) {
      return new Response(file);
    }
    return new Response("Not found", { status: 404 });
  },
});

console.log(`http://localhost:${Number(process.env.PORT) || 3000}`);
