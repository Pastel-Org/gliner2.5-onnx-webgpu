#!/usr/bin/env node
/**
 * Assemble the Cloudflare Pages deploy directory (_site/).
 * Zero dependencies: plain fs.cp. Layout:
 *
 *   _site/index.html                     <- apps/demo/index.html
 *   _site/tokens.css, demo-presets.mjs   <- apps/demo/*
 *   _site/packages/gliner25/src/*        <- package source (demo imports it)
 *
 * Pages direct-upload does not honor .gitignore, so _site/ must contain ONLY
 * these files. Never point wrangler at the repo root.
 */
import { rm, cp, mkdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const site = path.join(root, "_site");

await rm(site, { recursive: true, force: true });
await mkdir(site, { recursive: true });
await cp(path.join(root, "apps", "demo"), site, { recursive: true });
await cp(
  path.join(root, "packages", "gliner25", "src"),
  path.join(site, "packages", "gliner25", "src"),
  { recursive: true },
);
console.log(`_site assembled: ${site}`);
