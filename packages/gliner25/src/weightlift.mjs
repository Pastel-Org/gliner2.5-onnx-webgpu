/**
 * Weightlift ModelDefinition for GLiNER2.5 ONNX.
 *
 *   import { ModelManager } from "weightlift";
 *   import { glinerModel } from "./weightlift.mjs";
 *
 *   const models = new ModelManager({
 *     models: { gliner: glinerModel({ size: "small", ort, tokenizerFromPretrained }) },
 *   });
 *   const gliner = await models.load("gliner");
 *   await gliner.extract_entities(text, ["person", "organization"]);
 *
 * Weightlift only owns lifecycle + byte progress. This file is the adapter.
 * The playground at weightlift.dev can import `glinerModel` the same way it
 * imports `transformersModel`; that is a playground change, not a graph change.
 */

import { GLINER_MODELS, hfFileUrl, downloadModel } from "./gliner-boundary.mjs";
import { Gliner25 } from "./api.mjs";

function tokenizeWord(tokenizer) {
  return (token) => {
    const ids = tokenizer.encode(token, { add_special_tokens: false });
    return Array.isArray(ids) ? ids : Array.from(ids.ids ?? []);
  };
}

/**
 * @param {object} opts
 * @param {"small"|"base"|"multi"} [opts.size]
 * @param {any} opts.ort  onnxruntime-web module
 * @param {(repo: string) => Promise<any>} opts.tokenizerFromPretrained
 * @param {string} [opts.onnxPath]
 */
export function glinerModel({
  size = "small",
  ort,
  tokenizerFromPretrained,
  onnxPath = "onnx/model.onnx",
} = {}) {
  if (!ort) throw new Error("glinerModel requires onnxruntime-web as `ort`");
  if (!tokenizerFromPretrained) throw new Error("glinerModel requires tokenizerFromPretrained(repo)");
  const meta = GLINER_MODELS[size];
  if (!meta) throw new Error(`unknown size ${size}`);
  const url = hfFileUrl(meta.repo, onnxPath) + (onnxPath.includes("?") ? "" : "?v=4");

  return {
    async isCached() {
      try {
        if (!globalThis.caches) return false;
        const cache = await caches.open("gliner25-onnx");
        const hit = await cache.match(url);
        return Boolean(hit && hit.ok);
      } catch {
        return false;
      }
    },

    async load({ progress }) {
      progress.dispatch({ type: "start" });
      progress.dispatch({ type: "initiate", file: onnxPath });

      try {
        const bytes = await downloadModel(url, {
          onProgress: (loaded, total) => {
            progress.dispatch({
              type: "progress",
              file: onnxPath,
              loaded,
              total: total || undefined,
            });
            if (total) progress.dispatch({ type: "progress_total", loaded, total });
          },
        });
        progress.dispatch({ type: "done", file: onnxPath });

        progress.dispatch({ type: "initiate", file: "tokenizer" });
        const tokenizer = await tokenizerFromPretrained(meta.repo);
        progress.dispatch({ type: "done", file: "tokenizer" });

        const providers = [];
        if (globalThis.navigator?.gpu) providers.push("webgpu");
        providers.push("wasm");
        const session = await ort.InferenceSession.create(bytes.buffer, {
          executionProviders: providers,
        });

        let headsSession = null;
        const headsUrl = hfFileUrl(meta.repo, "onnx/heads.onnx") + "?v=4";
        try {
          progress.dispatch({ type: "initiate", file: "onnx/heads.onnx" });
          const headsBytes = await downloadModel(headsUrl, {
            onProgress: (loaded, total) => {
              progress.dispatch({
                type: "progress",
                file: "onnx/heads.onnx",
                loaded,
                total: total || undefined,
              });
            },
          });
          progress.dispatch({ type: "done", file: "onnx/heads.onnx" });
          headsSession = await ort.InferenceSession.create(headsBytes.buffer, {
            executionProviders: providers,
          });
        } catch {
          headsSession = null;
        }

        progress.dispatch({ type: "ready" });
        return new Gliner25({
          ort,
          session,
          tokenize: tokenizeWord(tokenizer),
          headsSession,
        });
      } catch (error) {
        const err = error instanceof Error ? error : new Error(String(error));
        progress.dispatch({ type: "error", error: err });
        throw err;
      }
    },

    async dispose(model) {
      try {
        await model.session?.release?.();
      } catch { /* already released */ }
    },
  };
}
