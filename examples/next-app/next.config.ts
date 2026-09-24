import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // The sema build step: every .sem.ts module goes through the SemantScript loader.
  turbopack: {
    rules: {
      "*.sem.ts": { loaders: ["@semantscript/compiler/loader"] },
    },
  },
  webpack(config) {
    config.module.rules.push({
      test: /\.sem\.ts$/,
      use: "@semantscript/compiler/loader",
    });
    return config;
  },
  // The runtime ships native ONNX Runtime and tokenizer bindings; keep them out of
  // the server bundle. The two native packages are listed for the linked-from-source
  // layout of this repository (Next bundles a linked package's own imports); an
  // installed @semantscript/core needs only itself here.
  serverExternalPackages: [
    "@semantscript/core",
    "onnxruntime-node",
    "tokenizers",
  ],
};

export default nextConfig;
