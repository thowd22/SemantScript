import eslint from "@eslint/js";
import tseslint from "typescript-eslint";

export default tseslint.config(
  {
    ignores: ["**/dist/**", "**/node_modules/**"],
  },
  {
    files: [
      "eslint.config.js",
      "scripts/**/*.mjs",
      "**/test/**/*.mjs",
      "cli/bin/*.js",
    ],
    extends: [eslint.configs.recommended],
    languageOptions: {
      globals: {
        AbortController: "readonly",
        console: "readonly",
        fetch: "readonly",
        process: "readonly",
        Request: "readonly",
        Response: "readonly",
        structuredClone: "readonly",
        URL: "readonly",
      },
    },
  },
  {
    files: [
      "compiler/src/**/*.ts",
      "runtime/src/**/*.ts",
      "cli/src/**/*.ts",
      "framework/src/**/*.ts",
      "benchmarks/refund/src/**/*.ts",
    ],
    extends: [
      eslint.configs.recommended,
      ...tseslint.configs.strictTypeChecked,
    ],
    languageOptions: {
      parserOptions: {
        projectService: true,
        tsconfigRootDir: import.meta.dirname,
      },
    },
  },
);
