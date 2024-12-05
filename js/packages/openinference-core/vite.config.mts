import path, { resolve } from "path";
import { defineConfig } from "vite";
import { globSync } from "glob";
import { fileURLToPath } from "url";

export default defineConfig({
  build: {
    lib: {
      // Map every source file as an entry point so that they are not bundled, each file becomes its
      // own chunk essentially.
      // see https://rollupjs.org/configuration-options/#input
      entry: Object.fromEntries(
        globSync("src/**/*.ts").map((file) => [
          // This remove `src/` as well as the file extension from each
          // file, so e.g. src/nested/foo.js becomes nested/foo
          path.relative(
            "src",
            file.slice(0, file.length - path.extname(file).length),
          ),
          // This expands the relative paths to absolute paths, so e.g.
          // src/nested/foo becomes /project/src/nested/foo.js
          fileURLToPath(new URL(file, import.meta.url)),
        ]),
      ),
      formats: ["es", "cjs"],
    },
    minify: false,
    rollupOptions: {
      output: [
        {
          format: "es",
          dir: resolve(__dirname, "dist", "esm"),
        },
        {
          format: "esm",
          dir: resolve(__dirname, "dist", "esnext"),
          entryFileNames: "[name].js",
        },
        {
          format: "cjs",
          dir: resolve(__dirname, "dist", "src"),
        },
      ],
      // make sure to externalize deps that shouldn't be bundled
      // into your library
      external: ["@opentelemetry/api", "@opentelemetry/core"],
    },
  },
});
