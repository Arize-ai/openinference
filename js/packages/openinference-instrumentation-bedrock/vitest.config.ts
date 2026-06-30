import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    globals: true,
    environment: "node",
    include: ["test/**/*.test.ts"],
    typecheck: {
      enabled: true,
    },
    server: {
      deps: {
        inline: [/@aws-sdk/],
      },
    },
  },
});
