// Build configuration for the MyaScreen front end (TanStack Start + React + Tailwind).
// `@lovable.dev/vite-tanstack-config` is a plain Vite preset that bundles the TanStack Start,
// React, Tailwind, tsconfig-paths and Nitro plugins. It works outside Lovable too; do not add
// those plugins again here or they will be registered twice.
import { defineConfig } from "@lovable.dev/vite-tanstack-config";

export default defineConfig({
  tanstackStart: {
    // Use src/server.ts (the SSR error wrapper) as the server entry.
    server: { entry: "server" },
  },
});
