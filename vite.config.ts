import { defineConfig } from "vite";

export default defineConfig({
  // Relative base works for GitHub Pages project sites and Cursor hosting root.
  base: "./",
  root: ".",
  server: { port: 5173, host: "127.0.0.1" },
  preview: { port: 4173, host: "127.0.0.1" },
});
