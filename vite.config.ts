import { defineConfig } from "vite";

export default defineConfig({
  // Project site under https://<user>.github.io/math-webgpu/
  base: "/math-webgpu/",
  root: ".",
  server: { port: 5173, host: true },
  preview: { port: 4173, host: true },
});
