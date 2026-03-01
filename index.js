// Re-export from compiled dist for npm installs.
// OpenClaw's plugin loader looks for index.ts/index.js in the plugin root
// but npm packages place the build output in dist/.
export * from "./dist/index.js";
export { default } from "./dist/index.js";
