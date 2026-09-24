// tsserver loads plugins with require() and expects the module itself to be the
// factory function, so this CommonJS entry unwraps the ESM default export.
import plugin = require("./ts-plugin.js");

export = plugin.default;
