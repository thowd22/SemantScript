// tsserver resolves a plugin by file layout (not the package "exports" map) and
// loads it with require(), so this CommonJS entry hands it the built factory.
module.exports = require("../dist/ts-plugin.cjs");
