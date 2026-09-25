import { loadSemaArtifact } from "@semantscript/core";

import { createApp } from "./app.js";

// No path: SEMANTSCRIPT_ARTIFACT when set, else .semantscript/artifact found
// upward from this file or the working directory.
await loadSemaArtifact();
const app = await createApp();
app.listen(3000, () => {
  console.log("refund service listening on http://localhost:3000");
});
