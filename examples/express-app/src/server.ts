import { loadSemaArtifact } from "@semantscript/core";

import { createApp } from "./app.js";

// No path: the runtime uses SEMANTSCRIPT_ARTIFACT when set, else it searches
// .semantscript/artifact upward from this compiled file and from the working
// directory, so the artifact deployed beside dist/ is found without setup.
await loadSemaArtifact();
const app = await createApp();
const port = Number(process.env["PORT"] ?? 3000);
app.listen(port, () => {
  console.log(
    `ticket API listening on http://localhost:${port} (POST /tickets, POST /refunds/:orderId)`,
  );
});
