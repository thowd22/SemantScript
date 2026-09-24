import express from "express";
import { loadSemaArtifact } from "@semantscript/core";

import { triage } from "./triage.sem.js";

const app = express();
app.use(express.json());

app.post("/tickets", (request, response) => {
  const { subject, body } = request.body as { subject?: string; body?: string };
  if (typeof subject !== "string" || typeof body !== "string") {
    response.status(400).json({ error: "subject and body are required" });
    return;
  }
  const ticket = {
    id: crypto.randomUUID(),
    subject,
    body,
    priority: triage(subject, body),
  };
  response.status(201).json(ticket);
});

// No path: the runtime uses SEMANTSCRIPT_ARTIFACT when set, else it searches
// .semantscript/artifact upward from this compiled file and from the working
// directory, so the artifact deployed beside dist/ is found without setup.
await loadSemaArtifact();
app.listen(3000, () => {
  console.log("ticket API listening on http://localhost:3000 (POST /tickets)");
});
