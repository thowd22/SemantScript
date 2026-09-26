// A serverless handler over the same compiled sema function. Deploy dist/,
// node_modules/ and .semantscript/artifact together (a zip or container image):
// `npx semantscript package --include deploy/lambda.mjs` writes that directory
// with this file at deploy/lambda.mjs (handler deploy/lambda.handler). The
// runtime finds the artifact by searching upward from the entry and the
// working directory, which is the bundle root on Lambda.
//
// The event shape follows API Gateway's HTTP API (v2): a JSON body with
// `subject` and `body`. The artifact loads once per execution environment, on
// the first invocation, which is the cold start measured in scripts/cold-start.mjs.
import { loadSemaArtifact } from "@semantscript/core";

import { triage } from "../dist/triage.sem.js";

const ready = loadSemaArtifact();

export async function handler(event) {
  await ready;
  let parsed;
  try {
    parsed = JSON.parse(event.body ?? "{}");
  } catch {
    return respond(400, { error: "body must be JSON" });
  }
  const { subject, body } = parsed;
  if (typeof subject !== "string" || typeof body !== "string") {
    return respond(400, { error: "subject and body are required" });
  }
  try {
    return respond(201, { subject, priority: triage(subject, body) });
  } catch (error) {
    // A runtime error (unknown function, low confidence without a fallback) is a 500.
    return respond(500, {
      error: error instanceof Error ? error.message : String(error),
    });
  }
}

function respond(statusCode, payload) {
  return {
    statusCode,
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  };
}
