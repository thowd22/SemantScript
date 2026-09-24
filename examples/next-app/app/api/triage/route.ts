import { NextResponse } from "next/server";
import { loadSemaArtifact } from "@semantscript/core";

import { triage } from "@/lib/triage.sem";

// Load the trained artifact once per server process, in the module that calls
// sema. (With @semantscript/core installed from the registry, `instrumentation.ts`
// is the usual place; in this repository the runtime is linked from source, and
// Turbopack bundles a linked package per entry, so the route loads it itself.)
const artifactReady = loadSemaArtifact(
  process.env.SEMANTSCRIPT_ARTIFACT ?? "artifact",
);

export async function POST(request: Request): Promise<NextResponse> {
  await artifactReady;
  const { subject, body } = (await request.json()) as {
    subject?: string;
    body?: string;
  };
  if (typeof subject !== "string" || typeof body !== "string") {
    return NextResponse.json(
      { error: "subject and body are required" },
      { status: 400 },
    );
  }
  return NextResponse.json(
    { subject, priority: triage(subject, body) },
    { status: 201 },
  );
}
