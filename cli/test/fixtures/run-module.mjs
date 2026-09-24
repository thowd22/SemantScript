// A compiled-program stand-in: it calls the runtime the way rewritten sema
// sites do, so `semantscript run` must have activated an artifact first.
import { __sema } from "@semantscript/core";

export const fixtureFunctionId = `nf_${"1".repeat(64)}`;

export function decide(a, b) {
  return __sema.call(fixtureFunctionId, { facts: { a, b } });
}

export async function decideLater(inputs) {
  return __sema.call(fixtureFunctionId, inputs);
}

export const notCallable = 42;
