// A compiled-program stand-in for `semantscript explain`: each export reaches
// the runtime the way rewritten sema sites do.
import { __sema } from "@semantscript/core";

export const fixtureFunctionId = `nf_${"1".repeat(64)}`;
export const changedFunctionId = `nf_${"9".repeat(64)}`;

export function decide(a, b) {
  return __sema.call(fixtureFunctionId, { facts: { a, b } });
}

/** The same site after its expression changed: the artifact has not been retrained. */
export function decideChanged(a, b) {
  return __sema.call(changedFunctionId, { facts: { a, b } });
}

export function noSema(text) {
  return text.toUpperCase();
}
