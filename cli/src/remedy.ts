import { remedy, type RemedyId } from "@semantscript/compiler";

/**
 * The fix text for one failure, from `diagnostics/remedies.json` through the
 * compiler's generated table (a pure module, so the CLI prints it even when
 * the runtime's native bindings do not load).
 */
export function remedyText(
  id: RemedyId,
  params: Readonly<Record<string, string | number>> = {},
): string {
  return (
    remedy as (
      id: RemedyId,
      params?: Readonly<Record<string, string | number>>,
    ) => string
  )(id, params);
}
