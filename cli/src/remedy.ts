import type { RemedyId } from "@semantscript/core";

/**
 * The fix text for one failure, from `diagnostics/remedies.json` through the
 * runtime's generated table. Loaded on demand, like the rest of
 * `@semantscript/core`, so a CLI whose runtime package is broken still starts
 * (and `semantscript doctor` can say so).
 */
export async function remedyText(
  id: RemedyId,
  params: Readonly<Record<string, string | number>> = {},
): Promise<string> {
  const { remedy } = await import("@semantscript/core");
  return (
    remedy as (
      id: RemedyId,
      params?: Readonly<Record<string, string | number>>,
    ) => string
  )(id, params);
}
