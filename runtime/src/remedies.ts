import { REMEDY_TEMPLATES } from "./remedies.generated.js";

/** Every remedy a tool prints, keyed by its id in `diagnostics/remedies.json`. */
export type RemedyId = keyof typeof REMEDY_TEMPLATES;

/** The placeholders one remedy's fix template names. */
export type RemedyParams<Id extends RemedyId> = Readonly<
  Record<(typeof REMEDY_TEMPLATES)[Id]["params"][number], string | number>
>;

/** The family a remedy belongs to: verifier, trainer-process, runtime or cli. */
export function remedyFamily(id: RemedyId): string {
  return template(id).family;
}

/**
 * The fix text a tool prints for a failure: the template from
 * `diagnostics/remedies.json` with every `{name}` replaced by `params[name]`.
 * An unknown id, a missing parameter or one the template does not name is a
 * programming error and throws.
 */
export function remedy<Id extends RemedyId>(
  id: Id,
  ...args: (typeof REMEDY_TEMPLATES)[Id]["params"] extends readonly []
    ? []
    : [params: RemedyParams<Id>]
): string {
  const entry = template(id);
  const params: Readonly<Record<string, string | number>> = args[0] ?? {};
  const names: readonly string[] = entry.params;
  for (const key of Object.keys(params)) {
    if (!names.includes(key)) {
      throw new TypeError(`remedy ${id} has no parameter ${key}`);
    }
  }
  return entry.fix.replace(/\{([a-z][a-zA-Z0-9]*)\}/gu, (_match, name) => {
    const value = params[name as string];
    if (value === undefined) {
      throw new TypeError(`remedy ${id} needs parameter ${String(name)}`);
    }
    return String(value);
  });
}

function template(id: string): {
  readonly family: string;
  readonly fix: string;
  readonly params: readonly string[];
} {
  if (!Object.hasOwn(REMEDY_TEMPLATES, id)) {
    throw new TypeError(`unknown remedy ${JSON.stringify(id)}`);
  }
  return REMEDY_TEMPLATES[id as RemedyId];
}
