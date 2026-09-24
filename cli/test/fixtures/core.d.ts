declare const boundedIntKind: unique symbol;
export type BoundedInt<Min extends number, Max extends number> = number & {
  readonly [boundedIntKind]?: readonly [Min, Max];
};
interface SemaExample<T> {
  readonly inputs: Readonly<Record<string, unknown>>;
  readonly output: T;
}
declare const semaConstraintKind: unique symbol;
interface SemaConstraint<T> {
  readonly [semaConstraintKind]: T;
}
interface SemaOptions<T> {
  readonly examples?: readonly SemaExample<T>[];
  readonly constraints?: readonly SemaConstraint<unknown>[];
}
interface ConfiguredSemaTag<T> {
  (strings: TemplateStringsArray, ...inputs: unknown[]): T;
}
interface ConfiguredDiagnosticSemaTag<T> {
  (strings: TemplateStringsArray, ...inputs: unknown[]): { value: T };
}
interface DiagnosticSemaTag {
  <T>(strings: TemplateStringsArray, ...inputs: unknown[]): { value: T };
  <T>(options: SemaOptions<T>): ConfiguredDiagnosticSemaTag<T>;
}
interface SemaTag {
  <T>(strings: TemplateStringsArray, ...inputs: unknown[]): T;
  <T>(options: SemaOptions<T>): ConfiguredSemaTag<T>;
  readonly withConfidence: DiagnosticSemaTag;
}
export declare const sema: SemaTag;
export declare function always<T>(
  predicate: () => boolean,
  output: T,
): SemaConstraint<T>;
export declare function never<T>(
  predicate: () => boolean,
  output: T,
): SemaConstraint<T>;
export declare const __sema: {
  call<T>(id: string, inputs: Readonly<Record<string, unknown>>): T;
};
