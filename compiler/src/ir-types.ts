export type PrimitiveInputType = {
  readonly kind: "string" | "boolean" | "number" | "null";
};

export interface LiteralInputType {
  readonly kind: "literal";
  readonly value: string | number | boolean | null;
}

export interface EnumInputType {
  readonly kind: "enum";
  readonly name: string;
  readonly base: "string" | "number";
  readonly values: readonly string[] | readonly number[];
}

export interface ArrayInputType {
  readonly kind: "array";
  readonly items: InputType;
}

export interface TupleInputType {
  readonly kind: "tuple";
  readonly items: readonly InputType[];
}

export interface ObjectInputField {
  readonly name: string;
  readonly optional: boolean;
  readonly type: InputType;
}

export interface ObjectInputType {
  readonly kind: "object";
  readonly name: string;
  readonly fields: readonly ObjectInputField[];
}

export interface UnionInputType {
  readonly kind: "union";
  readonly variants: readonly InputType[];
}

export type InputType =
  | PrimitiveInputType
  | LiteralInputType
  | EnumInputType
  | ArrayInputType
  | TupleInputType
  | ObjectInputType
  | UnionInputType;

export interface InputEntry {
  readonly name: string;
  readonly index: number;
  readonly tsType: string;
  readonly type: InputType;
}

export interface BooleanHeadSpec {
  readonly kind: "nominal";
  readonly sourceKind: "boolean";
  readonly support: readonly [false, true];
}

export interface NominalStringHeadSpec {
  readonly kind: "nominal";
  readonly sourceKind: "string-union" | "string-enum";
  readonly support: readonly string[];
}

export interface NominalNumberHeadSpec {
  readonly kind: "nominal";
  readonly sourceKind: "number-enum";
  readonly support: readonly number[];
}

export interface OrdinalStringHeadSpec {
  readonly kind: "ordinal";
  readonly sourceKind: "ordinal-string";
  readonly support: readonly string[];
  readonly expectedValue: "zero-based-rank";
}

export interface OrdinalNumberHeadSpec {
  readonly kind: "ordinal";
  readonly sourceKind: "bounded-int" | "bounded-number";
  readonly minimum: string;
  readonly maximum: string;
  readonly step: string;
  readonly supportDecimal: readonly string[];
  readonly expectedValue: "numeric";
}

export type HeadSpec =
  | BooleanHeadSpec
  | NominalStringHeadSpec
  | NominalNumberHeadSpec
  | OrdinalStringHeadSpec
  | OrdinalNumberHeadSpec;

export interface ScalarOutputSpec {
  readonly kind: "scalar";
  readonly tsType: string;
  readonly head: HeadSpec;
}

export interface ObjectOutputField {
  readonly name: string;
  readonly head: HeadSpec;
}

export interface ObjectOutputSpec {
  readonly kind: "object";
  readonly tsType: string;
  readonly fields: readonly ObjectOutputField[];
}

export type OutputSpec = ScalarOutputSpec | ObjectOutputSpec;

export type TemplatePart =
  | { readonly kind: "text"; readonly text: string }
  | { readonly kind: "input"; readonly name: string };
