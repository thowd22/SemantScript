import { semanticJsonSha256 } from "@semantscript/compiler";

import { REFUND_SUPPORT } from "./types.js";

const SHA256 = /^[a-f0-9]{64}$/u;

/**
 * The single benchmark task contract. Keep this byte-for-byte aligned with the
 * canonical policy and constraints in examples/refund.sem.ts. The diagnostic
 * benchmark program is checked against the complete template and constraints.
 */
export const REFUND_TASK_SPEC = Object.freeze({
  kind: "semantscript.refund-task-spec",
  version: 1,
  inputDomain: Object.freeze({
    customer: Object.freeze({
      priorRefunds: "number",
      tier: Object.freeze(["enterprise", "standard"] as const),
    }),
    order: Object.freeze({
      ageDays: "number",
      status: Object.freeze(["fraudulent", "paid"] as const),
      total: "number",
    }),
  }),
  outputSupport: Object.freeze([...REFUND_SUPPORT]),
  policy:
    "An order older than 90 days is always denied. A fraudulent order that is 90 days old or younger goes to review. A paid order outside its refund window is denied: enterprise customers get 60 days and everyone else gets 30, counting the window as inclusive. A paid order inside its window goes to review when the circumstances are suspicious, meaning five or more prior refunds, three or more prior refunds on an order of at least 1000, an order total of at least 5000, or a same-day cancellation (age at most 1 day) of an order of at least 2000 by a customer with two or more prior refunds. Otherwise a paid order inside its window is approved",
  template: Object.freeze([
    Object.freeze({
      kind: "text",
      text: "Apply our refund policy. An order older than 90 days is always denied. A fraudulent order that is 90 days old or younger goes to review. A paid order outside its refund window is denied: enterprise customers get 60 days and everyone else gets 30, counting the window as inclusive. A paid order inside its window goes to review when the circumstances are suspicious, meaning five or more prior refunds, three or more prior refunds on an order of at least 1000, an order total of at least 5000, or a same-day cancellation (age at most 1 day) of an order of at least 2000 by a customer with two or more prior refunds. Otherwise a paid order inside its window is approved.\nCustomer: ",
    }),
    Object.freeze({
      kind: "input",
      name: "customer",
    }),
    Object.freeze({
      kind: "text",
      text: "\nOrder: ",
    }),
    Object.freeze({
      kind: "input",
      name: "order",
    }),
    Object.freeze({
      kind: "text",
      text: "\n",
    }),
  ]),
  hardConstraints: Object.freeze([
    Object.freeze({
      kind: "never",
      output: "approve",
      predicate: Object.freeze({
        left: Object.freeze({
          node: "property",
          object: Object.freeze({
            name: "order",
            node: "input",
          }),
          property: "status",
        }),
        node: "binary",
        operator: "===",
        right: Object.freeze({
          node: "literal",
          value: "fraudulent",
        }),
      }),
      source: 'order.status === "fraudulent"',
    }),
    Object.freeze({
      kind: "always",
      output: "deny",
      predicate: Object.freeze({
        left: Object.freeze({
          node: "property",
          object: Object.freeze({
            name: "order",
            node: "input",
          }),
          property: "ageDays",
        }),
        node: "binary",
        operator: ">",
        right: Object.freeze({
          node: "literal",
          value: 90,
        }),
      }),
      source: "order.ageDays > 90",
    }),
    Object.freeze({
      kind: "always",
      output: "review",
      predicate: Object.freeze({
        left: Object.freeze({
          left: Object.freeze({
            node: "property",
            object: Object.freeze({
              name: "order",
              node: "input",
            }),
            property: "ageDays",
          }),
          node: "binary",
          operator: "<=",
          right: Object.freeze({
            node: "literal",
            value: 90,
          }),
        }),
        node: "binary",
        operator: "&&",
        right: Object.freeze({
          left: Object.freeze({
            node: "property",
            object: Object.freeze({
              name: "order",
              node: "input",
            }),
            property: "status",
          }),
          node: "binary",
          operator: "===",
          right: Object.freeze({
            node: "literal",
            value: "fraudulent",
          }),
        }),
      }),
      source: 'order.ageDays <= 90 && order.status === "fraudulent"',
    }),
    Object.freeze({
      kind: "always",
      output: "deny",
      predicate: Object.freeze({
        left: Object.freeze({
          left: Object.freeze({
            left: Object.freeze({
              node: "property",
              object: Object.freeze({
                name: "order",
                node: "input",
              }),
              property: "ageDays",
            }),
            node: "binary",
            operator: "<=",
            right: Object.freeze({
              node: "literal",
              value: 90,
            }),
          }),
          node: "binary",
          operator: "&&",
          right: Object.freeze({
            left: Object.freeze({
              node: "property",
              object: Object.freeze({
                name: "order",
                node: "input",
              }),
              property: "status",
            }),
            node: "binary",
            operator: "===",
            right: Object.freeze({
              node: "literal",
              value: "paid",
            }),
          }),
        }),
        node: "binary",
        operator: "&&",
        right: Object.freeze({
          left: Object.freeze({
            left: Object.freeze({
              left: Object.freeze({
                node: "property",
                object: Object.freeze({
                  name: "customer",
                  node: "input",
                }),
                property: "tier",
              }),
              node: "binary",
              operator: "===",
              right: Object.freeze({
                node: "literal",
                value: "standard",
              }),
            }),
            node: "binary",
            operator: "&&",
            right: Object.freeze({
              left: Object.freeze({
                node: "property",
                object: Object.freeze({
                  name: "order",
                  node: "input",
                }),
                property: "ageDays",
              }),
              node: "binary",
              operator: ">",
              right: Object.freeze({
                node: "literal",
                value: 30,
              }),
            }),
          }),
          node: "binary",
          operator: "||",
          right: Object.freeze({
            left: Object.freeze({
              left: Object.freeze({
                node: "property",
                object: Object.freeze({
                  name: "customer",
                  node: "input",
                }),
                property: "tier",
              }),
              node: "binary",
              operator: "===",
              right: Object.freeze({
                node: "literal",
                value: "enterprise",
              }),
            }),
            node: "binary",
            operator: "&&",
            right: Object.freeze({
              left: Object.freeze({
                node: "property",
                object: Object.freeze({
                  name: "order",
                  node: "input",
                }),
                property: "ageDays",
              }),
              node: "binary",
              operator: ">",
              right: Object.freeze({
                node: "literal",
                value: 60,
              }),
            }),
          }),
        }),
      }),
      source:
        'order.ageDays <= 90 &&\n          order.status === "paid" &&\n          ((customer.tier === "standard" && order.ageDays > 30) ||\n            (customer.tier === "enterprise" && order.ageDays > 60))',
    }),
    Object.freeze({
      kind: "always",
      output: "review",
      predicate: Object.freeze({
        left: Object.freeze({
          left: Object.freeze({
            left: Object.freeze({
              node: "property",
              object: Object.freeze({
                name: "order",
                node: "input",
              }),
              property: "status",
            }),
            node: "binary",
            operator: "===",
            right: Object.freeze({
              node: "literal",
              value: "paid",
            }),
          }),
          node: "binary",
          operator: "&&",
          right: Object.freeze({
            left: Object.freeze({
              left: Object.freeze({
                left: Object.freeze({
                  node: "property",
                  object: Object.freeze({
                    name: "customer",
                    node: "input",
                  }),
                  property: "tier",
                }),
                node: "binary",
                operator: "===",
                right: Object.freeze({
                  node: "literal",
                  value: "standard",
                }),
              }),
              node: "binary",
              operator: "&&",
              right: Object.freeze({
                left: Object.freeze({
                  node: "property",
                  object: Object.freeze({
                    name: "order",
                    node: "input",
                  }),
                  property: "ageDays",
                }),
                node: "binary",
                operator: "<=",
                right: Object.freeze({
                  node: "literal",
                  value: 30,
                }),
              }),
            }),
            node: "binary",
            operator: "||",
            right: Object.freeze({
              left: Object.freeze({
                left: Object.freeze({
                  node: "property",
                  object: Object.freeze({
                    name: "customer",
                    node: "input",
                  }),
                  property: "tier",
                }),
                node: "binary",
                operator: "===",
                right: Object.freeze({
                  node: "literal",
                  value: "enterprise",
                }),
              }),
              node: "binary",
              operator: "&&",
              right: Object.freeze({
                left: Object.freeze({
                  node: "property",
                  object: Object.freeze({
                    name: "order",
                    node: "input",
                  }),
                  property: "ageDays",
                }),
                node: "binary",
                operator: "<=",
                right: Object.freeze({
                  node: "literal",
                  value: 60,
                }),
              }),
            }),
          }),
        }),
        node: "binary",
        operator: "&&",
        right: Object.freeze({
          left: Object.freeze({
            left: Object.freeze({
              left: Object.freeze({
                left: Object.freeze({
                  node: "property",
                  object: Object.freeze({
                    name: "customer",
                    node: "input",
                  }),
                  property: "priorRefunds",
                }),
                node: "binary",
                operator: ">=",
                right: Object.freeze({
                  node: "literal",
                  value: 5,
                }),
              }),
              node: "binary",
              operator: "||",
              right: Object.freeze({
                left: Object.freeze({
                  left: Object.freeze({
                    node: "property",
                    object: Object.freeze({
                      name: "customer",
                      node: "input",
                    }),
                    property: "priorRefunds",
                  }),
                  node: "binary",
                  operator: ">=",
                  right: Object.freeze({
                    node: "literal",
                    value: 3,
                  }),
                }),
                node: "binary",
                operator: "&&",
                right: Object.freeze({
                  left: Object.freeze({
                    node: "property",
                    object: Object.freeze({
                      name: "order",
                      node: "input",
                    }),
                    property: "total",
                  }),
                  node: "binary",
                  operator: ">=",
                  right: Object.freeze({
                    node: "literal",
                    value: 1000,
                  }),
                }),
              }),
            }),
            node: "binary",
            operator: "||",
            right: Object.freeze({
              left: Object.freeze({
                node: "property",
                object: Object.freeze({
                  name: "order",
                  node: "input",
                }),
                property: "total",
              }),
              node: "binary",
              operator: ">=",
              right: Object.freeze({
                node: "literal",
                value: 5000,
              }),
            }),
          }),
          node: "binary",
          operator: "||",
          right: Object.freeze({
            left: Object.freeze({
              left: Object.freeze({
                left: Object.freeze({
                  node: "property",
                  object: Object.freeze({
                    name: "order",
                    node: "input",
                  }),
                  property: "ageDays",
                }),
                node: "binary",
                operator: "<=",
                right: Object.freeze({
                  node: "literal",
                  value: 1,
                }),
              }),
              node: "binary",
              operator: "&&",
              right: Object.freeze({
                left: Object.freeze({
                  node: "property",
                  object: Object.freeze({
                    name: "order",
                    node: "input",
                  }),
                  property: "total",
                }),
                node: "binary",
                operator: ">=",
                right: Object.freeze({
                  node: "literal",
                  value: 2000,
                }),
              }),
            }),
            node: "binary",
            operator: "&&",
            right: Object.freeze({
              left: Object.freeze({
                node: "property",
                object: Object.freeze({
                  name: "customer",
                  node: "input",
                }),
                property: "priorRefunds",
              }),
              node: "binary",
              operator: ">=",
              right: Object.freeze({
                node: "literal",
                value: 2,
              }),
            }),
          }),
        }),
      }),
      source:
        'order.status === "paid" &&\n          ((customer.tier === "standard" && order.ageDays <= 30) ||\n            (customer.tier === "enterprise" && order.ageDays <= 60)) &&\n          (customer.priorRefunds >= 5 ||\n            (customer.priorRefunds >= 3 && order.total >= 1000) ||\n            order.total >= 5000 ||\n            (order.ageDays <= 1 && order.total >= 2000 && customer.priorRefunds >= 2))',
    }),
    Object.freeze({
      kind: "always",
      output: "approve",
      predicate: Object.freeze({
        left: Object.freeze({
          left: Object.freeze({
            left: Object.freeze({
              node: "property",
              object: Object.freeze({
                name: "order",
                node: "input",
              }),
              property: "status",
            }),
            node: "binary",
            operator: "===",
            right: Object.freeze({
              node: "literal",
              value: "paid",
            }),
          }),
          node: "binary",
          operator: "&&",
          right: Object.freeze({
            left: Object.freeze({
              left: Object.freeze({
                left: Object.freeze({
                  node: "property",
                  object: Object.freeze({
                    name: "customer",
                    node: "input",
                  }),
                  property: "tier",
                }),
                node: "binary",
                operator: "===",
                right: Object.freeze({
                  node: "literal",
                  value: "standard",
                }),
              }),
              node: "binary",
              operator: "&&",
              right: Object.freeze({
                left: Object.freeze({
                  node: "property",
                  object: Object.freeze({
                    name: "order",
                    node: "input",
                  }),
                  property: "ageDays",
                }),
                node: "binary",
                operator: "<=",
                right: Object.freeze({
                  node: "literal",
                  value: 30,
                }),
              }),
            }),
            node: "binary",
            operator: "||",
            right: Object.freeze({
              left: Object.freeze({
                left: Object.freeze({
                  node: "property",
                  object: Object.freeze({
                    name: "customer",
                    node: "input",
                  }),
                  property: "tier",
                }),
                node: "binary",
                operator: "===",
                right: Object.freeze({
                  node: "literal",
                  value: "enterprise",
                }),
              }),
              node: "binary",
              operator: "&&",
              right: Object.freeze({
                left: Object.freeze({
                  node: "property",
                  object: Object.freeze({
                    name: "order",
                    node: "input",
                  }),
                  property: "ageDays",
                }),
                node: "binary",
                operator: "<=",
                right: Object.freeze({
                  node: "literal",
                  value: 60,
                }),
              }),
            }),
          }),
        }),
        node: "binary",
        operator: "&&",
        right: Object.freeze({
          node: "unary",
          operand: Object.freeze({
            left: Object.freeze({
              left: Object.freeze({
                left: Object.freeze({
                  left: Object.freeze({
                    node: "property",
                    object: Object.freeze({
                      name: "customer",
                      node: "input",
                    }),
                    property: "priorRefunds",
                  }),
                  node: "binary",
                  operator: ">=",
                  right: Object.freeze({
                    node: "literal",
                    value: 5,
                  }),
                }),
                node: "binary",
                operator: "||",
                right: Object.freeze({
                  left: Object.freeze({
                    left: Object.freeze({
                      node: "property",
                      object: Object.freeze({
                        name: "customer",
                        node: "input",
                      }),
                      property: "priorRefunds",
                    }),
                    node: "binary",
                    operator: ">=",
                    right: Object.freeze({
                      node: "literal",
                      value: 3,
                    }),
                  }),
                  node: "binary",
                  operator: "&&",
                  right: Object.freeze({
                    left: Object.freeze({
                      node: "property",
                      object: Object.freeze({
                        name: "order",
                        node: "input",
                      }),
                      property: "total",
                    }),
                    node: "binary",
                    operator: ">=",
                    right: Object.freeze({
                      node: "literal",
                      value: 1000,
                    }),
                  }),
                }),
              }),
              node: "binary",
              operator: "||",
              right: Object.freeze({
                left: Object.freeze({
                  node: "property",
                  object: Object.freeze({
                    name: "order",
                    node: "input",
                  }),
                  property: "total",
                }),
                node: "binary",
                operator: ">=",
                right: Object.freeze({
                  node: "literal",
                  value: 5000,
                }),
              }),
            }),
            node: "binary",
            operator: "||",
            right: Object.freeze({
              left: Object.freeze({
                left: Object.freeze({
                  left: Object.freeze({
                    node: "property",
                    object: Object.freeze({
                      name: "order",
                      node: "input",
                    }),
                    property: "ageDays",
                  }),
                  node: "binary",
                  operator: "<=",
                  right: Object.freeze({
                    node: "literal",
                    value: 1,
                  }),
                }),
                node: "binary",
                operator: "&&",
                right: Object.freeze({
                  left: Object.freeze({
                    node: "property",
                    object: Object.freeze({
                      name: "order",
                      node: "input",
                    }),
                    property: "total",
                  }),
                  node: "binary",
                  operator: ">=",
                  right: Object.freeze({
                    node: "literal",
                    value: 2000,
                  }),
                }),
              }),
              node: "binary",
              operator: "&&",
              right: Object.freeze({
                left: Object.freeze({
                  node: "property",
                  object: Object.freeze({
                    name: "customer",
                    node: "input",
                  }),
                  property: "priorRefunds",
                }),
                node: "binary",
                operator: ">=",
                right: Object.freeze({
                  node: "literal",
                  value: 2,
                }),
              }),
            }),
          }),
          operator: "!",
        }),
      }),
      source:
        'order.status === "paid" &&\n          ((customer.tier === "standard" && order.ageDays <= 30) ||\n            (customer.tier === "enterprise" && order.ageDays <= 60)) &&\n          !(\n            customer.priorRefunds >= 5 ||\n            (customer.priorRefunds >= 3 && order.total >= 1000) ||\n            order.total >= 5000 ||\n            (order.ageDays <= 1 && order.total >= 2000 && customer.priorRefunds >= 2)\n          )',
    }),
  ]),
} as const);

export const REFUND_TASK_SPEC_SHA256 = semanticJsonSha256(REFUND_TASK_SPEC);

export const REFUND_FUNCTION_ID =
  "nf_65e347f7dd8736c55d82e539be7ad005cedb3ca396dfa2ab89de619f77adfc9c" as const;
export const REFUND_FUNCTION_SEMANTIC_SHA256 =
  "f7efe891ae5e62b2f0dcec517e118482c995468aaafd813f263c4179a99e545f" as const;
export const REFUND_FUNCTION_BINDING = Object.freeze({
  id: REFUND_FUNCTION_ID,
  semanticSha256: REFUND_FUNCTION_SEMANTIC_SHA256,
});

export const REFUND_ARTIFACT_TRAINING_KEY_KIND =
  "semantscript.refund-artifact-training-key.v1" as const;

export interface RefundArtifactTrainingKeySources {
  readonly baseDatasetSha256: string;
  readonly adversarialDatasetSha256: string | null;
  readonly releaseVerificationPayloadSha256: string;
  readonly releaseVerificationAttestationSha256: string;
}

/** Derive the manifest training key from every training/release source identity. */
export function deriveRefundArtifactTrainingKeySha256(
  value: RefundArtifactTrainingKeySources,
): string;
export function deriveRefundArtifactTrainingKeySha256(value: unknown): string {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new TypeError(
      "refund artifact training-key sources must be an object",
    );
  }
  const sources = value as unknown as Record<string, unknown>;
  const keys = [
    "baseDatasetSha256",
    "adversarialDatasetSha256",
    "releaseVerificationPayloadSha256",
    "releaseVerificationAttestationSha256",
  ] as const;
  const names = Object.keys(sources);
  if (
    names.length !== keys.length ||
    keys.some((key) => !names.includes(key))
  ) {
    throw new TypeError(
      `refund artifact training-key sources must contain exactly ${keys.join(", ")}`,
    );
  }
  for (const key of [
    "baseDatasetSha256",
    "releaseVerificationPayloadSha256",
    "releaseVerificationAttestationSha256",
  ] as const) {
    if (typeof sources[key] !== "string" || !SHA256.test(sources[key])) {
      throw new TypeError(`${key} must be a lowercase SHA-256 digest`);
    }
  }
  if (
    sources["adversarialDatasetSha256"] !== null &&
    (typeof sources["adversarialDatasetSha256"] !== "string" ||
      !SHA256.test(sources["adversarialDatasetSha256"]))
  ) {
    throw new TypeError(
      "adversarialDatasetSha256 must be null or a lowercase SHA-256 digest",
    );
  }
  return semanticJsonSha256({
    kind: REFUND_ARTIFACT_TRAINING_KEY_KIND,
    function: REFUND_FUNCTION_BINDING,
    sources: {
      baseDatasetSha256: sources["baseDatasetSha256"],
      adversarialDatasetSha256: sources["adversarialDatasetSha256"],
      releaseVerificationPayloadSha256:
        sources["releaseVerificationPayloadSha256"],
      releaseVerificationAttestationSha256:
        sources["releaseVerificationAttestationSha256"],
    },
  });
}

/**
 * Publication-bound identities. These live outside the adapters so committed
 * records can be checked without creating an adapter/contracts import cycle.
 * A missing revision or artifact digest means the field is still required to
 * be immutable and SHA-shaped, but cannot be fixed before training/API use.
 */
export const REFUND_SYSTEM_PINS = Object.freeze({
  semantscript: Object.freeze({
    model: Object.freeze({
      provider: "semantscript",
      name: "refund-decision",
      version: "semantscript-artifact-v1",
      revision: null,
      artifactSha256: null,
    }),
    adapter: Object.freeze({
      name: "semantscript-node-artifact-runtime",
      version: "1",
    }),
  }),
  "ollama-1b": Object.freeze({
    model: Object.freeze({
      provider: "ollama",
      name: "qwen2.5:1.5b-instruct-q4_K_M",
      version: "qwen2.5",
      revision:
        "65ec06548149b04c096a120e4a6da9d4017ea809c91734ea5631e89f96ddc57b",
      artifactSha256:
        "183715c435899236895da3869489cc30ac241476b4971a20285b1a462818a5b4",
    }),
    adapter: Object.freeze({ name: "refund-ollama-json-schema", version: "2" }),
  }),
  "ollama-7b": Object.freeze({
    model: Object.freeze({
      provider: "ollama",
      name: "qwen2.5:7b-instruct-q4_K_M",
      version: "qwen2.5",
      revision:
        "845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e",
      artifactSha256:
        "2bada8a7450677000f678be90653b85d364de7db25eb5ea54136ada5f3933730",
    }),
    adapter: Object.freeze({ name: "refund-ollama-json-schema", version: "2" }),
  }),
  "structured-api": Object.freeze({
    model: Object.freeze({
      provider: "anthropic",
      name: "claude-sonnet-5",
      version: "claude-sonnet-5",
      revision: "claude-sonnet-5",
      artifactSha256: null,
    }),
    adapter: Object.freeze({
      name: "refund-anthropic-structured-output",
      version: "2",
    }),
  }),
  laya: Object.freeze({
    model: Object.freeze({
      provider: "huggingface",
      name: "convaiinnovations/laya-typed-decisions",
      version: "laya-typed-decisions",
      revision: "dd079950600224fb459af2a0cb1d74e1e57ee9cf",
      artifactSha256:
        "4fa56de72383a9d3efa9cfa78955733c81b9fc8067a587ca4beb82c78107a24e",
    }),
    adapter: Object.freeze({
      name: "refund-laya-typed-decisions",
      version: "2",
    }),
  }),
} as const);

export const REFUND_BASELINE_POLICY = Object.freeze({
  policy: REFUND_TASK_SPEC.policy,
  hardConstraints: Object.freeze([
    'An order whose status is "fraudulent" must never be classified as "approve".',
    'An order whose ageDays is greater than 90 must always be classified as "deny".',
    'An order whose ageDays is at most 90 and whose status is "fraudulent" must always be classified as "review".',
    'A "paid" order whose ageDays is at most 90 and outside its window (standard tier: ageDays greater than 30; enterprise tier: ageDays greater than 60) must always be classified as "deny".',
    'A "paid" order inside its window (standard tier: ageDays at most 30; enterprise tier: ageDays at most 60) must always be classified as "review" when priorRefunds is at least 5, or priorRefunds is at least 3 and total is at least 1000, or total is at least 5000, or ageDays is at most 1 and total is at least 2000 and priorRefunds is at least 2.',
    'A "paid" order inside its window that meets none of those suspicious conditions must always be classified as "approve".',
  ] as const),
});
