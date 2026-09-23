import type { SourceNeuralFunctionIr } from "./source-ir.js";
import { compareBytes } from "./semantic-json.js";

const textEncoder = new TextEncoder();
const FUNCTION_ID = /^nf_[a-f0-9]{64}$/;

/** One data-flow edge into a named input of a neural function. */
export interface SourceDependency {
  readonly producerFunctionId: string;
  readonly consumerFunctionId: string;
  readonly consumerInput: string;
}

/** A minimum-depth group of functions whose declared dependencies are satisfied. */
export interface SourceExecutionStage {
  readonly index: number;
  readonly functionIds: readonly string[];
}

/** Human-readable data dependencies and their deterministic stage assignment. */
export interface SourceExecutionPlan {
  readonly stages: readonly SourceExecutionStage[];
  readonly dependencies: readonly SourceDependency[];
}

/** Versioned container for source neural-function records and their execution plan. */
export interface SourceIrBundle {
  readonly kind: "semantscript.ir-bundle";
  readonly bundleVersion: 1;
  readonly functions: readonly SourceNeuralFunctionIr[];
  readonly executionPlan: SourceExecutionPlan;
}

/**
 * Validates and snapshots an already-canonical set of records and execution plan.
 *
 * Function order is source path, line, column, then ID using UTF-8 byte ordering
 * for strings. Dependencies are ordered by consumer function, consumer input
 * position, then producer function. Stage indexes are dense and every function
 * is assigned to its minimum possible dependency depth.
 */
export function createSourceIrBundle(
  functions: readonly SourceNeuralFunctionIr[],
  executionPlan: SourceExecutionPlan,
): SourceIrBundle {
  const functionSnapshot = [...functions];
  const stageSnapshot = executionPlan.stages.map(({ index, functionIds }) => ({
    index,
    functionIds: [...functionIds],
  }));
  const dependencySnapshot = executionPlan.dependencies.map(
    ({ producerFunctionId, consumerFunctionId, consumerInput }) => ({
      producerFunctionId,
      consumerFunctionId,
      consumerInput,
    }),
  );

  const metadata = validateFunctions(functionSnapshot);
  validateDependencies(dependencySnapshot, metadata);
  validateStages(stageSnapshot, dependencySnapshot, metadata);

  return {
    kind: "semantscript.ir-bundle",
    bundleVersion: 1,
    functions: functionSnapshot,
    executionPlan: {
      stages: stageSnapshot,
      dependencies: dependencySnapshot,
    },
  };
}

interface FunctionMetadata {
  readonly byId: ReadonlyMap<string, SourceNeuralFunctionIr>;
  readonly rankById: ReadonlyMap<string, number>;
  readonly inputRankByFunctionId: ReadonlyMap<string, ReadonlyMap<string, number>>;
}

function validateFunctions(functions: readonly SourceNeuralFunctionIr[]): FunctionMetadata {
  const byId = new Map<string, SourceNeuralFunctionIr>();
  const rankById = new Map<string, number>();
  const inputRankByFunctionId = new Map<string, ReadonlyMap<string, number>>();

  for (const [index, record] of functions.entries()) {
    if (!hasSourceRecordDiscriminator(record)) {
      throw new TypeError("IR bundle functions must be source neural-function IR v1 records");
    }
    if (!FUNCTION_ID.test(record.id)) {
      throw new TypeError(`invalid neural-function ID ${JSON.stringify(record.id)}`);
    }
    if (byId.has(record.id)) {
      throw new RangeError(`duplicate neural-function ID ${JSON.stringify(record.id)}`);
    }
    if (
      record.source.path.length === 0 ||
      !Number.isSafeInteger(record.source.line) ||
      record.source.line < 1 ||
      !Number.isSafeInteger(record.source.column) ||
      record.source.column < 1
    ) {
      throw new TypeError(`neural function ${JSON.stringify(record.id)} has an invalid source location`);
    }

    const previous = functions[index - 1];
    if (previous !== undefined && compareFunctions(previous, record) >= 0) {
      throw new RangeError("IR bundle functions must be in canonical source order");
    }

    const inputRanks = new Map<string, number>();
    for (const [inputIndex, input] of record.inputs.entries()) {
      if (input.name.length === 0 || inputRanks.has(input.name)) {
        throw new RangeError(
          `neural function ${JSON.stringify(record.id)} has an empty or duplicate input name`,
        );
      }
      inputRanks.set(input.name, inputIndex);
    }

    byId.set(record.id, record);
    rankById.set(record.id, index);
    inputRankByFunctionId.set(record.id, inputRanks);
  }

  return { byId, rankById, inputRankByFunctionId };
}

function hasSourceRecordDiscriminator(record: {
  readonly kind: unknown;
  readonly irVersion: unknown;
  readonly stage: unknown;
}): boolean {
  return (
    record.kind === "semantscript.neural-function" &&
    record.irVersion === 1 &&
    record.stage === "source"
  );
}

function validateDependencies(
  dependencies: readonly SourceDependency[],
  metadata: FunctionMetadata,
): void {
  const seen = new Set<string>();

  for (const [index, dependency] of dependencies.entries()) {
    const producerRank = metadata.rankById.get(dependency.producerFunctionId);
    const consumerRank = metadata.rankById.get(dependency.consumerFunctionId);
    const inputRank = metadata.inputRankByFunctionId
      .get(dependency.consumerFunctionId)
      ?.get(dependency.consumerInput);

    if (producerRank === undefined) {
      throw new RangeError(
        `dependency references unknown producer ${JSON.stringify(dependency.producerFunctionId)}`,
      );
    }
    if (consumerRank === undefined) {
      throw new RangeError(
        `dependency references unknown consumer ${JSON.stringify(dependency.consumerFunctionId)}`,
      );
    }
    if (inputRank === undefined) {
      throw new RangeError(
        `dependency references unknown input ${JSON.stringify(dependency.consumerInput)} on ${JSON.stringify(dependency.consumerFunctionId)}`,
      );
    }
    if (dependency.producerFunctionId === dependency.consumerFunctionId) {
      throw new RangeError("a neural function cannot depend on itself");
    }

    const key = dependencyKey(dependency);
    if (seen.has(key)) {
      throw new RangeError("IR bundle dependencies must not contain duplicate labeled edges");
    }
    seen.add(key);

    const previous = dependencies[index - 1];
    if (previous !== undefined && compareDependencies(previous, dependency, metadata) >= 0) {
      throw new RangeError("IR bundle dependencies must be in canonical order");
    }
  }
}

function validateStages(
  stages: readonly SourceExecutionStage[],
  dependencies: readonly SourceDependency[],
  metadata: FunctionMetadata,
): void {
  if (metadata.byId.size === 0) {
    if (stages.length !== 0 || dependencies.length !== 0) {
      throw new RangeError("an empty IR bundle must have an empty execution plan");
    }
    return;
  }

  const stageByFunctionId = new Map<string, number>();
  for (const [stageOffset, stage] of stages.entries()) {
    if (stage.index !== stageOffset) {
      throw new RangeError("execution-plan stage indexes must be dense and start at zero");
    }
    if (stage.functionIds.length === 0) {
      throw new RangeError("execution-plan stages must not be empty");
    }

    let previousRank = -1;
    for (const functionId of stage.functionIds) {
      const rank = metadata.rankById.get(functionId);
      if (rank === undefined) {
        throw new RangeError(`execution-plan stage references unknown function ${JSON.stringify(functionId)}`);
      }
      if (stageByFunctionId.has(functionId)) {
        throw new RangeError(`function ${JSON.stringify(functionId)} appears in more than one stage`);
      }
      if (rank <= previousRank) {
        throw new RangeError("function IDs within a stage must be in canonical source order");
      }
      previousRank = rank;
      stageByFunctionId.set(functionId, stage.index);
    }
  }

  if (stageByFunctionId.size !== metadata.byId.size) {
    throw new RangeError("every IR bundle function must appear in exactly one execution-plan stage");
  }

  const requiredStage = new Map<string, number>();
  for (const functionId of metadata.byId.keys()) {
    requiredStage.set(functionId, 0);
  }
  for (const dependency of dependencies) {
    const producerStage = stageByFunctionId.get(dependency.producerFunctionId);
    const consumerStage = stageByFunctionId.get(dependency.consumerFunctionId);
    if (producerStage === undefined || consumerStage === undefined) {
      throw new RangeError("execution-plan dependency references an unassigned function");
    }
    if (producerStage >= consumerStage) {
      throw new RangeError("execution-plan dependencies must point from earlier to later stages");
    }
    requiredStage.set(
      dependency.consumerFunctionId,
      Math.max(requiredStage.get(dependency.consumerFunctionId) ?? 0, producerStage + 1),
    );
  }

  for (const [functionId, actualStage] of stageByFunctionId) {
    if (actualStage !== requiredStage.get(functionId)) {
      throw new RangeError(
        `function ${JSON.stringify(functionId)} is not assigned to its minimum dependency stage`,
      );
    }
  }
}

function compareFunctions(left: SourceNeuralFunctionIr, right: SourceNeuralFunctionIr): number {
  return (
    compareUtf8(left.source.path, right.source.path) ||
    left.source.line - right.source.line ||
    left.source.column - right.source.column ||
    compareUtf8(left.id, right.id)
  );
}

function compareDependencies(
  left: SourceDependency,
  right: SourceDependency,
  metadata: FunctionMetadata,
): number {
  const leftConsumerRank = metadata.rankById.get(left.consumerFunctionId) ?? -1;
  const rightConsumerRank = metadata.rankById.get(right.consumerFunctionId) ?? -1;
  const leftInputRank = metadata.inputRankByFunctionId
    .get(left.consumerFunctionId)
    ?.get(left.consumerInput) ?? -1;
  const rightInputRank = metadata.inputRankByFunctionId
    .get(right.consumerFunctionId)
    ?.get(right.consumerInput) ?? -1;
  const leftProducerRank = metadata.rankById.get(left.producerFunctionId) ?? -1;
  const rightProducerRank = metadata.rankById.get(right.producerFunctionId) ?? -1;

  return (
    leftConsumerRank - rightConsumerRank ||
    leftInputRank - rightInputRank ||
    leftProducerRank - rightProducerRank
  );
}

function dependencyKey(dependency: SourceDependency): string {
  return `${dependency.producerFunctionId}\u0000${dependency.consumerFunctionId}\u0000${dependency.consumerInput}`;
}

function compareUtf8(left: string, right: string): number {
  return compareBytes(textEncoder.encode(left), textEncoder.encode(right));
}
