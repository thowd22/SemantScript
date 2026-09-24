import ts from "typescript";

const DEFAULT_MAXIMUM_DEPTH = 100;
const DEFAULT_MAXIMUM_WORK = 100_000;
const textEncoder = new TextEncoder();

export interface ExecutionPlanInputDescriptor {
  readonly name: string;
  readonly expression: ts.Expression;
}

/** Compiler-owned description of one already-discovered and identified sema site. */
export interface ExecutionPlanSiteDescriptor {
  readonly functionId: string;
  readonly normalizedSourcePath: string;
  readonly start: number;
  readonly expression: ts.TaggedTemplateExpression;
  readonly inputs: readonly ExecutionPlanInputDescriptor[];
}

export interface ExecutionPlanDependency {
  readonly producerFunctionId: string;
  readonly consumerFunctionId: string;
  readonly consumerInput: string;
}

export interface ExecutionPlanStage {
  readonly index: number;
  readonly functionIds: readonly string[];
  /** The domain adapters this stage applies, in canonical order (routed plans only). */
  readonly adapterRefs?: readonly string[];
}

/** A compile-time routed domain: one adapter and one encoder depth for its functions. */
export interface ExecutionPlanDomain {
  readonly name: string;
  readonly adapterRef: string;
  readonly encoderRef: string;
  readonly encoderDepth: number | null;
  readonly functionIds: readonly string[];
}

export interface SemaExecutionPlan {
  readonly dependencies: readonly ExecutionPlanDependency[];
  readonly stages: readonly ExecutionPlanStage[];
  readonly domains?: readonly ExecutionPlanDomain[];
}

export interface BuildSemaExecutionPlanOptions {
  readonly maximumDepth?: number;
  readonly maximumWork?: number;
  readonly sourceFiles?: readonly ts.SourceFile[];
}

export type ExecutionPlanErrorCode =
  | "SEMA_EXECUTION_PLAN_BUDGET"
  | "SEMA_EXECUTION_PLAN_CYCLE"
  | "SEMA_EXECUTION_PLAN_INVALID";

export class ExecutionPlanError extends Error {
  readonly code: ExecutionPlanErrorCode;

  constructor(code: ExecutionPlanErrorCode, message: string) {
    super(message);
    this.name = "ExecutionPlanError";
    this.code = code;
  }
}

/** Builds the deterministic data-dependency plan for compiler-provided sema sites. */
export function buildSemaExecutionPlan(
  checker: ts.TypeChecker,
  descriptors: readonly ExecutionPlanSiteDescriptor[],
  options: BuildSemaExecutionPlanOptions = {},
): SemaExecutionPlan {
  const limits = resolveLimits(options);
  const sites = [...descriptors].sort(compareSites);
  validateDescriptors(sites);

  const sitesByExpression = new Map<
    ts.TaggedTemplateExpression,
    ExecutionPlanSiteDescriptor
  >();
  const sitesById = new Map<string, ExecutionPlanSiteDescriptor>();
  const siteRankById = new Map<string, number>();
  const inputRankBySiteId = new Map<string, ReadonlyMap<string, number>>();
  for (const [siteRank, site] of sites.entries()) {
    sitesByExpression.set(site.expression, site);
    sitesById.set(site.functionId, site);
    siteRankById.set(site.functionId, siteRank);
    inputRankBySiteId.set(
      site.functionId,
      new Map(site.inputs.map(({ name }, inputRank) => [name, inputRank])),
    );
  }

  const provenance = new ProvenanceResolver(
    checker,
    sitesByExpression,
    limits,
    options.sourceFiles ??
      sites.map(({ expression }) => expression.getSourceFile()),
  );
  const labelsByEdge = new Map<string, Set<string>>();
  for (const consumer of sites) {
    for (const input of consumer.inputs) {
      for (const producerId of provenance.forExpression(input.expression)) {
        provenance.consumeExpansion();
        const edgeKey = dependencyKey(producerId, consumer.functionId);
        let labels = labelsByEdge.get(edgeKey);
        if (labels === undefined) {
          labels = new Set<string>();
          labelsByEdge.set(edgeKey, labels);
        }
        labels.add(input.name);
      }
    }
  }

  const dependencies = [...labelsByEdge.entries()]
    .flatMap(([key, labels]) => {
      const separator = key.indexOf("\u0000");
      const producerFunctionId = key.slice(0, separator);
      const consumerFunctionId = key.slice(separator + 1);
      return [...labels].map(
        (consumerInput) =>
          ({
            producerFunctionId,
            consumerFunctionId,
            consumerInput,
          }) satisfies ExecutionPlanDependency,
      );
    })
    .sort((left, right) =>
      compareDependencies(left, right, siteRankById, inputRankBySiteId),
    );

  const stages = createStages(sites, dependencies, sitesById);
  return { dependencies, stages };
}

export interface RoutedFunctionDescriptor {
  readonly functionId: string;
  readonly domain: string;
  readonly adapterRef: string;
  readonly encoderRef: string;
  readonly encoderDepth: number | null;
}

/**
 * Records the routed domains on a plan: one entry per domain in first-function
 * order, and per stage the adapters it applies, so the runtime and trainer read
 * the routing from the bundle instead of recomputing it.
 */
export function routeExecutionPlan(
  plan: SemaExecutionPlan,
  functions: readonly RoutedFunctionDescriptor[],
): SemaExecutionPlan {
  const byId = new Map(functions.map((entry) => [entry.functionId, entry]));
  const domains = new Map<
    string,
    ExecutionPlanDomain & { readonly functionIds: string[] }
  >();
  for (const stage of plan.stages) {
    for (const functionId of stage.functionIds) {
      const entry = byId.get(functionId);
      if (entry === undefined) {
        throw new ExecutionPlanError(
          "SEMA_EXECUTION_PLAN_INVALID",
          `routed plan lacks a domain for function ${functionId}`,
        );
      }
      const existing = domains.get(entry.domain);
      if (existing === undefined) {
        domains.set(entry.domain, {
          name: entry.domain,
          adapterRef: entry.adapterRef,
          encoderRef: entry.encoderRef,
          encoderDepth: entry.encoderDepth,
          functionIds: [functionId],
        });
      } else {
        if (
          existing.adapterRef !== entry.adapterRef ||
          existing.encoderRef !== entry.encoderRef ||
          existing.encoderDepth !== entry.encoderDepth
        ) {
          throw new ExecutionPlanError(
            "SEMA_EXECUTION_PLAN_INVALID",
            `domain ${entry.domain} is routed inconsistently across its functions`,
          );
        }
        existing.functionIds.push(functionId);
      }
    }
  }
  const stages = plan.stages.map((stage) => ({
    ...stage,
    adapterRefs: [
      ...new Set(
        stage.functionIds.map(
          (functionId) => byId.get(functionId)?.adapterRef ?? "",
        ),
      ),
    ],
  }));
  return {
    dependencies: plan.dependencies,
    stages,
    domains: [...domains.values()],
  };
}

interface ResolvedLimits {
  readonly maximumDepth: number;
  readonly maximumWork: number;
}

function resolveLimits(options: BuildSemaExecutionPlanOptions): ResolvedLimits {
  return {
    maximumDepth: positiveSafeInteger(
      options.maximumDepth ?? DEFAULT_MAXIMUM_DEPTH,
      "maximumDepth",
    ),
    maximumWork: positiveSafeInteger(
      options.maximumWork ?? DEFAULT_MAXIMUM_WORK,
      "maximumWork",
    ),
  };
}

function positiveSafeInteger(value: number, name: string): number {
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new RangeError(`${name} must be a positive safe integer`);
  }
  return value;
}

function validateDescriptors(
  sites: readonly ExecutionPlanSiteDescriptor[],
): void {
  const ids = new Set<string>();
  const expressions = new Set<ts.TaggedTemplateExpression>();

  for (const site of sites) {
    if (
      site.functionId.length === 0 ||
      site.normalizedSourcePath.length === 0
    ) {
      invalid(
        "execution-plan sites require nonempty function IDs and normalized source paths",
      );
    }
    if (!Number.isSafeInteger(site.start) || site.start < 0) {
      invalid(
        `execution-plan site ${JSON.stringify(site.functionId)} has an invalid start`,
      );
    }
    if (ids.has(site.functionId)) {
      invalid(
        `duplicate execution-plan function ID ${JSON.stringify(site.functionId)}`,
      );
    }
    if (expressions.has(site.expression)) {
      invalid(
        `duplicate execution-plan expression for ${JSON.stringify(site.functionId)}`,
      );
    }
    ids.add(site.functionId);
    expressions.add(site.expression);

    const inputNames = new Set<string>();
    for (const input of site.inputs) {
      if (input.name.length === 0 || inputNames.has(input.name)) {
        invalid(
          `execution-plan site ${JSON.stringify(site.functionId)} has an empty or duplicate input name`,
        );
      }
      inputNames.add(input.name);
    }
  }
}

class ProvenanceResolver {
  readonly #checker: ts.TypeChecker;
  readonly #sitesByExpression: ReadonlyMap<
    ts.TaggedTemplateExpression,
    ExecutionPlanSiteDescriptor
  >;
  readonly #limits: ResolvedLimits;
  readonly #symbolMemo = new Map<ts.Symbol, ReadonlySet<string>>();
  readonly #activeSymbols: ts.Symbol[] = [];
  readonly #ignoredSymbols = new Set<ts.Symbol>();
  readonly #sourceFiles = new Set<ts.SourceFile>();
  readonly #indexedSourceFiles = new Set<ts.SourceFile>();
  readonly #writesBySymbol = new Map<ts.Symbol, AssignmentWrite[]>();
  #consumerExpression: ts.Expression | undefined;
  #work = 0;

  constructor(
    checker: ts.TypeChecker,
    sitesByExpression: ReadonlyMap<
      ts.TaggedTemplateExpression,
      ExecutionPlanSiteDescriptor
    >,
    limits: ResolvedLimits,
    sourceFiles: readonly ts.SourceFile[],
  ) {
    this.#checker = checker;
    this.#sitesByExpression = sitesByExpression;
    this.#limits = limits;
    for (const sourceFile of sourceFiles) {
      if (!sourceFile.isDeclarationFile) this.#sourceFiles.add(sourceFile);
    }
  }

  forExpression(expression: ts.Expression): ReadonlySet<string> {
    this.#symbolMemo.clear();
    this.#consumerExpression = expression;
    try {
      return this.#forNode(expression, 0);
    } finally {
      this.#consumerExpression = undefined;
    }
  }

  consumeExpansion(): void {
    this.#consumeWork();
  }

  #forNode(node: ts.Node, depth: number): ReadonlySet<string> {
    this.#consume(depth);

    if (ts.isTaggedTemplateExpression(node)) {
      const site = this.#sitesByExpression.get(node);
      if (site !== undefined) {
        return new Set([site.functionId]);
      }
    }

    if (ts.isIdentifier(node)) {
      const shorthand = ts.isShorthandPropertyAssignment(node.parent)
        ? this.#checker.getShorthandAssignmentValueSymbol(node.parent)
        : undefined;
      const symbol = shorthand ?? this.#checker.getSymbolAtLocation(node);
      if (symbol === undefined) return new Set();
      const resolved = resolveAliases(symbol, this.#checker);
      return this.#ignoredSymbols.has(resolved)
        ? new Set()
        : this.#forSymbol(resolved, depth + 1);
    }

    const result = new Set<string>();
    ts.forEachChild(node, (child) => {
      this.#merge(result, this.#forNode(child, depth + 1));
    });
    return result;
  }

  #forSymbol(input: ts.Symbol, depth: number): ReadonlySet<string> {
    this.#consume(depth);
    const symbol = resolveAliases(input, this.#checker);
    const memoized = this.#symbolMemo.get(symbol);
    if (memoized !== undefined) {
      return memoized;
    }

    const cycleStart = this.#activeSymbols.indexOf(symbol);
    if (cycleStart >= 0) {
      const cycle = [...this.#activeSymbols.slice(cycleStart), symbol]
        .map((entry) => entry.getName())
        .join(" -> ");
      throw new ExecutionPlanError(
        "SEMA_EXECUTION_PLAN_CYCLE",
        `initializer provenance cycle: ${cycle}`,
      );
    }

    this.#activeSymbols.push(symbol);
    const result = new Set<string>();
    try {
      for (const declaration of symbol.declarations ?? []) {
        const variable = containingInitializedVariable(declaration);
        if (variable?.initializer !== undefined) {
          this.#merge(result, this.#forNode(variable.initializer, depth + 1));
          for (const initializer of bindingInitializers(
            declaration,
            variable,
          )) {
            this.#merge(result, this.#forNode(initializer, depth + 1));
          }
        }
      }

      if ((symbol.declarations ?? []).some(isAssignmentTrackedDeclaration)) {
        this.#ignoredSymbols.add(symbol);
        try {
          for (const expression of this.#writeExpressions(symbol)) {
            this.#merge(result, this.#forNode(expression, depth + 1));
          }
        } finally {
          this.#ignoredSymbols.delete(symbol);
        }
      }
    } finally {
      this.#activeSymbols.pop();
    }

    if (this.#ignoredSymbols.size === 0) {
      this.#symbolMemo.set(symbol, result);
    }
    return result;
  }

  #consume(depth: number): void {
    if (depth > this.#limits.maximumDepth) {
      throw new ExecutionPlanError(
        "SEMA_EXECUTION_PLAN_BUDGET",
        `execution-plan provenance exceeds maximum depth ${String(this.#limits.maximumDepth)}`,
      );
    }
    this.#consumeWork();
  }

  #consumeWork(): void {
    this.#work += 1;
    if (this.#work > this.#limits.maximumWork) {
      throw new ExecutionPlanError(
        "SEMA_EXECUTION_PLAN_BUDGET",
        `execution-plan provenance exceeds maximum work ${String(this.#limits.maximumWork)}`,
      );
    }
  }

  #merge(target: Set<string>, source: ReadonlySet<string>): void {
    for (const value of source) {
      this.#consumeWork();
      target.add(value);
    }
  }

  #writeExpressions(symbol: ts.Symbol): readonly ts.Expression[] {
    for (const declaration of symbol.declarations ?? []) {
      const sourceFile = declaration.getSourceFile();
      if (!sourceFile.isDeclarationFile) this.#sourceFiles.add(sourceFile);
    }
    for (const sourceFile of this.#sourceFiles) {
      if (!this.#indexedSourceFiles.has(sourceFile))
        this.#indexWrites(sourceFile);
    }

    const consumer = this.#consumerExpression;
    const writes = this.#writesBySymbol.get(symbol) ?? [];
    return consumer === undefined
      ? writes.map(({ expression }) => expression)
      : writes
          .filter(({ node }) => writeMayReachConsumer(node, consumer))
          .map(({ expression }) => expression);
  }

  #indexWrites(sourceFile: ts.SourceFile): void {
    this.#indexedSourceFiles.add(sourceFile);
    const record = (target: ts.Node, expression: ts.Expression): void => {
      for (const symbol of assignmentTargetSymbols(target, this.#checker)) {
        const entries = this.#writesBySymbol.get(symbol) ?? [];
        entries.push({ expression, node: target });
        this.#writesBySymbol.set(symbol, entries);
      }
    };
    const visit = (node: ts.Node): void => {
      this.#consumeWork();
      if (
        ts.isBinaryExpression(node) &&
        node.operatorToken.kind >= ts.SyntaxKind.FirstAssignment &&
        node.operatorToken.kind <= ts.SyntaxKind.LastAssignment
      ) {
        record(node.left, node.right);
      } else if (
        (ts.isForInStatement(node) || ts.isForOfStatement(node)) &&
        !ts.isVariableDeclarationList(node.initializer)
      ) {
        record(node.initializer, node.expression);
      }
      ts.forEachChild(node, visit);
    };
    visit(sourceFile);
  }
}

interface AssignmentWrite {
  readonly expression: ts.Expression;
  readonly node: ts.Node;
}

function resolveAliases(symbol: ts.Symbol, checker: ts.TypeChecker): ts.Symbol {
  const visited = new Set<ts.Symbol>();
  let current = symbol;
  while (
    (current.flags & ts.SymbolFlags.Alias) !== 0 &&
    !visited.has(current)
  ) {
    visited.add(current);
    const resolved = checker.getAliasedSymbol(current);
    if (resolved === current) break;
    current = resolved;
  }
  return current;
}

function containingInitializedVariable(
  declaration: ts.Declaration,
): ts.VariableDeclaration | undefined {
  let current: ts.Node = declaration;
  while (
    ts.isBindingElement(current) ||
    ts.isArrayBindingPattern(current) ||
    ts.isObjectBindingPattern(current)
  ) {
    current = current.parent;
  }
  if (
    !ts.isVariableDeclaration(current) ||
    !ts.isVariableDeclarationList(current.parent)
  ) {
    return undefined;
  }
  return current;
}

function isAssignmentTrackedDeclaration(declaration: ts.Declaration): boolean {
  let current: ts.Node = declaration;
  while (
    ts.isBindingElement(current) ||
    ts.isArrayBindingPattern(current) ||
    ts.isObjectBindingPattern(current)
  ) {
    current = current.parent;
  }
  return ts.isVariableDeclaration(current) || ts.isParameter(current);
}

function bindingInitializers(
  declaration: ts.Declaration,
  variable: ts.VariableDeclaration,
): readonly ts.Expression[] {
  const result: ts.Expression[] = [];
  let current: ts.Node = declaration;
  while (current !== variable) {
    if (ts.isBindingElement(current) && current.initializer !== undefined) {
      result.push(current.initializer);
    }
    current = current.parent;
  }
  return result;
}

function assignmentTargetSymbols(
  target: ts.Node,
  checker: ts.TypeChecker,
): ReadonlySet<ts.Symbol> {
  const result = new Set<ts.Symbol>();
  const visit = (node: ts.Node): void => {
    if (ts.isIdentifier(node)) {
      const shorthand = ts.isShorthandPropertyAssignment(node.parent)
        ? checker.getShorthandAssignmentValueSymbol(node.parent)
        : undefined;
      const symbol = shorthand ?? checker.getSymbolAtLocation(node);
      if (symbol !== undefined) result.add(resolveAliases(symbol, checker));
      return;
    }
    if (
      ts.isPropertyAccessExpression(node) ||
      ts.isElementAccessExpression(node)
    ) {
      visit(node.expression);
      return;
    }
    if (
      ts.isParenthesizedExpression(node) ||
      ts.isAsExpression(node) ||
      ts.isTypeAssertionExpression(node) ||
      ts.isNonNullExpression(node)
    ) {
      visit(node.expression);
      return;
    }
    if (
      ts.isBinaryExpression(node) &&
      node.operatorToken.kind === ts.SyntaxKind.EqualsToken
    ) {
      visit(node.left);
      return;
    }
    if (ts.isArrayLiteralExpression(node)) {
      for (const element of node.elements) {
        if (!ts.isOmittedExpression(element)) visit(element);
      }
      return;
    }
    if (ts.isSpreadElement(node) || ts.isSpreadAssignment(node)) {
      visit(node.expression);
      return;
    }
    if (ts.isObjectLiteralExpression(node)) {
      for (const property of node.properties) {
        if (ts.isPropertyAssignment(property)) {
          visit(property.initializer);
        } else if (ts.isShorthandPropertyAssignment(property)) {
          visit(property.name);
        } else if (ts.isSpreadAssignment(property)) {
          visit(property.expression);
        }
      }
    }
  };
  visit(target);
  return result;
}

function writeMayReachConsumer(
  write: ts.Node,
  consumer: ts.Expression,
): boolean {
  const writeSource = write.getSourceFile();
  const consumerSource = consumer.getSourceFile();
  if (writeSource !== consumerSource) return true;

  const writeContainer = executionContainer(write);
  const consumerContainer = executionContainer(consumer);
  return (
    writeContainer !== consumerContainer ||
    write.getStart(writeSource) < consumer.getStart(consumerSource)
  );
}

function executionContainer(node: ts.Node): ts.Node {
  let current: ts.Node = node;
  while (!ts.isSourceFile(current)) {
    if (ts.isFunctionLike(current)) return current;
    current = current.parent;
  }
  return current;
}

function createStages(
  sites: readonly ExecutionPlanSiteDescriptor[],
  dependencies: readonly ExecutionPlanDependency[],
  sitesById: ReadonlyMap<string, ExecutionPlanSiteDescriptor>,
): readonly ExecutionPlanStage[] {
  const incoming = new Map<string, Set<string>>();
  const outgoing = new Map<string, Set<string>>();
  for (const site of sites) {
    incoming.set(site.functionId, new Set());
    outgoing.set(site.functionId, new Set());
  }
  for (const dependency of dependencies) {
    incoming
      .get(dependency.consumerFunctionId)
      ?.add(dependency.producerFunctionId);
    outgoing
      .get(dependency.producerFunctionId)
      ?.add(dependency.consumerFunctionId);
  }

  const remaining = new Map(
    sites.map((site) => [
      site.functionId,
      incoming.get(site.functionId)?.size ?? 0,
    ]),
  );
  const ready: ExecutionPlanSiteDescriptor[] = [];
  for (const site of sites) {
    if (remaining.get(site.functionId) === 0) pushReadySite(ready, site);
  }
  const topological: ExecutionPlanSiteDescriptor[] = [];
  const stageById = new Map<string, number>();

  while (ready.length > 0) {
    const site = popReadySite(ready);
    if (site === undefined) break;
    topological.push(site);
    let stage = 0;
    for (const predecessor of incoming.get(site.functionId) ?? []) {
      stage = Math.max(stage, (stageById.get(predecessor) ?? 0) + 1);
    }
    stageById.set(site.functionId, stage);

    const consumers = [...(outgoing.get(site.functionId) ?? [])]
      .map((id) => sitesById.get(id))
      .filter(
        (candidate): candidate is ExecutionPlanSiteDescriptor =>
          candidate !== undefined,
      )
      .sort(compareSites);
    for (const consumer of consumers) {
      const count = (remaining.get(consumer.functionId) ?? 0) - 1;
      remaining.set(consumer.functionId, count);
      if (count === 0) pushReadySite(ready, consumer);
    }
  }

  if (topological.length !== sites.length) {
    const cycle = findDependencyCycle(sites, outgoing, sitesById);
    throw new ExecutionPlanError(
      "SEMA_EXECUTION_PLAN_CYCLE",
      `sema dependency cycle: ${cycle.join(" -> ")}`,
    );
  }

  const grouped = new Map<number, ExecutionPlanSiteDescriptor[]>();
  for (const site of topological) {
    const stage = stageById.get(site.functionId) ?? 0;
    const entries = grouped.get(stage) ?? [];
    entries.push(site);
    grouped.set(stage, entries);
  }
  return [...grouped.entries()]
    .sort(([left], [right]) => left - right)
    .map(([index, entries]) => ({
      index,
      functionIds: entries
        .sort(compareSites)
        .map(({ functionId }) => functionId),
    }));
}

function pushReadySite(
  heap: ExecutionPlanSiteDescriptor[],
  site: ExecutionPlanSiteDescriptor,
): void {
  heap.push(site);
  let index = heap.length - 1;

  while (index > 0) {
    const parent = Math.floor((index - 1) / 2);
    const parentSite = heap[parent];
    if (parentSite === undefined || compareSites(parentSite, site) <= 0) break;
    heap[index] = parentSite;
    index = parent;
  }
  heap[index] = site;
}

function popReadySite(
  heap: ExecutionPlanSiteDescriptor[],
): ExecutionPlanSiteDescriptor | undefined {
  const first = heap[0];
  const last = heap.pop();
  if (first === undefined || last === undefined || heap.length === 0)
    return first;

  let index = 0;
  while (index < heap.length) {
    const left = index * 2 + 1;
    const right = left + 1;
    if (left >= heap.length) break;
    let child = left;
    const leftSite = heap[left];
    const rightSite = heap[right];
    if (
      leftSite === undefined ||
      (rightSite !== undefined && compareSites(rightSite, leftSite) < 0)
    ) {
      child = right;
    }
    const childSite = heap[child];
    if (childSite === undefined || compareSites(last, childSite) <= 0) break;
    heap[index] = childSite;
    index = child;
  }
  heap[index] = last;
  return first;
}

function findDependencyCycle(
  sites: readonly ExecutionPlanSiteDescriptor[],
  outgoing: ReadonlyMap<string, ReadonlySet<string>>,
  sitesById: ReadonlyMap<string, ExecutionPlanSiteDescriptor>,
): readonly string[] {
  const state = new Map<string, "active" | "complete">();
  const stack: string[] = [];
  const consumersById = new Map(
    sites.map((site) => [
      site.functionId,
      [...(outgoing.get(site.functionId) ?? [])]
        .map((id) => sitesById.get(id))
        .filter(
          (candidate): candidate is ExecutionPlanSiteDescriptor =>
            candidate !== undefined,
        )
        .sort(compareSites),
    ]),
  );

  interface Frame {
    readonly site: ExecutionPlanSiteDescriptor;
    readonly consumers: readonly ExecutionPlanSiteDescriptor[];
    next: number;
  }

  for (const root of sites) {
    if (state.has(root.functionId)) continue;
    const frames: Frame[] = [
      {
        site: root,
        consumers: consumersById.get(root.functionId) ?? [],
        next: 0,
      },
    ];
    state.set(root.functionId, "active");
    stack.push(root.functionId);

    while (frames.length > 0) {
      const frame = frames.at(-1);
      if (frame === undefined) break;
      if (frame.next >= frame.consumers.length) {
        frames.pop();
        stack.pop();
        state.set(frame.site.functionId, "complete");
        continue;
      }

      const consumer = frame.consumers[frame.next];
      frame.next += 1;
      if (consumer === undefined) continue;
      if (state.get(consumer.functionId) === "active") {
        const start = stack.indexOf(consumer.functionId);
        return [...stack.slice(start), consumer.functionId];
      }
      if (state.get(consumer.functionId) === "complete") continue;

      state.set(consumer.functionId, "active");
      stack.push(consumer.functionId);
      frames.push({
        site: consumer,
        consumers: consumersById.get(consumer.functionId) ?? [],
        next: 0,
      });
    }
  }
  return ["unknown"];
}

function dependencyKey(
  producerFunctionId: string,
  consumerFunctionId: string,
): string {
  return `${producerFunctionId}\u0000${consumerFunctionId}`;
}

function compareDependencies(
  left: ExecutionPlanDependency,
  right: ExecutionPlanDependency,
  siteRankById: ReadonlyMap<string, number>,
  inputRankBySiteId: ReadonlyMap<string, ReadonlyMap<string, number>>,
): number {
  const leftProducer = siteRankById.get(left.producerFunctionId);
  const rightProducer = siteRankById.get(right.producerFunctionId);
  const leftConsumer = siteRankById.get(left.consumerFunctionId);
  const rightConsumer = siteRankById.get(right.consumerFunctionId);
  if (
    leftProducer === undefined ||
    rightProducer === undefined ||
    leftConsumer === undefined ||
    rightConsumer === undefined
  ) {
    invalid("execution-plan dependency references an unknown site");
  }
  const leftInput = inputRankBySiteId
    .get(left.consumerFunctionId)
    ?.get(left.consumerInput);
  const rightInput = inputRankBySiteId
    .get(right.consumerFunctionId)
    ?.get(right.consumerInput);
  if (leftInput === undefined || rightInput === undefined) {
    invalid("execution-plan dependency references an unknown input");
  }
  return (
    leftConsumer - rightConsumer ||
    leftInput - rightInput ||
    leftProducer - rightProducer
  );
}

function compareSites(
  left: ExecutionPlanSiteDescriptor,
  right: ExecutionPlanSiteDescriptor,
): number {
  return (
    compareUtf8(left.normalizedSourcePath, right.normalizedSourcePath) ||
    left.start - right.start ||
    compareUtf8(left.functionId, right.functionId)
  );
}

function compareUtf8(left: string, right: string): number {
  const leftBytes = textEncoder.encode(left);
  const rightBytes = textEncoder.encode(right);
  const length = Math.min(leftBytes.length, rightBytes.length);
  for (let index = 0; index < length; index += 1) {
    const difference = (leftBytes[index] ?? 0) - (rightBytes[index] ?? 0);
    if (difference !== 0) return difference;
  }
  return leftBytes.length - rightBytes.length;
}

function invalid(message: string): never {
  throw new ExecutionPlanError("SEMA_EXECUTION_PLAN_INVALID", message);
}
