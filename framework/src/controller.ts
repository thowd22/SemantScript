/**
 * Controller-style handlers with sema expressions inside (TASK-8.1).
 *
 * A controller is a class whose methods are decorated with an HTTP method and
 * path. The framework owns nothing else: the handler receives a plain request
 * context, returns a value (sent as JSON) or a `reply`, and its sema
 * expressions run as written, in source order, inside one request scope so
 * sibling expressions over the same inputs share encoder and adapter passes
 * (see `withSemaScope` in the runtime).
 */
import {
  semaScopePasses,
  withSemaScope,
  type SemaStagePasses,
} from "@semantscript/core";

export type HttpMethod = "GET" | "POST" | "PUT" | "PATCH" | "DELETE";

export interface RequestContext {
  readonly method: HttpMethod;
  readonly path: string;
  readonly params: Readonly<Record<string, string>>;
  readonly query: Readonly<Record<string, string>>;
  readonly headers: Readonly<Record<string, string>>;
  readonly body: unknown;
}

export type HandlerResult = unknown;
/** A handler returns a value (JSON 200), a `Reply`, or a promise of either. */
export type Handler = (context: RequestContext) => unknown;

export interface Route {
  readonly method: HttpMethod;
  readonly path: string;
  readonly name: string;
  readonly handler: Handler;
}

/** A response with an explicit status; anything else a handler returns is JSON 200. */
export class Reply {
  constructor(
    readonly status: number,
    readonly body: unknown,
    readonly headers: Readonly<Record<string, string>> = {},
  ) {}
}

/** Return a `Reply` with an explicit status, body and headers from a handler. */
export function reply(
  status: number,
  body: unknown,
  headers?: Readonly<Record<string, string>>,
): Reply {
  return new Reply(status, body, headers);
}

/** A deterministic guard failed; mapped to its HTTP status (422 by default). */
export class RequirementError extends Error {
  readonly status: number;
  readonly code = "SEMA_REQUIREMENT_FAILED";

  constructor(message: string, status = 422) {
    super(message);
    this.name = "RequirementError";
    this.status = status;
  }
}

/**
 * A deterministic guard, written before or after a neural result and run
 * exactly there: sema calls are synchronous, so a guard above an expression
 * runs before the model is consulted and a guard below sees the model's value.
 */
export function require(
  condition: unknown,
  message: string,
  status = 422,
): asserts condition {
  if (!condition) {
    throw new RequirementError(message, status);
  }
}

interface RouteMetadata {
  readonly name: string;
  readonly method: HttpMethod;
  readonly path: string;
}

const routeMetadata = new WeakMap<object, readonly RouteMetadata[]>();
const controllerPrefix = new WeakMap<object, string>();
/** Method decorators run before their class decorator, which collects them. */
let pendingRoutes: RouteMetadata[] = [];

/**
 * Class decorator: every route the methods declared is mounted under `prefix`.
 * Standard (TC39) decorators: the method decorators of a class run first and
 * queue their routes; `@Controller` then binds the queue to the class, so a
 * class with routes must carry `@Controller`.
 */
export function Controller(prefix = "") {
  return function decorate(
    target: abstract new (...args: never[]) => object,
  ): void {
    controllerPrefix.set(target, normalizePrefix(prefix));
    routeMetadata.set(target, pendingRoutes);
    pendingRoutes = [];
  };
}

function method(verb: HttpMethod) {
  return function route(path = "") {
    return function decorate(
      _value: unknown,
      context: { readonly name: string | symbol },
    ): void {
      if (typeof context.name !== "string") {
        throw new TypeError("route handlers must be named methods");
      }
      pendingRoutes.push({
        name: context.name,
        method: verb,
        path: normalizePath(path),
      });
    };
  };
}

export const Get = method("GET");
export const Post = method("POST");
export const Put = method("PUT");
export const Patch = method("PATCH");
export const Delete = method("DELETE");

/** The routes of a controller instance, in declaration order, with bound handlers. */
export function routesOf(controller: object): readonly Route[] {
  const owner = controller.constructor;
  const prefix = controllerPrefix.get(owner) ?? "";
  const routes = routeMetadata.get(owner);
  if (routes === undefined || routes.length === 0) {
    throw new TypeError(
      `${owner.name || "controller"} declares no routes; decorate the class with @Controller and its methods with @Get, @Post, ...`,
    );
  }
  return routes.map((metadata) => {
    const member = (controller as Record<string, unknown>)[metadata.name];
    if (typeof member !== "function") {
      throw new TypeError(`route ${metadata.name} is not a method`);
    }
    const handler = (member as Handler).bind(controller);
    return {
      method: metadata.method,
      path: joinPaths(prefix, metadata.path),
      name: metadata.name,
      handler,
    };
  });
}

export interface HandledResponse {
  readonly status: number;
  readonly body: unknown;
  readonly headers: Readonly<Record<string, string>>;
  /** Model passes the request's sema expressions performed, after in-request sharing. */
  readonly passes: SemaStagePasses;
}

/**
 * Runs one handler inside a request scope and maps its outcome to a response:
 * a `Reply` as given, any other value as JSON 200, a `RequirementError` as its
 * status with `{ error, code }`, and anything else as 500.
 */
export async function handle(
  route: Route,
  context: RequestContext,
): Promise<HandledResponse> {
  return withSemaScope(async () => {
    const passes = (): SemaStagePasses =>
      semaScopePasses() ?? { encoder: 0, adapter: 0, head: 0 };
    try {
      const result = await route.handler(context);
      if (result instanceof Reply) {
        return {
          status: result.status,
          body: result.body,
          headers: result.headers,
          passes: passes(),
        };
      }
      return {
        status: 200,
        body: result ?? null,
        headers: {},
        passes: passes(),
      };
    } catch (error) {
      if (error instanceof RequirementError) {
        return {
          status: error.status,
          body: { error: error.message, code: error.code },
          headers: {},
          passes: passes(),
        };
      }
      return {
        status: 500,
        body: {
          error: error instanceof Error ? error.message : String(error),
          code: "SEMA_HANDLER_FAILED",
        },
        headers: {},
        passes: passes(),
      };
    }
  });
}

function normalizePrefix(prefix: string): string {
  const trimmed = prefix.trim().replace(/\/+$/u, "");
  if (trimmed === "") return "";
  return trimmed.startsWith("/") ? trimmed : `/${trimmed}`;
}

function normalizePath(path: string): string {
  const trimmed = path.trim();
  if (trimmed === "" || trimmed === "/") return "";
  return trimmed.startsWith("/")
    ? trimmed.replace(/\/+$/u, "")
    : `/${trimmed.replace(/\/+$/u, "")}`;
}

function joinPaths(prefix: string, path: string): string {
  const joined = `${prefix}${path}`;
  return joined === "" ? "/" : joined;
}
