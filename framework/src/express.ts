/**
 * Express and Nest integration: mount decorated controllers on any app or
 * router with `app.get(path, handler)`-style registration (Express 4 and 5,
 * Nest's Express adapter), and a middleware that opens one sema request scope
 * per request for handlers written outside the decorator style.
 */
import { withSemaScope } from "@semantscript/core";

import {
  handle,
  routesOf,
  type HttpMethod,
  type RequestContext,
} from "./controller.js";

/** The subset of an Express request the framework reads. */
export interface ExpressLikeRequest {
  readonly method: string;
  readonly path?: string;
  readonly url?: string;
  readonly params?: Readonly<Record<string, string | string[] | undefined>>;
  readonly query?: unknown;
  readonly headers: Readonly<Record<string, string | string[] | undefined>>;
  readonly body?: unknown;
}

export interface ExpressLikeResponse {
  status(code: number): this;
  set(name: string, value: string): this;
  json(body: unknown): unknown;
  on?(event: "finish" | "close", listener: () => void): unknown;
}

export type ExpressLikeNext = (error?: unknown) => void;

type Registrar = (
  path: string,
  handler: (
    request: ExpressLikeRequest,
    response: ExpressLikeResponse,
    next: ExpressLikeNext,
  ) => void,
) => unknown;

/** Anything with `get`, `post`, ... registration methods: an Express app or router. */
export type ExpressLikeApp = Record<Lowercase<HttpMethod>, Registrar>;

/** Registers every route of every controller on the app; returns the routes mounted. */
export function mountControllers(
  app: ExpressLikeApp,
  controllers: readonly object[],
): readonly string[] {
  const mounted: string[] = [];
  for (const controller of controllers) {
    for (const route of routesOf(controller)) {
      const register = app[route.method.toLowerCase() as Lowercase<HttpMethod>];
      register.call(app, route.path, (request, response, next) => {
        handle(route, expressContext(request, route.method, route.path)).then(
          (handled) => {
            for (const [name, value] of Object.entries(handled.headers))
              response.set(name, value);
            response.set(
              "x-sema-passes",
              `${String(handled.passes.encoder)}/${String(handled.passes.adapter)}/${String(handled.passes.head)}`,
            );
            response.status(handled.status).json(handled.body);
          },
          (error: unknown) => {
            next(error);
          },
        );
      });
      mounted.push(`${route.method} ${route.path}`);
    }
  }
  return mounted;
}

/**
 * Express or Nest middleware that runs the rest of the request inside one sema
 * request scope, so sema expressions in plain handlers share passes too. The
 * scope closes when the response finishes.
 */
export function semaRequestMiddleware(): (
  request: ExpressLikeRequest,
  response: ExpressLikeResponse,
  next: ExpressLikeNext,
) => void {
  return (_request, response, next) => {
    void withSemaScope(
      () =>
        new Promise<void>((resolve) => {
          if (typeof response.on === "function") {
            response.on("finish", resolve);
            response.on("close", resolve);
          } else {
            resolve();
          }
          next();
        }),
    );
  };
}

function expressContext(
  request: ExpressLikeRequest,
  method: HttpMethod,
  path: string,
): RequestContext {
  const params: Record<string, string> = {};
  for (const [name, value] of Object.entries(request.params ?? {})) {
    if (typeof value === "string") params[name] = value;
    else if (Array.isArray(value)) params[name] = value.join("/");
  }
  const query: Record<string, string> = {};
  if (request.query !== null && typeof request.query === "object") {
    for (const [name, value] of Object.entries(
      request.query as Record<string, unknown>,
    )) {
      if (typeof value === "string") query[name] = value;
      else if (Array.isArray(value)) query[name] = value.map(String).join(",");
    }
  }
  const headers: Record<string, string> = {};
  for (const [name, value] of Object.entries(request.headers)) {
    if (typeof value === "string") headers[name.toLowerCase()] = value;
    else if (Array.isArray(value))
      headers[name.toLowerCase()] = value.join(", ");
  }
  return {
    method,
    path: request.path ?? path,
    params,
    query,
    headers,
    body: request.body,
  };
}
