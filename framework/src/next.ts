/**
 * Next.js App Router integration: the exported route handlers of a
 * `route.ts` file from a decorated controller, over the web `Request` and
 * `Response` types, each request in its own sema scope.
 */
import {
  handle,
  routesOf,
  type HttpMethod,
  type RequestContext,
  type Route,
} from "./controller.js";

export type NextRouteHandler = (
  request: Request,
  context?: {
    readonly params?:
      | Record<string, string | string[]>
      | Promise<Record<string, string | string[]>>;
  },
) => Promise<Response>;

/**
 * One handler per HTTP method the controller declares: `export const { GET, POST } = nextRouteHandlers(new TicketController())`.
 * A controller with several routes for one method dispatches on the request path's tail.
 */
export function nextRouteHandlers(
  controller: object,
): Partial<Record<HttpMethod, NextRouteHandler>> {
  const routes = routesOf(controller);
  const byMethod = new Map<HttpMethod, Route[]>();
  for (const route of routes) {
    byMethod.set(route.method, [...(byMethod.get(route.method) ?? []), route]);
  }
  const handlers: Partial<Record<HttpMethod, NextRouteHandler>> = {};
  for (const [method, candidates] of byMethod) {
    handlers[method] = async (request, context) => {
      const url = new URL(request.url);
      const route = selectRoute(candidates, url.pathname);
      const params = await resolveParams(context?.params);
      const handled = await handle(
        route,
        await nextContext(request, method, url, params),
      );
      return Response.json(handled.body, {
        status: handled.status,
        headers: {
          ...handled.headers,
          "x-sema-passes": `${String(handled.passes.encoder)}/${String(handled.passes.adapter)}/${String(handled.passes.head)}`,
        },
      });
    };
  }
  return handlers;
}

function selectRoute(candidates: readonly Route[], pathname: string): Route {
  const first = candidates[0];
  if (first === undefined) throw new TypeError("no route");
  if (candidates.length === 1) return first;
  const match = candidates.find(
    (route) => route.path !== "/" && pathname.endsWith(route.path),
  );
  return match ?? first;
}

async function resolveParams(
  params:
    | Record<string, string | string[]>
    | Promise<Record<string, string | string[]>>
    | undefined,
): Promise<Record<string, string>> {
  const resolved = params === undefined ? {} : await params;
  return Object.fromEntries(
    Object.entries(resolved).map(([name, value]) => [
      name,
      Array.isArray(value) ? value.join("/") : value,
    ]),
  );
}

async function nextContext(
  request: Request,
  method: HttpMethod,
  url: URL,
  params: Record<string, string>,
): Promise<RequestContext> {
  const headers: Record<string, string> = {};
  request.headers.forEach((value, name) => {
    headers[name.toLowerCase()] = value;
  });
  let body: unknown = undefined;
  if (method !== "GET" && method !== "DELETE") {
    const text = await request.text();
    if (text.length > 0) {
      try {
        body = JSON.parse(text);
      } catch {
        body = text;
      }
    }
  }
  return {
    method,
    path: url.pathname,
    params,
    query: Object.fromEntries(url.searchParams.entries()),
    headers,
    body,
  };
}
