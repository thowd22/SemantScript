---
id: TASK-8.1
title: Controller abstraction and HTTP runtime
status: Done
assignee:
  - '@claude'
created_date: '2026-09-19 18:23'
updated_date: '2026-09-25 01:39'
labels:
  - runtime
milestone: m-4
dependencies:
  - TASK-7.1
parent_task_id: TASK-8
ordinal: 29000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Transcript turn 9 point 9: controller-style handlers with sema expressions inside. Implement as thin integrations (decorators/middleware) for Express, Nest and Next route handlers that batch sema evaluation per request via the execution plan, rather than a standalone HTTP server.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Decorator-based controllers route HTTP requests to handlers
- [x] #2 sema expressions inside a handler are executed via the execution plan
- [x] #3 Deterministic require() guards run before and after neural results as written
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Runtime: request scopes (withSemaScope, AsyncLocalStorage) that the worker honours: within a scope, sentence and function embeddings are cached per (encoder, input) and (adapter, input) so sibling sema sites over the same inputs share passes, dropped when the scope ends; invoke responses report passes and the scope accumulates them (semaScopePasses).
2. Framework package @semantscript/framework (framework/): standard decorators @Controller, @Get, @Post, @Put, @Patch, @Delete with routesOf(); a RequestContext (params, query, body, headers) and reply(); require(condition, message, status) throwing RequirementError mapped to 4xx; mountControllers(app, controllers) for Express-style apps, nextRouteHandlers(controller) for Next App Router route files, semaRequestMiddleware() for Express and Nest; every handler runs inside a request scope so its sema sites are evaluated per the execution plan's fusion within the request.
3. Tests: runtime scope sharing (two functions over one input in one scope: one encoder pass; two scopes: two), Express server round trip through decorators with passes asserted, Next handler invocation, guard ordering (a failing pre-guard prevents the neural call; a post-guard sees the neural value; source order preserved), error mapping.
4. Docs: framework README, runtime README scope section, docs index, CONTRIBUTING layout; wire the workspace into build, lint and test gates.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Runtime: request scopes (withSemaScope over AsyncLocalStorage; invoke messages carry a scope id; the worker keeps sentence and function embeddings per scope, at most 64 per kind, and drops them on end-scope; the control buffer reports the passes of each call; semaScopePasses accumulates them). Framework package framework/ (@semantscript/framework): @Controller and @Get/@Post/@Put/@Patch/@Delete standard decorators (methods queue routes, the class decorator binds them), routesOf, handle (one scope per handler; Reply, JSON 200, RequirementError to its status, 500 otherwise), require guards, mountControllers for Express-style apps, semaRequestMiddleware for Express and Nest, nextRouteHandlers for the App Router. Tests: runtime request-scope test (two sibling calls over one input in a scope: 1 encoder, 1 adapter, 3 heads; distinct inputs, separate scopes, async spans, error release); framework tests (routes, guard ordering with a trace: pre-guard, neural, neural, post-guard, and a failing pre-guard performing zero passes; Express round trip over HTTP with x-sema-passes 1/1/2; middleware; Next handlers). Workspace wired into build, lint and test gates.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added @semantscript/framework and runtime request scopes. Decorated controllers (@Controller, @Get, @Post, ...) mount on Express or Nest apps and become Next.js App Router handlers; every handler runs inside one sema request scope, so the sema expressions it evaluates share encoder and adapter passes over identical inputs as an execution-plan stage would, with the passes reported per response; require() guards are plain synchronous assertions that run before and after neural results exactly as written, mapped to HTTP statuses. Verified by the runtime scope test and five framework tests including an HTTP round trip and a guard-ordering trace; 291 Node tests pass, lint clean.
<!-- SECTION:FINAL_SUMMARY:END -->
