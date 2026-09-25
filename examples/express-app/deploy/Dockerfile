# syntax=docker/dockerfile:1
# Runs the ticket API with its artifact. Build from the repository root so the
# linked compiler, runtime and framework packages are in the build context:
#
#   docker build -f examples/express-app/deploy/Dockerfile -t ticket-api .
#   docker run -p 3000:3000 ticket-api
#
# The build context is limited by Dockerfile.dockerignore beside this file
# (an allowlist: the root workspace manifests, the three linked packages and
# this example), so it works from a fresh clone and ignores local installs.
# The artifact is whatever sits in examples/express-app/.semantscript/artifact:
# the release `semantscript train` published, or `npm run fixture-artifact`'s
# fixture when only the wiring is being checked (CI builds with the fixture).
#
# Stage 1 installs the workspace, builds compiler, runtime and framework with
# tsc -b, then compiles the example with tspc (the ts-patch transformer). It
# then installs the production tree a second time with --install-links, which
# packs the file: packages into real copies with their own dependencies, so the
# image needs no repository layout and no symlinks. onnxruntime-node and
# tokenizers ship prebuilt binaries for linux/x64 and linux/arm64; nothing is
# compiled, and the optional CUDA download is skipped.
#
# Stage 2 keeps the compiled app, the production node_modules and the artifact.
# The IR bundle (which holds the prompt text) is not copied: the runtime needs
# only the artifact.
FROM node:22-bookworm-slim AS build
ENV ONNXRUNTIME_NODE_INSTALL=skip \
    npm_config_fund=false \
    npm_config_audit=false \
    npm_config_update_notifier=false
WORKDIR /repo
COPY package.json package-lock.json tsconfig.base.json tsconfig.json ./
COPY cli/package.json ./cli/package.json
COPY benchmarks/refund/package.json ./benchmarks/refund/package.json
COPY compiler ./compiler
COPY runtime ./runtime
COPY framework ./framework
RUN npm ci && npx tsc -b compiler runtime framework
COPY examples/express-app ./examples/express-app
WORKDIR /repo/examples/express-app
RUN npm install && npm run build && rm dist/semantscript.ir.v1.json
WORKDIR /repo/examples/production
RUN cp ../express-app/package.json . && npm install --omit=dev --install-links

FROM node:22-bookworm-slim AS runtime
ENV NODE_ENV=production
WORKDIR /app
COPY --from=build /repo/examples/express-app/package.json ./
COPY --from=build /repo/examples/production/node_modules ./node_modules
COPY --from=build /repo/examples/express-app/dist ./dist
# The runtime finds the artifact by searching upward from dist/server.js.
COPY examples/express-app/.semantscript/artifact ./.semantscript/artifact
USER node
EXPOSE 3000
CMD ["node", "--enable-source-maps", "dist/server.js"]
