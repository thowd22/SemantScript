# syntax=docker/dockerfile:1
# Runs the ticket API with its trained artifact. Build from the repository root
# so the linked compiler and runtime packages are in context:
#
#   docker build -f examples/express-app/deploy/Dockerfile -t ticket-api .
#   docker run -p 3000:3000 ticket-api
#
# Stage 1 compiles with tspc (the ts-patch transformer). Stage 2 keeps only the
# compiled app, its production dependencies and the artifact. `npm ci --omit=dev`
# installs @semantscript/core, whose onnxruntime-node and tokenizers dependencies
# ship prebuilt binaries for linux/x64 and linux/arm64; nothing is compiled or
# downloaded by hand.
FROM node:22-bookworm-slim AS build
WORKDIR /repo
COPY compiler ./compiler
COPY runtime ./runtime
COPY examples/express-app ./examples/express-app
WORKDIR /repo/examples/express-app
RUN npm ci && npm run build

FROM node:22-bookworm-slim AS runtime
ENV NODE_ENV=production
WORKDIR /app
COPY --from=build /repo/compiler /repo/compiler
COPY --from=build /repo/runtime /repo/runtime
COPY --from=build /repo/examples/express-app/package.json /repo/examples/express-app/package-lock.json ./
RUN npm ci --omit=dev
COPY --from=build /repo/examples/express-app/dist ./dist
# The artifact published by `semantscript train` (default .semantscript/artifact).
# The runtime finds it by searching upward from dist/server.js.
COPY examples/express-app/.semantscript/artifact ./.semantscript/artifact
EXPOSE 3000
CMD ["node", "--enable-source-maps", "dist/server.js"]
