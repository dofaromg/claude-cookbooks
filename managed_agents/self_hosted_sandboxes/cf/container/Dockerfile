FROM debian:12-slim
ARG ANT_VERSION=1.9.0
ARG TARGETOS=linux
ARG TARGETARCH=amd64
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*
ADD https://github.com/anthropics/anthropic-cli/releases/download/v${ANT_VERSION}/ant_${ANT_VERSION}_${TARGETOS}_${TARGETARCH}.tar.gz /tmp/ant.tgz
RUN tar -xzf /tmp/ant.tgz -C /usr/local/bin ant \
 && chmod +x /usr/local/bin/ant && rm /tmp/ant.tgz
WORKDIR /workspace
# --max-idle: exit 60s after session.status_idle/end_turn, reset on any event.
ENTRYPOINT ["ant", "beta:worker", "run", "--workdir", "/workspace", "--unrestricted-paths", "--max-idle", "60s", "--log-format", "json"]
