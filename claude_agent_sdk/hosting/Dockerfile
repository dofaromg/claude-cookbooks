# syntax=docker/dockerfile:1
#
# Build context is claude_agent_sdk/ (the parent of hosting/), because the
# agent imports from research_agent/ and utils/ which are siblings of hosting/.
#
#   cd claude_agent_sdk/
#   docker build -f hosting/Dockerfile -t research-agent .

FROM python:3.11-slim

# The Agent SDK shells out to the Claude Code CLI under the hood. Install Node
# and the CLI so `query()` has a runtime to drive.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates \
 && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
 && apt-get install -y --no-install-recommends nodejs \
 && npm install -g @anthropic-ai/claude-code@2.1.140 \
 && apt-get purge -y curl \
 && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/*

# Pin WORKDIR. Session transcripts are stored under
# $CLAUDE_CONFIG_DIR/projects/<encoded-cwd>/, so a stable cwd is what lets
# `resume=` find prior sessions across container restarts.
WORKDIR /app

COPY hosting/requirements.txt ./hosting/requirements.txt
RUN pip install --no-cache-dir -r hosting/requirements.txt

# The agent from notebook 00 and the shared utilities it imports.
COPY research_agent/ ./research_agent/
COPY utils/ ./utils/

# Hosting layer: server, ephemeral runner, entrypoint.
COPY hosting/server.py hosting/run_once.py hosting/entrypoint.sh ./hosting/
RUN chmod +x hosting/entrypoint.sh \
 && touch hosting/__init__.py

# Redirect session storage to /data. Each tier mounts something persistent
# there explicitly (compose bind mount, modal.Volume, k8s PVC) — no VOLUME
# instruction here because Modal's image builder rejects it, and an unmounted
# VOLUME would just leak anonymous volumes under plain `docker run` anyway.
ENV CLAUDE_CONFIG_DIR=/data

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

ENTRYPOINT ["./hosting/entrypoint.sh"]
