FROM debian:bookworm-slim
COPY --from=ghcr.io/astral-sh/uv:0.7.8 /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y bash \
    build-essential \
    procps \
    wget \
    curl \
    git \
    ca-certificates \
    less \
    vim \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN uv venv --python 3.12
RUN uv sync

RUN git submodule update --init --recursive

CMD ["/bin/bash"]