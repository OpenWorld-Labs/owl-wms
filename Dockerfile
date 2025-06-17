FROM nvcr.io/nvidia/physicsnemo/physicsnemo:25.06

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

RUN python -m pip install -e .

RUN git submodule update --init --recursive

CMD ["/bin/bash"]