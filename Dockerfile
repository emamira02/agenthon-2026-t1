# Agenthon 2026 Track 1 agent image (linux/amd64)
FROM python:3.13-slim

LABEL qfbench2.interface_version="2.0"
LABEL org.opencontainers.image.source="https://github.com/emamira02/agenthon-2026-t1"
LABEL org.opencontainers.image.licenses="Apache-2.0"

RUN apt-get update && apt-get install -y --no-install-recommends build-essential wget ca-certificates \
    && cd /tmp \
    && wget -q https://github.com/ta-lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz \
    && tar xzf ta-lib-0.6.4-src.tar.gz && cd ta-lib-0.6.4 \
    && ./configure --prefix=/usr && make -j"$(nproc)" && make install \
    && cd / && rm -rf /tmp/ta-lib* \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /opt/agent/requirements.txt
RUN pip install --no-cache-dir -r /opt/agent/requirements.txt

COPY agent.py /opt/agent/agent.py
RUN mkdir -p /app/output /output \
    && ln -s /input/environment/data /app/data \
    && python -m py_compile /opt/agent/agent.py

ENV HOME=/tmp MPLCONFIGDIR=/tmp/mpl XDG_CACHE_HOME=/tmp/cache NUMBA_CACHE_DIR=/tmp/numba \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
WORKDIR /app
ENTRYPOINT ["python", "/opt/agent/agent.py"]
