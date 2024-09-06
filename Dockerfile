FROM nvcr.io/nvidia/pytorch:24.07-py3

LABEL org.opencontainers.image.source=https://github.com/xiaozheyao/fmengine
LABEL org.opencontainers.image.description="FMEngine: Utilities for Foundation Models"
LABEL org.opencontainers.image.licenses=MIT

WORKDIR /fmengine

COPY . .
RUN pip install flash-attn --no-build-isolation
RUN pip install -r requirements-docker.txt
RUN pip install -e .