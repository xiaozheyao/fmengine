FROM nvcr.io/nvidia/pytorch:24.07-py3

WORKDIR /fmengine

COPY . .
RUN pip install flash-attn --no-build-isolation
RUN pip install -r requirements-docker.txt