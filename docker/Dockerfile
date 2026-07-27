# ---- build stage ----
FROM python:3.9-slim AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git wget \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . /src

# build wheels for your package + deps
RUN pip install --no-cache-dir --upgrade pip wheel \
 && pip wheel --no-cache-dir --wheel-dir /wheels .

RUN cd /opt ;git clone https://github.com/CellProfiling/densenet.git
RUN cd /opt/densenet; git checkout 19dad140b4ba4a279f05bd7f91084911da7ade04
RUN pip install -r /opt/densenet/predict/requirements.txt

RUN mkdir /opt/densenet/models
RUN wget https://github.com/CellProfiling/densenet/releases/download/v0.1.0/external_crop512_focal_slov_hardlog_class_densenet121_dropout_i768_aug2_5folds_fold0_final.pth -O /opt/densenet/models/model.pth

# ---- runtime stage ----
FROM python:3.9-slim

COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir /wheels/* \
 && rm -rf /wheels

COPY --from=build /opt/densenet /opt/densenet

# the package exposes a CLI users run as `cellmaps_image_embeddingcmd.py -h`
ENTRYPOINT ["cellmaps_image_embeddingcmd.py"]
CMD ["--help"]
