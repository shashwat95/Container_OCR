FROM ultralytics/ultralytics:latest-jetson-jetpack4

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1 \
    TZ=Asia/Kolkata

RUN apt-get update --allow-insecure-repositories && apt-get install -y --no-install-recommends \
    python3-pip \
    libpq-dev \
    git \
    tzdata \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/

# Fix setuptools version first
RUN pip3 install --upgrade pip
RUN pip3 install setuptools==59.6.0

# Then install flasgger using a compatible version
RUN pip3 install flasgger==0.9.5 

# Install the rest of the requirements
RUN pip3 install --no-cache-dir -r requirements.txt

# #for easyocr
# RUN pip3 install python-bidi
# ENV PYTHONIOENCODING=utf-8

RUN pip3 install scikit-image
# RUN pip3 install pillow==9.0.1

#for torch2trt
# RUN git clone --recursive -b jax-jp4.6.1-trt7 https://github.com/akamboj2/torch2trt.git torch2trt && \
#     cd torch2trt && \
#     python3 setup.py install && \
#     cd ../ && \
#     rm -rf torch2trt


# COPY . /app/
# RUN cd /app/scene-text-recognition/EasyOCR && \
#     python3 setup.py install && \
#     cd .. && cd ..

RUN mkdir -p /app/evidence_images /app/logs && \
    chmod -R 777 /app

USER root
