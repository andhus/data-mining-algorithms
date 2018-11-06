FROM wattyinc/python-timeseries:0.4

USER root

RUN ln -sf /usr/share/zoneinfo/Etc/UTC /etc/localtime

RUN useradd --create-home --home-dir /opt/service --shell /bin/bash service
WORKDIR /opt/service/app

COPY apt-requirements.txt .
RUN apt-get update && \
    apt-get install --assume-yes python-pip && \
    pip install --upgrade pip && \
    apt-get install -y $(cat apt-requirements.txt)

COPY requirements.txt .

RUN pip install \
    --no-cache-dir \
    --trusted-host pypi.watty.io \
    -r requirements.txt

COPY recurrent_patterns_predictor recurrent_patterns_predictor

COPY run_tests.sh \
     setup.py \
     ./

RUN pip install --no-deps -e .
RUN chown -R service:service /opt/service

USER service

CMD [ "recurrent-patterns-predictor" ]



+++++++++++



ARG UBUNTU_VERSION=16.04
FROM ubuntu:${UBUNTU_VERSION}

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        libcurl3-dev \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python-dev \
        rsync \
        software-properties-common \
        unzip \
        zip \
        zlib1g-dev \
        openjdk-8-jdk \
        openjdk-8-jre-headless \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ARG USE_PYTHON_3_NOT_2=True
ARG _PY_SUFFIX=${USE_PYTHON_3_NOT_2:+3}
ARG PYTHON=python${_PY_SUFFIX}
ARG PIP=pip${_PY_SUFFIX}

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

RUN apt-get update && apt-get install -y \
    ${PYTHON} \
    ${PYTHON}-pip

RUN ${PIP} install --upgrade \
    pip \
    setuptools

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    openjdk-8-jdk \
    ${PYTHON}-dev \
    swig

# Install bazel
RUN echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list && \
    curl https://bazel.build/bazel-release.pub.gpg | apt-key add - && \
    apt-get update && \
    apt-get install -y bazel

COPY bashrc /etc/bash.bashrc
RUN chmod a+rwx /etc/bash.bashrc




RUN mkdir datasets
COPY datasets/download_news_groups.sh datasets/.
RUN ./datasets/download_news_groups.sh
