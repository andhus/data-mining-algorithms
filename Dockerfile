FROM alpine

RUN apk add --no-cache libpng freetype libstdc++ python py-pip bash
RUN apk add --no-cache --virtual .build-deps \
	    gcc \
	    build-base \
	    python-dev \
	    libpng-dev \
	    musl-dev \
	    freetype-dev
RUN ln -s /usr/include/locale.h /usr/include/xlocale.h

RUN mkdir /home/data-mining-algorithms
WORKDIR /home/data-mining-algorithms

COPY scripts ./scripts
RUN mkdir datasets

RUN ./scripts/download_news_groups.sh -t ./datasets

RUN pip install pip --upgrade
COPY python/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY python ./python
RUN pip install -e python/. --no-deps
RUN nosetests -sv python/data_mining/
