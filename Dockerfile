FROM python:3.9.16-bullseye

RUN apt-get update \
    && apt-get install -y --no-install-recommends g++ make gcc git build-essential ca-certificates curl \
    libxslt-dev libxml2-dev libc-dev libssl-dev libffi-dev zlib1g-dev libopenblas-base libomp-dev \
    libmagic-dev poppler-utils tesseract-ocr libreoffice \
    && update-ca-certificates

WORKDIR /app
RUN pip install "torch==2.0.0"
ADD ./requirements.txt .
RUN pip install -r requirements.txt


ADD . .
RUN rm -rf /app/ramjet/settings/prd.*

RUN python setup.py install

RUN adduser --disabled-password --gecos '' laisky \
    && chown -R laisky:laisky /app
USER laisky

CMD [ "python" , "-m" , "ramjet" ]
