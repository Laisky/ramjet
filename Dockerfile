FROM python:3.8.7-buster

RUN apt-get update \
    && apt-get install -y --no-install-recommends g++ make gcc git build-essential ca-certificates curl \
    libxslt-dev libxml2-dev libc-dev libssl-dev libffi-dev zlib1g-dev \
    && update-ca-certificates

WORKDIR /app
ADD ./requirements.txt .
RUN pip install -r requirements.txt

RUN rm -rf /app/ramjet/settings/prd.*

ADD . .
RUN python setup.py install

ENTRYPOINT python -m ramjet
