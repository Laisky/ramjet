FROM python:3.11.7-bullseye

RUN apt-get update \
    && apt-get install -y --no-install-recommends g++ make gcc git build-essential ca-certificates curl \
    libc-dev libssl-dev libffi-dev zlib1g-dev python3-dev \
    && update-ca-certificates

WORKDIR /app
ADD ./requirements.txt .
ADD ./constraints.txt .
RUN PIP_CONSTRAINT=constraints.txt pip install -r requirements.txt

ADD . .
RUN rm -rf /app/ramjet/settings/prd.*

RUN python setup.py install

RUN adduser --disabled-password --gecos '' laisky \
    && chown -R laisky:laisky /app
USER laisky

CMD [ "python" , "-m" , "ramjet" ]
