FROM python:3.10.13-bullseye

RUN apt-get update \
    && apt-get install -y --no-install-recommends g++ make gcc git build-essential ca-certificates curl \
    libc-dev libssl-dev libffi-dev zlib1g-dev \
    && update-ca-certificates

WORKDIR /app
ADD ./requirements.txt .
RUN pip install -r requirements.txt

ADD . .
RUN rm -rf /app/ramjet/settings/prd.*

RUN python setup.py install

RUN adduser --disabled-password --gecos '' laisky \
    && chown -R laisky:laisky /app
USER laisky

CMD [ "python" , "-m" , "ramjet" ]
