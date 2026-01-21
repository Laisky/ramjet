FROM python:3.12.6-bullseye

RUN apt-get update \
    && apt-get install -y --no-install-recommends g++ make gcc git build-essential ca-certificates curl \
    libc-dev libssl-dev libffi-dev zlib1g-dev python3-dev \
    && update-ca-certificates

ENV PDM_VENV_IN_PROJECT=1 \
    PDM_IGNORE_SAVED_PYTHON=1 \
    PDM_CHECK_UPDATE=false \
    PATH="/app/.venv/bin:${PATH}"

WORKDIR /app
RUN pip install --no-cache-dir pdm

COPY pyproject.toml pdm.lock LICENSE ./
RUN pdm install --prod --frozen-lockfile --no-editable --no-self \
    && rm -rf /root/.cache

COPY . .
RUN pdm install --prod --frozen-lockfile --no-editable
RUN rm -rf /app/ramjet/settings/prd.*

RUN adduser --disabled-password --gecos '' laisky \
    && chown -R laisky:laisky /app
USER laisky

CMD [ "python" , "-m" , "ramjet" ]
