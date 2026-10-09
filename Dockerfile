FROM python:3.12.15-bookworm@sha256:5560e9ab8709f459489e5b8aa696eda8a07ef821e14bb122be62d91234bfa98b

RUN apt-get update \
    && apt-get install -y --no-install-recommends g++ make gcc git build-essential ca-certificates curl \
    libc-dev libssl-dev libffi-dev zlib1g-dev python3-dev \
    && update-ca-certificates

ENV PDM_VENV_IN_PROJECT=1 \
    PDM_IGNORE_SAVED_PYTHON=1 \
    PDM_CHECK_UPDATE=false \
    PATH="/app/.venv/bin:${PATH}"

WORKDIR /app
COPY .scripts/pdm-bootstrap.txt /tmp/pdm-bootstrap.txt
RUN pip install --no-cache-dir --only-binary=:all: --require-hashes -r /tmp/pdm-bootstrap.txt

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
