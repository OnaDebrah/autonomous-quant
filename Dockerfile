FROM python:3.14-slim-trixie

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN useradd --uid 10001 --user-group --create-home aq \
    && install -d -o aq -g aq /data

WORKDIR /app
COPY pyproject.toml LICENSE ./
COPY src/ ./src/

ARG AQ_EXTRAS=""
RUN python -m pip install --no-cache-dir ".${AQ_EXTRAS}"

WORKDIR /data
USER aq
ENTRYPOINT ["aq"]
CMD ["--help"]
