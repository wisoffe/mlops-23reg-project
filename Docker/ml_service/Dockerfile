FROM python:3.9

WORKDIR /code

COPY ./pyproject.toml /code/
COPY ./poetry.lock /code/
COPY ./src /code/src
COPY ./Docker/ml_service/.env /code/.env

RUN pip install --upgrade pip
RUN pip install poetry

# Variant 1. Installing packages via poetry:
# RUN poetry config virtualenvs.create false \
#     && poetry install --no-dev --no-interaction --no-ansi

# Variant 2. Installing packages via pip (get deps from poetry):
RUN poetry export --without-hashes -f requirements.txt --output requirements.txt
RUN pip install -r requirements.txt

CMD ["uvicorn", "src.app.inference:app", "--host", "0.0.0.0", "--port", "80"]

