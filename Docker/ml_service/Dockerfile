FROM python:3.9

#
WORKDIR /code

#
RUN pip install --upgrade pip
RUN pip install poetry

#
COPY ./pyproject.toml /code/
COPY ./poetry.lock /code/

COPY ./src /code/src
COPY ./Docker/ml_service/.env /code/.env


RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi


CMD ["uvicorn", "src.app.inference:app", "--host", "0.0.0.0", "--port", "80"]


