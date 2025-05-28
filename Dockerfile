FROM python:3.11-buster

RUN pip install poetry

COPY . .

RUN apt update && apt install ffmpeg -y

RUN poetry install

ENTRYPOINT ["poetry", "run", "fastapi", "run", "moody_backend/"]