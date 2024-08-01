FROM python:3.10-slim

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app

# Copy the static html files into a directory that matches the structure of the current project

CMD ["fastapi", "run", "app/main.py", "--port", "8000"]
