FROM python:3.10-slim
WORKDIR /quickdraw
COPY ./backend/requirements.txt .
RUN pip install --no-cache-dir --upgrade -r ./requirements.txt
COPY ./backend/app ./app
CMD ["fastapi", "run", "app/main.py", "--workers", "2", "--port", "3000"]
