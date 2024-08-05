FROM node:alpine AS build_frontend
COPY ./frontend/package.json ./frontend/package-lock.json ./
RUN npm install
COPY ./frontend .
ENV NEXTPROD NO
RUN npm run build

FROM python:3.10-slim AS build_backend
WORKDIR /quickdraw
COPY ./backend/requirements.txt .
RUN pip install --no-cache-dir --upgrade -r ./requirements.txt
COPY ./backend/app ./app
COPY --from=build_frontend /out ./out
CMD ["fastapi", "run", "app/main.py", "--port", "3000"]
