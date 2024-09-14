FROM node:alpine AS build_frontend
COPY ./frontend/package.json ./frontend/package-lock.json ./
RUN npm install
COPY ./frontend .
ENV NEXTPROD NO
RUN npm run build

FROM nginx:alpine AS nginx
COPY ./deploy/nginx/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build_frontend /out ./out
CMD ["nginx", "-g", "daemon off;"]
