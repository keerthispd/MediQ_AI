# Build from the repository root: docker build -f frontend/Dockerfile .
FROM node:22-alpine AS build
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --silent
COPY frontend/ ./
RUN npm run build

# Nginx serves the built app and proxies /api to the backend container
FROM nginx:stable-alpine
COPY deploy/nginx/app.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/build /usr/share/nginx/html
EXPOSE 80
