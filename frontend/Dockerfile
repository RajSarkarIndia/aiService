FROM node:22-alpine AS build

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

FROM node:22-alpine AS runtime

WORKDIR /app

RUN npm install --global http-server
COPY --from=build /app/dist/frontend/browser ./dist

EXPOSE 4200

CMD ["http-server", "dist", "--host", "0.0.0.0", "--port", "4000"]
