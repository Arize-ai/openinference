FROM node:latest

WORKDIR /app

COPY . /app/

RUN npm install && npm run build

EXPOSE 8000

CMD ["node", "--import", "./dist/instrumentation.cjs", "./dist/index.cjs"]
