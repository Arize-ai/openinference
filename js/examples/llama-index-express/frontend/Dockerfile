FROM node:latest

WORKDIR /app

COPY . /app/

RUN npm install && npm run build

EXPOSE 3000

CMD ["npm", "run", "dev"]
