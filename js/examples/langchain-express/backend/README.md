This is a [Arize](https://www.arize.com/) project using [Express](https://expressjs.com/)

## Getting Started

First, install the dependencies:

```
npm install
```

Second, run the development server:

```
npm run dev
```

Then call the express API endpoint `/api/chat` to see the result:

```
curl --location 'localhost:8000/api/chat' \
--header 'Content-Type: text/plain' \
--data '{ "messages": [{ "role": "user", "content": "Hello" }] }'
```

You can start editing the API by modifying `src/controllers/chat.controller.ts`. The endpoint auto-updates as you save the file.

## Production

First, build the project:

```
npm run build
```

You can then run the production server:

```
NODE_ENV=production npm run start
```

> Note that the `NODE_ENV` environment variable is set to `production`. This disables CORS for all origins.
