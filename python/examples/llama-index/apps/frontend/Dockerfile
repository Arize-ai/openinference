FROM node:20-alpine as build

WORKDIR /app

# Install dependencies
COPY package.json package-lock.* ./
RUN npm install

ENV NEXT_PUBLIC_CHAT_API http://localhost:8000/api/chat

# Build the application
COPY . .
RUN npm run build

# ====================================
FROM build as release

CMD ["npm", "run", "start"]