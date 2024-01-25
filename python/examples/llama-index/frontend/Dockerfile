# Build Stage
FROM node:20 AS build

# Set the working directory
WORKDIR /app

# Copy package.json and package-lock.json
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of the code
COPY . .

# Build the application
RUN npm run build 

# Production Stage
FROM node:20-slim

# Set the working directory
WORKDIR /app

# Copy the rest of your application's code
COPY . /app/

# Install production dependencies
COPY package*.json ./
RUN npm install --only=production

# The command to run when the container starts
CMD ["npm", "start"]
