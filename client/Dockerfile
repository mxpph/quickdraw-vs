# Use an official Node runtime as a parent image
FROM node

WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Install dependencies
RUN npm install

# Build the app
RUN npm run build

EXPOSE 3000

# Serve the app
CMD ["npm", "start"]
