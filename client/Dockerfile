# Switch to alpine in the future
FROM node

WORKDIR /app

# Copying and installing this first is better
COPY app/package.json app/package-lock.json ./

# Install dependencies
RUN npm install

# Copy the current directory contents into the container at /app
COPY . .

# Build the app
RUN npm run build

EXPOSE 3000

# Serve the app
CMD ["npm", "start"]
