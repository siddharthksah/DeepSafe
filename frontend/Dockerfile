# Use official Node.js image
FROM node:16-alpine AS build

# Set working directory
WORKDIR /app

# Copy package files
COPY package.json ./

# Install dependencies
RUN npm install

# Copy the rest of the application code
COPY . ./

# Fix permissions for node_modules/.bin
RUN chmod -R 755 node_modules/.bin

# Build the app
RUN npm run build

# Nginx for serving the built app
FROM nginx:alpine

# Copy build files
COPY --from=build /app/build /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port 80
EXPOSE 80

# Start Nginx server
CMD ["nginx", "-g", "daemon off;"]