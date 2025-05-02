# Build stage
FROM node:16-alpine as build

# Set working directory
WORKDIR /app

# Copy package.json files
COPY package*.json ./

# Install dependencies with proper permissions
RUN npm install --legacy-peer-deps

# Copy all files
COPY . .

# Set proper permissions for node_modules
RUN chmod -R 755 node_modules/

# Build the application
RUN npm run build

# Production stage
FROM nginx:stable-alpine

# Copy build files from build stage
COPY --from=build /app/build /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port
EXPOSE 80

# Start Nginx server
CMD ["nginx", "-g", "daemon off;"]