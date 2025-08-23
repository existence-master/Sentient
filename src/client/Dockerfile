# src/client/Dockerfile

# 1. Builder Stage: Installs dependencies and builds the app
FROM node:20-slim AS builder
WORKDIR /app

# Set the environment to production for the build stage
ENV NODE_ENV=production

# ---- MODIFIED SECTION ----
# Declare all build-time arguments that your application needs.
ARG NEXT_PUBLIC_APP_SERVER_URL
ARG AUTH0_SECRET
ARG INTERNAL_APP_SERVER_URL
ARG NEXT_PUBLIC_ENVIRONMENT
ARG SELF_HOST_AUTH_TOKEN
ARG APP_BASE_URL
ARG NEXT_PUBLIC_APP_BASE_URL
ARG AUTH0_ISSUER_BASE_URL
ARG AUTH0_DOMAIN
ARG AUTH0_CLIENT_ID
ARG AUTH0_CLIENT_SECRET
ARG AUTH0_AUDIENCE
ARG AUTH0_SCOPE
ARG MONGO_URI
ARG MONGO_DB_NAME
ARG VAPID_PRIVATE_KEY
ARG VAPID_ADMIN_EMAIL
ARG NEXT_PUBLIC_POSTHOG_KEY
ARG NEXT_PUBLIC_POSTHOG_HOST
ARG NEXT_PUBLIC_AUTH0_NAMESPACE
ARG NEXT_PUBLIC_LANDING_PAGE_URL
ARG NEXT_PUBLIC_VAPID_PUBLIC_KEY

# Set them as environment variables for the build process
ENV NEXT_PUBLIC_APP_SERVER_URL=$NEXT_PUBLIC_APP_SERVER_URL
ENV AUTH0_SECRET=$AUTH0_SECRET
ENV INTERNAL_APP_SERVER_URL=$INTERNAL_APP_SERVER_URL
ENV NEXT_PUBLIC_ENVIRONMENT=$NEXT_PUBLIC_ENVIRONMENT
ENV SELF_HOST_AUTH_TOKEN=$SELF_HOST_AUTH_TOKEN
ENV APP_BASE_URL=$APP_BASE_URL
ENV NEXT_PUBLIC_APP_BASE_URL=$NEXT_PUBLIC_APP_BASE_URL
ENV AUTH0_ISSUER_BASE_URL=$AUTH0_ISSUER_BASE_URL
ENV AUTH0_DOMAIN=$AUTH0_DOMAIN
ENV AUTH0_CLIENT_ID=$AUTH0_CLIENT_ID
ENV AUTH0_CLIENT_SECRET=$AUTH0_CLIENT_SECRET
ENV AUTH0_AUDIENCE=$AUTH0_AUDIENCE
ENV AUTH0_SCOPE=$AUTH0_SCOPE
ENV VAPID_PRIVATE_KEY=$VAPID_PRIVATE_KEY
ENV VAPID_ADMIN_EMAIL=$VAPID_ADMIN_EMAIL
ENV MONGO_URI=$MONGO_URI
ENV MONGO_DB_NAME=$MONGO_DB_NAME
ENV NEXT_PUBLIC_POSTHOG_KEY=$NEXT_PUBLIC_POSTHOG_KEY
ENV NEXT_PUBLIC_POSTHOG_HOST=$NEXT_PUBLIC_POSTHOG_HOST
ENV NEXT_PUBLIC_AUTH0_NAMESPACE=$NEXT_PUBLIC_AUTH0_NAMESPACE
ENV NEXT_PUBLIC_LANDING_PAGE_URL=$NEXT_PUBLIC_LANDING_PAGE_URL
ENV NEXT_PUBLIC_VAPID_PUBLIC_KEY=$NEXT_PUBLIC_VAPID_PUBLIC_KEY
# --------------------------

# Copy package.json and lock file to leverage Docker cache
COPY package*.json ./
RUN npm ci

# Copy the rest of the application source code
COPY . .

# Build the Next.js application for production
RUN npm run build


# 2. Runner Stage: Creates the final, lightweight image
FROM node:20-alpine AS runner
WORKDIR /app

# Set the environment to production
ENV NODE_ENV=production
ENV PORT=3000

# Copy the standalone Next.js server from the builder stage
COPY --from=builder /app/.next/standalone ./
# Copy public assets and static build artifacts
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/static ./.next/static

# Expose the port the app will run on
EXPOSE 3000

# The runner stage will get its env vars from the docker-compose 'env_file'
# The command to start the optimized Next.js server
CMD ["node", "server.js"]