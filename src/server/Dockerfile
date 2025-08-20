# Use a slim Python base image
FROM python:3.11-slim

# Set working directory early
WORKDIR /app

# Install system dependencies first, as they change infrequently
RUN apt-get update && apt-get install -y --no-install-recommends \
    supervisor \
    nginx \
    curl \
    dos2unix \
    ca-certificates \
    gnupg \
    python3-pip \
    # Add textract dependencies
    antiword \
    poppler-utils \
    tesseract-ocr \
    unrtf \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies to a separate layer
# This layer will be cached unless requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# --- Build Arguments ---
# These arguments are passed from docker-compose.yaml or the `docker build` command.
ARG OPENAI_API_KEY
ARG OPENAI_MODEL_NAME

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    DEBIAN_FRONTEND=noninteractive
# Set Ollama to listen on all interfaces within the container
ENV OLLAMA_HOST=0.0.0.0

# --- Conditional Ollama Installation ---
# This checks if the API key is set to 'ollama'. If so, install Ollama and pull the specified model.
# This is checked during the 'docker build' process.
RUN if [ "$OPENAI_API_KEY" = "ollama" ]; then \
    echo "Ollama API key detected. Installing Ollama..."; \
    curl -fsSL https://ollama.com/install.sh | sh; \
    echo "Pulling model: ${OPENAI_MODEL_NAME}"; \
    /usr/local/bin/ollama pull ${OPENAI_MODEL_NAME}; \
    else \
    echo "Ollama configuration not detected. Skipping installation."; \
    fi

# Create required directories
RUN mkdir -p /etc/supervisor/conf.d /var/log/supervisor

# Copy configuration files and server code
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY nginx.conf /etc/nginx/sites-available/sentient
COPY . .

# Fix line endings for .env and start.sh (if present)
RUN if [ -f /app/.env ]; then dos2unix /app/.env; fi \
    && dos2unix /app/start.sh \
    && chmod +x /app/start.sh

# Remove default Nginx site configuration and enable our custom one
RUN rm -f /etc/nginx/sites-enabled/default

# Enable custom Nginx site
RUN ln -s /etc/nginx/sites-available/sentient /etc/nginx/sites-enabled/sentient

# Expose port
EXPOSE 5000

# Start our processes using supervisord
CMD ["/app/start.sh"]