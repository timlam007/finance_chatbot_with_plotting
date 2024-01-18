# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install any needed system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Adjust permissions to allow for non-root usage
RUN mkdir -p /app/data && \
    chown -R 1001:0 /app && \
    chmod -R g=u /app

# Copy just the requirements.txt initially and install Python dependencies
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code
COPY . .

# Change to a non-root user
USER 1001

# Expose the port the app runs on
EXPOSE 8501

# Define the command to run the app
# Uncomment the HEALTHCHECK line if you want to use it
# HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

