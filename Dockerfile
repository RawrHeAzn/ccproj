# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
# Prevents Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED=1
# Set noninteractive frontend for apt-get to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive
# Tell ODBC where to find system-wide config
ENV ODBCSYSINI=/etc
# Update path for ODBC Driver 18
ENV LD_LIBRARY_PATH=/opt/microsoft/msodbcsql18/lib64:${LD_LIBRARY_PATH}

# Install prerequisites and tools for adding repo
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        gnupg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Add Microsoft GPG key and repository for Debian 12 (Bookworm)
RUN curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg && \
    curl -fsSL https://packages.microsoft.com/config/debian/12/prod.list > /etc/apt/sources.list.d/mssql-release.list

# Install unixodbc, ODBC Driver 18, and configure
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        unixodbc \
        unixodbc-dev \
        unixodbc-bin \
    # Install the driver AFTER apt lists are updated
    && ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql18 && \
    # Configure odbcinst.ini using the base symlink name
    printf "[ODBC Driver 18 for SQL Server]\nDescription=Microsoft ODBC Driver 18 for SQL Server\nDriver=/opt/microsoft/msodbcsql18/lib64/libmsodbcsql-18.so\nUsageCount=1\n" > /etc/odbcinst.ini && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose the port the app runs on
EXPOSE 10000

# Command to run the application
CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=10000"] 