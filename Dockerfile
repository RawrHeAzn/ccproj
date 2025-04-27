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

# Install prerequisites including tools needed for adding repo and unixodbc
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        gnupg \
        unixodbc \
        unixodbc-dev \
        unixodbc-bin \
        # lsb-release is needed if dynamically detecting debian version, but we hardcode 12
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Add Microsoft GPG key and repository for Debian 12 (Bookworm)
RUN curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg && \
    curl -fsSL https://packages.microsoft.com/config/debian/12/prod.list > /etc/apt/sources.list.d/mssql-release.list

# Install Microsoft ODBC Driver 18
RUN apt-get update && \
    ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql18 && \
    # Configure odbcinst.ini for Driver 18
    printf "[ODBC Driver 18 for SQL Server]\nDescription=Microsoft ODBC Driver 18 for SQL Server\nDriver=/opt/microsoft/msodbcsql18/lib64/libmsodbcsql-18.1.so.1.1\nUsageCount=1\n" > /etc/odbcinst.ini && \
    # Verify the driver path exists (optional sanity check)
    # ls -l /opt/microsoft/msodbcsql18/lib64/libmsodbcsql-18.1.so.1.1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy just the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port the app runs on
EXPOSE 10000

# Command to run the application
CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=10000"] 