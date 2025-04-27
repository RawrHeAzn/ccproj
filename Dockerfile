# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
# Prevents Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED=1
# Set noninteractive frontend for apt-get to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive
# Tell ODBC where to find system-wide config
ENV ODBCSYSINI=/etc
# Ensure the ODBC library path is included
ENV LD_LIBRARY_PATH=/opt/microsoft/msodbcsql17/lib64:${LD_LIBRARY_PATH}

# Install system dependencies including ODBC driver and tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl gnupg unixodbc unixodbc-dev unixodbc-bin && \
    # Add Microsoft repository for ODBC driver
    curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - && \
    curl https://packages.microsoft.com/config/ubuntu/$(lsb_release -rs)/prod.list > /etc/apt/sources.list.d/mssql-release.list && \
    apt-get update && \
    # Install the ODBC driver, accepting the EULA
    ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql17 && \
    # Configure odbcinst.ini
    printf "[ODBC Driver 17 for SQL Server]\nDescription=Microsoft ODBC Driver 17 for SQL Server\nDriver=/opt/microsoft/msodbcsql17/lib64/libmsodbcsql-17.so\nUsageCount=1\n" > /etc/odbcinst.ini && \
    # Clean up apt lists to reduce image size
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

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