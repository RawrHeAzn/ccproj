#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

echo "Updating apt-get..."
apt-get update

echo "Installing Microsoft SQL ODBC Driver..."
curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
curl https://packages.microsoft.com/config/ubuntu/20.04/prod.list > /etc/apt/sources.list.d/mssql-release.list
apt-get update
ACCEPT_EULA=Y apt-get install -y msodbcsql17 unixodbc-dev

echo "Installing Python requirements..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Build completed successfully."
