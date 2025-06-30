#!/bin/bash

# Setup script to create directories with proper permissions
# Run this BEFORE docker-compose up to avoid permission issues

echo " Setting up directories for MLflow and Docker volumes..."

# Create directories if they don't exist
mkdir -p artifacts
mkdir -p mlruns  
mkdir -p mlflow_db
mkdir -p data/raw
mkdir -p data/processed


# Set proper permissions (read/write for user, group, and others)
chmod 755 artifacts mlruns mlflow_db

# Get current user ID and group ID
USER_ID=$(id -u)
GROUP_ID=$(id -g)

echo "  Directories created with proper permissions:"
echo "   - artifacts/ (for MLflow artifacts)"
echo "   - mlruns/ (for MLflow experiment tracking)"  
echo "   - mlflow_db/ (for MLflow database)"
echo ""
echo " Current user: $(whoami) (UID: $USER_ID, GID: $GROUP_ID)"
echo " Directory permissions:"
ls -la | grep -E "(artifacts|mlruns|mlflow_db)"
echo ""
echo " Ready to run: docker-compose up" 