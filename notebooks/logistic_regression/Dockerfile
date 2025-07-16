# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy files
COPY requirements.txt ./
COPY main.py ./
COPY logistic_regression_pipeline.pkl ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Default command
CMD ["python", "main.py", "1700", "120", "45", "20", "9", "50"]