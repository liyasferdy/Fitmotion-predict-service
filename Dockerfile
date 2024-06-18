# Use the official Python image from the Docker Hub
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install onnxruntime

# Copy the rest of the application code
COPY . .

# Set the PORT environment variable
ENV PORT 8080

# Expose the port that the application will run on
EXPOSE 8080

# Command to run the application
CMD ["python", "main.py"]