# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run uvicorn server when the container launches
# Use the host and port from the config_loader defaults if not overridden by config.yaml
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]