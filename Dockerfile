# Use a lightweight Python base image
FROM python:3.9-slim

# Define build-time argument for the model location
ARG MODEL_PATH=./model.pkl

# Install dependencies
RUN pip install fastapi pandas scikit-learn uvicorn python-multipart

# Copy the model into the container
COPY ${MODEL_PATH} /app/model.pkl

# Copy the application code
COPY app/ /app
WORKDIR /app

# Expose the scoring API's port
EXPOSE 8888

# Run the scoring service
CMD ["python", "service.py"]
