# Use official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirement.txt .

# Install any needed packages specified in requirement.txt
RUN pip install --no-cache-dir -r requirement.txt

# Copy the rest of the application code
COPY . .

# Railway assigns a dynamic port via the $PORT environment variable.
# We expose a default but the run command must use the variable.
EXPOSE 8501

# Run the application using the dynamic $PORT
CMD sh -c "streamlit run bioverse.py --server.port=${PORT:-8501} --server.address=0.0.0.0"
