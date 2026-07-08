# Start from an official Python image
# Why 3.13-slim: matches your Python version, slim = smaller image size
FROM python:3.13-slim

# Set the working directory inside the container
# Why: all our commands run from this folder inside the container
WORKDIR /app

# Copy requirements first
# Why: Docker caches this layer — if requirements don't change,
# it won't reinstall packages on every rebuild. Saves time.
COPY requirements.txt .

# Install all Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything else into the container
# Why: after packages are installed, copy our actual code
COPY . .

# The command to run when container starts
CMD ["python", "telecom_agent.py"]