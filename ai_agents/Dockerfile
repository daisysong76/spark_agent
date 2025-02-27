# Use an official lightweight Python base
FROM python:3.9-slim

# For better performance in Docker
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt /app/requirements.txt

# Install dependencies (pin versions in your requirements.txt for reproducibility)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . /app

# Expose FastAPI port
EXPOSE 8000

# In production, run with a robust server (e.g., gunicorn + uvicorn workers)
CMD ["uvicorn", "ai_agent.app.main:app", "--host", "0.0.0.0", "--port", "8000"]


# Production notes:

#     You might switch to a multi-stage build or use poetry/conda if needed.
#     For heavy ML dependencies, consider a base image with system-level libraries installed (libtorch, etc.).

