# Use the builder image to install dependencies
FROM python:3.12.4-bookworm AS builder
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FLASK_APP=application
RUN python -m venv .venv
COPY requirements.txt ./
RUN .venv/bin/pip install -r requirements.txt

# Use a slim image for the runtime environment
FROM python:3.12.4-slim-bookworm
WORKDIR /app
COPY --from=builder /app/.venv .venv/
COPY . .

# Install dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

ENV FLASK_APP=application
CMD ["/app/.venv/bin/flask", "run", "--host=0.0.0.0", "--port=8080"]
