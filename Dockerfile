FROM python:3.10-slim

# Create a non-root user that Hugging Face expects (UID 1000)
RUN useradd -m -u 1000 user

WORKDIR /app

# Switch to root to install system dependencies
USER root

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    libx11-xcb1 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-xinerama0 \
    libxcb-xkb1 \
    libxkbcommon-x11-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and fix permissions so the 'user' can access them
COPY --chown=user:user requirements.txt .

# Switch to the non-root user for installing packages and running the app
USER user
ENV PATH="/home/user/.local/bin:${PATH}"

RUN pip install --no-cache-dir --user -r requirements.txt

# Copy the rest of your app code and ensure correct ownership
COPY --chown=user:user . .

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]