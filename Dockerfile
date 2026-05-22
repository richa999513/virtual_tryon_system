FROM python:3.10-slim

# 1. Create the user and the app directory
RUN useradd -m -u 1000 user
WORKDIR /app

# 2. Switch to root to install system libs AND set folder permissions
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

# Fix permission: Give the 'user' ownership of the /app directory
RUN chown user:user /app

# 3. Switch to the user for the rest of the build
USER user
ENV PATH="/home/user/.local/bin:${PATH}"

# Copy requirements and install
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy the rest of the code
COPY --chown=user:user . .

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]