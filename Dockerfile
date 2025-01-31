# Используем базовый образ с поддержкой GPU
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Устанавливаем Python 3.9 и необходимые системные зависимости
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем зависимости
COPY requirements.txt .

# Устанавливаем Python-зависимости
RUN pip3 install --no-cache-dir -r requirements.txt

# Устанавливаем Ollama через скрипт
RUN curl -fsSL https://ollama.com/install.sh | sh

# Копируем исходный код приложения
COPY . .

RUN ollama serve & \
    sleep 5 && \
    ollama pull deepseek-r1:8b


# Указываем порт, который будет использоваться приложением
EXPOSE 7000

# Команда для запуска приложения
CMD ["/bin/bash", "-c", "ollama serve & exec uvicorn main:app --host 0.0.0.0 --port 7000 --workers 4 --proxy-headers --forwarded-allow-ips '*'"]
