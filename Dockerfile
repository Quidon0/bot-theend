FROM python:3.11-slim

WORKDIR /app

# Установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование кода и базы знаний
COPY mainAI.py .
COPY knowledge_base.json .

EXPOSE 8001

CMD ["python", "mainAI.py"]