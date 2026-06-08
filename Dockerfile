# --- Stage 1: Build ---
FROM python:3.11-slim as builder
WORKDIR /app

# Добавлен libpq-dev, так как он необходим для сборки psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends gcc python3-dev libpq-dev && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# 1. Принудительно устанавливаем CPU-версию torch ДО установки requirements
RUN pip install --user --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 2. Устанавливаем остальные зависимости
RUN pip install --user --no-cache-dir -r requirements.txt

# --- Stage 2: Final ---
FROM python:3.11-slim
WORKDIR /app

# 3. Задаем пути, включая директорию для кэша моделей HuggingFace
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/home/aiuser/.local/bin:$PATH \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    HF_HOME=/app/.cache/huggingface

# Устанавливаем libpq5 для работы psycopg2 в финальном образе
RUN apt-get update && apt-get install -y --no-install-recommends libpq5 && rm -rf /var/lib/apt/lists/*

RUN groupadd -r aiuser && useradd -r -g aiuser -d /home/aiuser aiuser

COPY --from=builder /root/.local /home/aiuser/.local/

# 4. Копируем ВЕСЬ код, а не только отдельные папки
COPY . /app/

# Создаем папку для кэша и выдаем права пользователю aiuser
RUN mkdir -p /app/.cache/huggingface && chown -R aiuser:aiuser /app /home/aiuser

USER aiuser 

# 5. "Запекаем" веса модели прямо в образ при сборке
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

EXPOSE 8001
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "2"]