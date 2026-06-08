@echo off
title SiteOfSites AI Support Bot
python requirements.py
echo Запуск интеллектуальной системы поддержки...
echo Сервер будет доступен по адресу: http://127.0.0.1:8001
echo Нажмите Ctrl+C для остановки.
uvicorn app.api.main:app --host 0.0.0.0 --port 8001 --reload
pause