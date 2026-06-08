@echo off
chcp 65001
echo Тестирование AI бота...
echo.

set /p MSG="Введите ваш вопрос боту: "

echo.
echo Отправка запроса...
curl -X POST "http://127.0.0.1:8001/api/ai/chat" -H "Content-Type: application/json" -d "{\"message\": \"%MSG%\"}"
echo.
pause