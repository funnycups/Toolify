FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

ENV PYTHONUNBUFFERED=1 \
	PYTHONIOENCODING=UTF-8 \
	PYTHONDONTWRITEBYTECODE=1

CMD ["python", "-u", "main.py"]