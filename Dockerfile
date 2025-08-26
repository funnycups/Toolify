FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

ENV PYTHONUNBUFFERED=1 \
	PYTHONIOENCODING=UTF-8 \
	PYTHONDONTWRITEBYTECODE=1

ENTRYPOINT ["python", "-u", "-m", "uvicorn"]

CMD ["main:app", "--host", "0.0.0.0", "--port", "8000"]