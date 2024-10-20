FROM python:3.8-slim

WORKDIR /app

COPY Requirements.txt .
RUN pip install --no-cache-dir -r Requirements.txt

COPY src/ .

CMD ["python", "app.py"]