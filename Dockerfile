FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

# Pinger aur Uvicorn dono ko ek saath chala rahe hain
CMD ["sh", "-c", "python pinger.py & uvicorn app:app --host 0.0.0.0 --port 7860"]