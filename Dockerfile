FROM python:3.10.4-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY Iris-Prediction/ ./Iris-Prediction/

EXPOSE 5000 

CMD ["python", "Iris-Prediction/app.py"]