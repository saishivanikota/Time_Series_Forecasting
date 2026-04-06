FROM python:3.14-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY website.py ./
COPY "Gold Price.csv" ./

EXPOSE 8501

CMD ["streamlit", "run", "website.py", "--server.port=8501", "--server.address=0.0.0.0"]
