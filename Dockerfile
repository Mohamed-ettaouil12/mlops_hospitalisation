FROM python:3.10-slim

WORKDIR /app

# Copier les dépendances et les installer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le projet
COPY . .

# Lancer l'API
CMD ["python", "src/api.py"]