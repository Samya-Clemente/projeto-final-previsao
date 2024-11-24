
# Use a imagem base oficial do Python
FROM python:3.9-slim

# Defina o diretório de trabalho
WORKDIR /app

# Copie o arquivo requirements.txt para o contêiner
COPY requirements.txt .

# Instale as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copie o restante do código da aplicação para o contêiner
COPY . .

# Exponha a porta que o Streamlit usa
EXPOSE 8501

# Comando para rodar a aplicação
CMD ["streamlit", "run", "front-end.py"]