# --- Estágio 1: Base e Instalação de Dependências do Sistema ---
FROM python:3.11-slim as builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# Instala as dependências do sistema operacional necessárias
# Colocando cada pacote em sua linha para clareza
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      cmake \
      libgl1 \
      libglib2.0-0 \
    && \
    rm -rf /var/lib/apt/lists/*

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Copia o ficheiro de requisitos primeiro (para aproveitar o cache do Docker)
COPY requirements.txt .

# Cria um ambiente virtual dentro do container (boa prática)
RUN python -m venv /opt/venv
# Ativa o venv para os próximos comandos RUN
ENV PATH="/opt/venv/bin:$PATH"

# Instala as dependências Python listadas no requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# --- Estágio 2: Imagem Final de Execução ---
FROM python:3.11-slim

# Define o diretório de trabalho
WORKDIR /app

# Copia o ambiente virtual com as dependências instaladas do estágio 'builder'
COPY --from=builder /opt/venv /opt/venv

# Copia o código da nossa aplicação para o container
COPY main.py .

# Define variáveis de ambiente para produção
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/venv/bin:$PATH"
ENV HOST="0.0.0.0"
ENV PORT="8000"

# Expõe a porta que a nossa aplicação vai usar
EXPOSE 8000

# Comando para executar a aplicação quando o container iniciar
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]