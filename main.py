# --- Importações de Bibliotecas ---
import io
import face_recognition
import numpy as np
import cv2 
import easyocr
import httpx 
import re 
import tempfile 
import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Path 
from fastapi.middleware.cors import CORSMiddleware # <<< IMPORTAÇÃO DO CORS
from pydantic import BaseModel, Field
from PIL import Image
from typing import List, Dict, Any 
from scipy.spatial import distance as dist 

# --- ======================================================= ---
# --- BASE DE DADOS "MOCK" PARA BACKGROUND CHECK              ---
# --- ======================================================= ---
MOCK_RISK_LIST = {
    "11111111111": {
        "is_pep": True, 
        "on_sanctions_list": False, 
        "risk_level": "ALTO", 
        "details": "Cliente identificado como Pessoa Exposta Politicamente (PEP)."
    },
    "22222222222": {
        "is_pep": False, 
        "on_sanctions_list": True, 
        "risk_level": "BLOQUEAR", 
        "details": "Cliente encontrado em lista de sanções internacionais (OFAC). Transação bloqueada."
    },
    "33333333333": {
        "is_pep": True, 
        "on_sanctions_list": True, 
        "risk_level": "BLOQUEAR", 
        "details": "Cliente PEP e em lista de sanções. Risco máximo."
    }
}

# --- Constantes para Liveness (Ajustáveis) ---
MAR_THRESHOLD = 0.35 # Limiar para considerar um sorriso (ajustar experimentalmente)
MAR_CONSECUTIVE_FRAMES = 2 # Quantos frames seguidos acima do limiar para confirmar sorriso
MAX_FRAMES_TO_PROCESS = 60 # Analisar no máximo ~2 segundos de vídeo (a 30fps)

# --- Configuração Inicial do EasyOCR ---
# Executado uma vez quando o servidor arranca.
try:
    print("A carregar o modelo EasyOCR para Português...")
    reader = easyocr.Reader(['pt'], gpu=False)
    print("Modelo EasyOCR carregado com sucesso.")
except Exception as e:
    print(f"Erro CRÍTICO ao carregar o modelo EasyOCR: {e}")
    reader = None 

# --- Modelos de Dados (Pydantic Schemas) ---

class OCRResponse(BaseModel):
    filename: str
    content_type: str
    extracted_text: str

class LivenessCheckResponse(BaseModel):
    liveness_check_passed: bool = Field(..., description="O teste de Prova de Vida (ex: sorriso) passou?")
    action_detected: str | None = Field(None, example="smile", description="Qual ação foi detectada (se passou)?")
    face_match_passed: bool | None = Field(None, description="O rosto do vídeo bate com o do documento (se liveness passou)?")
    face_distance: float | None = Field(None, example=0.45, description="Distância entre os rostos (se comparados).")
    confidence_score: float | None = Field(None, example=55.0, description="Score de confiança da comparação (se comparados).")
    detail: str = Field(..., example="Prova de vida confirmada (sorriso detectado) e rosto compatível.")

class KYCResponse(BaseModel):
    cpf: str = Field(..., example="123.456.789-00")
    situacao_cadastral: str = Field(..., example="REGULAR")
    nome: str = Field(..., example="Joao da Silva")
    data_nascimento: str = Field(..., example="1980-01-01")
    ano_obito: str | None = Field(..., example=None) 
    source: str = Field(..., example="BrasilAPI (Receita Federal v2)")

class KYBResponse(BaseModel):
    cnpj: str = Field(..., example="00.000.000/0001-00")
    razao_social: str = Field(..., example="NOME DA EMPRESA LTDA")
    nome_fantasia: str | None = Field(..., example="NOME FANTASIA")
    situacao_cadastral: str = Field(..., example="ATIVA")
    data_situacao_cadastral: str | None = Field(..., example="2005-11-03")
    data_inicio_atividade: str | None = Field(..., example="1990-01-01")
    cnae_fiscal_descricao: str = Field(..., example="Atividade principal da empresa")
    logradouro: str | None
    numero: str | None
    bairro: str | None
    municipio: str | None
    uf: str = Field(..., example="SP")
    cep: str | None
    telefone: str | None = Field(..., example="(11) 99999-9999")
    email: str | None = Field(..., example="contato@empresa.com")
    quadro_societario: List[Dict[str, Any]] = Field(..., description="Lista de sócios da empresa")
    source: str = Field(..., example="BrasilAPI (Receita Federal v1)")

class BackgroundCheckResponse(BaseModel):
    cpf: str = Field(..., example="111.111.111-11")
    is_pep: bool = Field(..., description="É uma Pessoa Exposta Politicamente?")
    on_sanctions_list: bool = Field(..., description="Está em alguma lista de sanções (ex: OFAC)?")
    risk_level: str = Field(..., example="ALTO", description="Nível de risco calculado (BAIXO, ALTO, BLOQUEAR)")
    details: str = Field(..., example="Cliente identificado como Pessoa Exposta Politicamente (PEP).")


# --- Helper Function: Calcular o Mouth Aspect Ratio (MAR) ---
def calculate_mar(mouth_landmarks):
    # Standard 68 points (0-indexed): Mouth 48-67. Inner lips 60-67. Corners 60, 64.
    # Verifica se a lista tem o número esperado de pontos (20 para a boca 48-67)
    if len(mouth_landmarks) < 20: 
        print("Aviso: Landmarks da boca incompletos para calcular MAR.")
        return 0.0 
    try:
        # Mapeamento de índices asssumindo que mouth_landmarks[0] é o ponto 48
        p61 = mouth_landmarks[13] # Ponto 61 (índice 13)
        p62 = mouth_landmarks[14] # Ponto 62 (índice 14)
        p63 = mouth_landmarks[15] # Ponto 63 (índice 15)
        p67 = mouth_landmarks[19] # Ponto 67 (índice 19)
        p66 = mouth_landmarks[18] # Ponto 66 (índice 18)
        p65 = mouth_landmarks[17] # Ponto 65 (índice 17)
        p60 = mouth_landmarks[12] # Ponto 60 (índice 12)
        p64 = mouth_landmarks[16] # Ponto 64 (índice 16)

        # Calcula pontos médios dos lábios internos superior e inferior
        mid_upper = (np.array(p61) + np.array(p62) + np.array(p63)) / 3
        mid_lower = (np.array(p67) + np.array(p66) + np.array(p65)) / 3
        
        # Distância vertical (altura da boca interna)
        A = dist.euclidean(mid_upper, mid_lower)
        # Distância horizontal (largura da boca nos cantos)
        C = dist.euclidean(p60, p64) 

        # Evitar divisão por zero
        if C == 0: 
            return 0.0
            
        mar = A / C
        return mar
        
    except IndexError:
        # Segurança caso o mapeamento de índices esteja inesperado
        print("Aviso: Índice fora dos limites ao calcular MAR. Verifique landmarks da boca.")
        return 0.0
    except Exception as e:
        # Captura outros erros inesperados
        print(f"Erro inesperado ao calcular MAR: {e}")
        return 0.0


# --- Inicialização da Aplicação FastAPI ---
app = FastAPI(
    title="Plataforma de Onboarding - Engine de Serviços",
    description="Microserviços para OCR, Biometria com Liveness Ativa, KYC (CPF/CNPJ) e Background Check.",
    version="6.1.0" # CORS adicionado!
)

# --- ======================================================= ---
# --- CONFIGURAÇÃO DO CORS (Cross-Origin Resource Sharing)    ---
# --- ======================================================= ---
# Lista das origens (frontends) que podem aceder a esta API
origins = [
    "https://lion-onboarding-ui.vercel.app", # O teu frontend na Vercel
    "http://localhost:3000",             # Desenvolvimento React local (porta comum)
    "http://localhost:8080",             # Servidor de teste frontend (python -m http.server)
    # Adiciona aqui outras origens se necessário (ex: outros domínios, portas diferentes)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # Permite apenas as origens listadas
    allow_credentials=True,    # Permite cookies (se usares no futuro)
    allow_methods=["*"],       # Permite todos os métodos (GET, POST, etc.)
    allow_headers=["*"],       # Permite todos os cabeçalhos
)
# --- ======================================================= ---


# --- Endpoint de "Saúde" (Raiz) ---
@app.get("/")
def read_root():
    """ Rota raiz para verificar se a API está online. """
    return {"status": "Engine de Onboarding Online", "serviços": ["/ocr", "/biometry/liveness", "/kyc/enrich-cpf", "/kyc/enrich-cnpj", "/kyc/background-check"]}

# --- ======================================================= ---
# --- SERVIÇO 1: EXTRAÇÃO DE TEXTO (OCR)                      ---
# --- ======================================================= ---
@app.post("/ocr/extract-text", response_model=OCRResponse)
async def extract_text_from_image(file: UploadFile = File(...)):
    """ Endpoint para extrair texto de uma imagem (documento) usando EasyOCR. """
    if reader is None:
        raise HTTPException(
            status_code=500, 
            detail="Erro Interno: O leitor de OCR (EasyOCR) não foi inicializado corretamente."
        )

    supported_types = ["image/jpeg", "image/png"]
    if file.content_type not in supported_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Tipo de ficheiro não suportado. Envie um JPG ou PNG. (Recebido: {file.content_type})"
        )

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Não foi possível descodificar a imagem. Ficheiro pode estar corrompido.")

        results = reader.readtext(img, detail=0, paragraph=False)
        full_text = "\n".join(results)

        if not full_text.strip():
            full_text = "[Nenhum texto legível encontrado na imagem]"

        return OCRResponse(
            filename=file.filename,
            content_type=file.content_type,
            extracted_text=full_text
        )

    except Exception as e:
        # Log do erro no servidor seria útil aqui em produção
        print(f"Erro no endpoint OCR: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Ocorreu um erro ao processar a imagem com EasyOCR: {str(e)}"
        )


# --- ======================================================= ---
# --- SERVIÇO 2: BIOMETRIA FACIAL COM LIVENESS ATIVA           ---
# --- ======================================================= ---
@app.post("/biometry/perform-liveness-check", response_model=LivenessCheckResponse)
async def perform_liveness_check(
    doc_image: UploadFile = File(..., description="Foto do documento (ex: CNH)"),
    liveness_video: UploadFile = File(..., description="Vídeo curto (.mp4, .webm) do cliente a sorrir.")
):
    """
    Endpoint para realizar Prova de Vida Ativa (detecção de sorriso em vídeo)
    e comparar o rosto com a foto do documento.
    """
    video_path = None # Inicializa para garantir que a variável existe no finally
    
    try:
        # --- PASSO 1: Salvar o Vídeo Temporariamente ---
        # Garante que temos um nome de ficheiro, mesmo que não seja enviado
        video_filename = liveness_video.filename if liveness_video.filename else "liveness_video.tmp"
        video_suffix = os.path.splitext(video_filename)[1] 
        # Cria ficheiro temporário de forma segura
        with tempfile.NamedTemporaryFile(delete=False, suffix=video_suffix) as temp_video:
            content = await liveness_video.read()
            if not content:
                 raise HTTPException(status_code=400, detail="Ficheiro de vídeo de liveness está vazio.")
            temp_video.write(content)
            video_path = temp_video.name # Guarda o caminho do ficheiro temporário
            
        # --- PASSO 2: Análise de Liveness (Detecção de Sorriso no Vídeo) ---
        
        video_capture = cv2.VideoCapture(video_path)
        if not video_capture.isOpened():
            # Log adicional pode ser útil
            print(f"Erro: Não foi possível abrir o vídeo em {video_path}")
            raise HTTPException(status_code=400, detail="Não foi possível abrir o ficheiro de vídeo de liveness.")

        smile_detected = False
        frame_count = 0
        mar_values = []
        best_liveness_frame = None 
        best_liveness_encoding = None 
        
        while frame_count < MAX_FRAMES_TO_PROCESS:
            ret, frame = video_capture.read()
            if not ret: 
                # Chegou ao fim do vídeo antes de atingir MAX_FRAMES
                print(f"Fim do vídeo atingido no frame {frame_count}")
                break 
            
            frame_count += 1
            
            # Converte frame para RGB (face_recognition usa RGB)
            # Redimensionar pode acelerar, mas pode perder precisão nos landmarks
            # frame_small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5) 
            # rgb_frame = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detecta rostos no frame atual (modelo 'hog' é mais rápido que 'cnn')
            face_locations = face_recognition.face_locations(rgb_frame, model="hog") 
            
            # Só processa landmarks se encontrou UM rosto (mais robusto)
            if len(face_locations) == 1:
                face_landmarks_list = face_recognition.face_landmarks(rgb_frame, face_locations)

                if face_landmarks_list: # Landmarks encontrados para o rosto
                    landmarks = face_landmarks_list[0]
                    
                    # Verifica se temos os pontos da boca
                    if 'mouth' in landmarks:
                        current_mar = calculate_mar(landmarks['mouth'])
                        mar_values.append(current_mar)

                        # Detecção do Sorriso
                        if current_mar > MAR_THRESHOLD:
                            # Conta quantos dos últimos MAR_CONSECUTIVE_FRAMES estavam acima
                            smile_counter = sum(1 for mar_val in mar_values[-MAR_CONSECUTIVE_FRAMES:] if mar_val > MAR_THRESHOLD)
                            
                            if smile_counter >= MAR_CONSECUTIVE_FRAMES:
                                print(f"Sorriso detectado no frame {frame_count} com MAR {current_mar:.2f}")
                                smile_detected = True
                                # Captura o encoding deste frame para comparação
                                # Usa a location já encontrada para otimizar
                                face_encodings = face_recognition.face_encodings(rgb_frame, [face_locations[0]]) 
                                if face_encodings:
                                    best_liveness_encoding = face_encodings[0]
                                    # best_liveness_frame = frame # Guardar o frame BGR se precisar para debug
                                break # Sorriso detectado, pode parar de processar
                        
                    else: # MAR abaixo do threshold
                        mar_values.append(0.0)
                else: # Landmarks da boca não encontrados
                     mar_values.append(0.0) 
            else: # Nenhum rosto ou múltiplos rostos detectados
                 if len(face_locations) > 1:
                     print(f"Aviso: Múltiplos rostos detectados no frame {frame_count}. Ignorando.")
                 mar_values.append(0.0) 

        video_capture.release() # Fecha o ficheiro de vídeo

        # --- PASSO 3: Verificar Resultado do Liveness ---
        if not smile_detected or best_liveness_encoding is None:
            print(f"Liveness falhou. Sorriso detectado: {smile_detected}, Encoding capturado: {best_liveness_encoding is not None}")
            return LivenessCheckResponse(
                liveness_check_passed=False,
                action_detected=None,
                face_match_passed=None,
                detail="Prova de vida falhou: Sorriso não detectado ou rosto não capturado claramente no vídeo."
            )
            
        # --- PASSO 4: Carregar e Processar Imagem do Documento ---
        doc_contents = await doc_image.read()
        if not doc_contents:
             raise HTTPException(status_code=400, detail="Ficheiro de imagem do documento está vazio.")
        doc_pil_image = Image.open(io.BytesIO(doc_contents))
        doc_np_image = np.array(doc_pil_image)
        
        # Encontra rosto no documento
        doc_face_encodings = face_recognition.face_encodings(doc_np_image)
        if not doc_face_encodings:
            print("Liveness OK, mas rosto não encontrado no documento.")
            # Nota: Considerar se isto deve ser um erro 400 ou parte da resposta
            return LivenessCheckResponse(
                liveness_check_passed=True, 
                action_detected="smile",
                face_match_passed=False, # Não houve comparação
                detail="Prova de vida OK, mas nenhum rosto encontrado na foto do documento para comparação."
            )
        # Assume o primeiro rosto encontrado no documento
        doc_encoding = doc_face_encodings[0] 

        # --- PASSO 5: Comparar Rosto do Vídeo com Rosto do Documento ---
        matches = face_recognition.compare_faces([doc_encoding], best_liveness_encoding, tolerance=0.6)
        is_match = bool(matches[0]) 
        face_distance = face_recognition.face_distance([doc_encoding], best_liveness_encoding)[0]
        # Converte distância para score de confiança (0-100), onde 0.6 = 0%, 0.0 = 100%
        # Ajuste: A fórmula anterior estava incorreta. Corrigindo:
        confidence = max(0.0, (0.6 - face_distance) / 0.6) * 100 if face_distance <= 0.6 else 0.0
        # Ou a mais simples 1.0 - distance: confidence = max(0.0, (1.0 - face_distance)) * 100
        # Mantendo a 1.0 - distance por enquanto, mas ciente do limite 0.6
        confidence = max(0.0, (1.0 - face_distance)) * 100


        print(f"Comparação facial: Match={is_match}, Distância={face_distance:.4f}, Confiança={confidence:.2f}%")

        # --- PASSO 6: Retornar o Resultado Completo ---
        if is_match:
            detail = f"Prova de vida confirmada (sorriso detectado) e rosto compatível com documento. (Score: {confidence:.2f}%)"
        else:
            detail = f"Prova de vida confirmada (sorriso detectado), MAS rosto INCOMPATÍVEL com documento. (Score: {confidence:.2f}%)"

        return LivenessCheckResponse(
            liveness_check_passed=True,
            action_detected="smile",
            face_match_passed=is_match,
            face_distance=round(face_distance, 4),
            confidence_score=round(confidence, 2),
            detail=detail
        )

    except HTTPException as http_exc:
        # Re-levanta exceções HTTP que já criámos (ex: vídeo vazio)
        raise http_exc
    except Exception as e:
        # Captura qualquer outro erro inesperado
        print(f"Erro inesperado durante liveness check: {e}") # Log detalhado no servidor
        # Retorna uma resposta genérica para o cliente
        return LivenessCheckResponse(
            liveness_check_passed=False,
            action_detected=None,
            face_match_passed=None,
            detail=f"Erro interno durante o processamento de liveness. Contacte o suporte."
        )
    finally:
        # --- Limpeza: Apagar o ficheiro de vídeo temporário ---
        if video_path and os.path.exists(video_path):
            try:
                os.remove(video_path)
                print(f"Ficheiro temporário {video_path} removido.")
            except Exception as e:
                print(f"Erro ao remover ficheiro temporário {video_path}: {e}")


# --- ======================================================= ---
# --- SERVIÇO 3: ENRIQUECIMENTO DE CPF (KYC)                   ---
# --- ======================================================= ---
@app.get("/kyc/enrich-cpf/", response_model=KYCResponse) 
async def enrich_cpf(
    cpf: str = Query(..., description="O número do CPF para consulta. Pode conter pontos e traços.", min_length=11, max_length=14)
):
    """
    Endpoint para enriquecer dados de CPF (KYC) consultando fontes públicas.
    Fonte: BrasilAPI (Receita Federal v2) - Nota: Fonte gratuita com dados limitados.
    """
    
    cpf_limpo = re.sub(r'\D', '', cpf)
    if len(cpf_limpo) != 11:
        raise HTTPException(status_code=400, detail="CPF inválido. O CPF deve conter 11 dígitos.")

    url = f"https://brasilapi.com.br/api/cpf/v2/{cpf_limpo}"
    
    try:
        async with httpx.AsyncClient() as client:
            # Timeout um pouco maior pode ajudar com APIs externas lentas
            response = await client.get(url, timeout=15.0) 
            # Verifica se a resposta foi sucesso (2xx)
            response.raise_for_status() 
            # Converte a resposta JSON
            data = response.json()
            
            # Monta a resposta Pydantic
            return KYCResponse(
                cpf=data.get("numero_de_cpf", cpf), # Usa o CPF original com máscara se disponível
                situacao_cadastral=data.get("situacao_cadastrar", "DESCONHECIDA"),
                nome=data.get("nome", "N/A"),
                data_nascimento=data.get("data_nascimento", "N/A"),
                ano_obito=data.get("ano_obito"), # Se for nulo na API, Pydantic aceita
                source="BrasilAPI (Receita Federal v2)"
            )
            
    except httpx.HTTPStatusError as e:
        # Trata erros específicos da API externa
        if e.response.status_code == 404:
            # Log ou tratamento específico para CPF não encontrado
            print(f"CPF {cpf_limpo} não encontrado na BrasilAPI v2.")
            raise HTTPException(status_code=404, detail=f"CPF {cpf_limpo} não encontrado na base da Receita Federal (v2).")
        else:
            # Outros erros HTTP da API externa
            print(f"Erro HTTP da BrasilAPI v2: {e.response.status_code} - {e.response.text}")
            raise HTTPException(status_code=e.response.status_code, detail=f"Erro ao consultar a API externa de CPF: Status {e.response.status_code}")
            
    except httpx.RequestError as e:
        # Erros de conexão, timeout, etc.
        print(f"Erro de conexão com BrasilAPI v2: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Serviço de consulta de CPF (BrasilAPI) indisponível ou lento: {str(e)}")
        
    except Exception as e:
        # Outros erros inesperados no nosso código
        print(f"Erro inesperado no serviço de KYC: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro interno no serviço de KYC.")


# --- ======================================================= ---
# --- SERVIÇO 4: ENRIQUECIMENTO DE CNPJ (KYB)                  ---
# --- ======================================================= ---
@app.get("/kyc/enrich-cnpj/", response_model=KYBResponse)
async def enrich_cnpj(
    cnpj: str = Query(..., description="O número do CNPJ para consulta. Pode conter pontos, traços e barras.", min_length=14, max_length=18)
):
    """
    Endpoint para enriquecer dados de CNPJ (KYB) consultando fontes públicas.
    Fonte: BrasilAPI (Receita Federal v1) - Fonte de dados completa.
    """
    
    cnpj_limpo = re.sub(r'\D', '', cnpj)
    if len(cnpj_limpo) != 14:
        raise HTTPException(status_code=400, detail="CNPJ inválido. O CNPJ deve conter 14 dígitos.")
        
    url = f"https://brasilapi.com.br/api/cnpj/v1/{cnpj_limpo}"
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=15.0)
            response.raise_for_status()
            data = response.json()
            
            # Tratamento mais seguro para telefone, evitando " None"
            ddd = data.get('ddd_telefone_1')
            telefone = data.get('telefone_1')
            telefone_completo = f"{ddd} {telefone}".strip() if ddd and telefone else None

            return KYBResponse(
                cnpj=data.get("cnpj", cnpj), # Usa o CNPJ original com máscara
                razao_social=data.get("razao_social", "N/A"),
                nome_fantasia=data.get("nome_fantasia"),
                situacao_cadastral=data.get("descricao_situacao_cadastral", "DESCONHECIDA"),
                data_situacao_cadastral=data.get("data_situacao_cadastral"),
                data_inicio_atividade=data.get("data_inicio_atividade"),
                cnae_fiscal_descricao=data.get("cnae_fiscal_descricao"),
                logradouro=data.get("logradouro"),
                numero=data.get("numero"),
                bairro=data.get("bairro"),
                municipio=data.get("municipio"),
                uf=data.get("uf"),
                cep=data.get("cep"),
                telefone=telefone_completo,
                email=data.get("email"),
                quadro_societario=data.get("qsa", []), # Retorna lista vazia se não houver QSA
                source="BrasilAPI (Receita Federal v1)"
            )
            
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            print(f"CNPJ {cnpj_limpo} não encontrado na BrasilAPI v1.")
            raise HTTPException(status_code=404, detail=f"CNPJ {cnpj_limpo} não encontrado na base da Receita Federal.")
        else:
            print(f"Erro HTTP da BrasilAPI v1: {e.response.status_code} - {e.response.text}")
            raise HTTPException(status_code=e.response.status_code, detail=f"Erro ao consultar a API externa de CNPJ: Status {e.response.status_code}")
            
    except httpx.RequestError as e:
        print(f"Erro de conexão com BrasilAPI v1: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Serviço de consulta de CNPJ (BrasilAPI) indisponível ou lento: {str(e)}")
        
    except Exception as e:
        print(f"Erro inesperado no serviço de KYB: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro interno no serviço de KYB.")


# --- ======================================================= ---
# --- SERVIÇO 5: BACKGROUND CHECK (PEPs & SANÇÕES)             ---
# --- ======================================================= ---
@app.get("/kyc/background-check/", response_model=BackgroundCheckResponse)
async def background_check(
    cpf: str = Query(..., description="O CPF a ser verificado em listas de risco (PEPs, Sanções). Pode conter pontos e traços.", min_length=11, max_length=14)
):
    """
    Endpoint para simular a verificação de listas restritivas (PEPs, Sanções).
    Fonte: Base de dados interna "MOCK" (simulada).
    """
    
    cpf_limpo = re.sub(r'\D', '', cpf)
    if len(cpf_limpo) != 11:
        raise HTTPException(status_code=400, detail="CPF inválido. O CPF deve conter 11 dígitos.")
        
    # Consulta a base de dados "mock"
    risk_profile = MOCK_RISK_LIST.get(cpf_limpo)
    
    if risk_profile:
        # CPF encontrado na lista de risco
        return BackgroundCheckResponse(
            cpf=cpf, # Retorna o CPF original com máscara
            is_pep=risk_profile.get("is_pep", False),
            on_sanctions_list=risk_profile.get("on_sanctions_list", False),
            risk_level=risk_profile.get("risk_level", "ALTO"), # Nível padrão se encontrado
            details=risk_profile.get("details", "Risco identificado, detalhes não especificados.")
        )
    else:
        # CPF "limpo" (não encontrado na lista)
        return BackgroundCheckResponse(
            cpf=cpf, # Retorna o CPF original com máscara
            is_pep=False,
            on_sanctions_list=False,
            risk_level="BAIXO",
            details="Nenhum risco identificado em listas restritivas simuladas."
        )

# --- Fim do ficheiro main.py ---