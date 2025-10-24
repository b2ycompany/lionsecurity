# --- Importações de Bibliotecas ---
import io
import face_recognition
import numpy as np
import cv2 # Essencial para processar vídeo
import easyocr
import httpx # Para chamar outras APIs
import re # Para limpar o CPF/CNPJ
import tempfile # Para guardar o vídeo temporariamente
import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Path
from fastapi.middleware.cors import CORSMiddleware # <<< IMPORTAÇÃO ADICIONADA
from pydantic import BaseModel, Field
from PIL import Image
from typing import List, Dict, Any
from scipy.spatial import distance as dist # Para calcular a distância (MAR do sorriso)

# --- ======================================================= ---
# --- BASE DE DADOS "MOCK" PARA BACKGROUND CHECK              ---
# --- ======================================================= ---
MOCK_RISK_LIST = {
    "11111111111": {"is_pep": True, "on_sanctions_list": False, "risk_level": "ALTO", "details": "Cliente PEP."},
    "22222222222": {"is_pep": False, "on_sanctions_list": True, "risk_level": "BLOQUEAR", "details": "Cliente em lista de sanções (OFAC)."},
    "33333333333": {"is_pep": True, "on_sanctions_list": True, "risk_level": "BLOQUEAR", "details": "Cliente PEP e em lista de sanções."}
}

# --- Constantes para Liveness (Ajustáveis) ---
MAR_THRESHOLD = 0.35
MAR_CONSECUTIVE_FRAMES = 2
MAX_FRAMES_TO_PROCESS = 60

# --- Configuração Inicial do EasyOCR (O nosso motor de OCR) ---
try:
    print("A carregar o modelo EasyOCR para Português...")
    reader = easyocr.Reader(['pt'], gpu=False)
    print("Modelo EasyOCR carregado com sucesso.")
except Exception as e:
    print(f"Erro CRÍTICO ao carregar o modelo EasyOCR: {e}")
    reader = None

# --- Modelos de Dados (para documentar a API) ---

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
    # Baseado nos 68 pontos do dlib (0-indexed)
    # Pontos internos da boca: 60 a 67
    # Pontos relevantes: 61, 62, 63 (lábio superior interno)
    #                    67, 66, 65 (lábio inferior interno)
    #                    60, 64 (cantos)
    try:
        # Altura vertical (média dos pontos internos)
        A = dist.euclidean(mouth_landmarks[61], mouth_landmarks[67]) # Ponto 62 vs 66 approx
        B = dist.euclidean(mouth_landmarks[62], mouth_landmarks[66])
        C = dist.euclidean(mouth_landmarks[63], mouth_landmarks[65])
        avg_height = (A + B + C) / 3.0

        # Largura horizontal (cantos)
        D = dist.euclidean(mouth_landmarks[60], mouth_landmarks[64])

        if D == 0: return 0.0 # Evitar divisão por zero
        mar = avg_height / D
        return mar
    except IndexError:
        # Caso a lista de landmarks da boca não tenha todos os pontos esperados
        return 0.0
    except Exception:
        # Outro erro inesperado
        return 0.0

# --- Inicialização da Aplicação FastAPI ---
app = FastAPI(
    title="Plataforma de Onboarding - Engine de Serviços",
    description="Microserviços para OCR, Biometria com Liveness Ativa, KYC (CPF/CNPJ) e Background Check.",
    version="6.1.0" # <<< VERSÃO ATUALIZADA COM CORS
)

# --- ======================================================= ---
# --- CONFIGURAÇÃO DO CORS (Cross-Origin Resource Sharing)    ---
# --- ======================================================= ---
origins = [
    "https://lion-onboarding-ui.vercel.app", # <<< O TEU FRONTEND NA VERCEL
    "http://localhost",         # Para testes locais gerais
    "http://localhost:8080",    # Para testes com `python -m http.server`
    "http://127.0.0.1",         # Alternativa para localhost
    "http://127.0.0.1:8080",    # Alternativa para localhost com porta
    # Adiciona aqui outros URLs se precisares no futuro (ex: staging)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # Lista de origens permitidas
    allow_credentials=True,    # Permite cookies (importante para sessões futuras)
    allow_methods=["*"],       # Permite todos os métodos HTTP (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],       # Permite todos os cabeçalhos HTTP
)
# --- ======================================================= ---


# --- Endpoint de "Saúde" (Raiz) ---
@app.get("/")
def read_root():
    return {"status": "Engine de Onboarding Online", "serviços": ["/ocr", "/biometry/liveness", "/kyc/enrich-cpf", "/kyc/enrich-cnpj", "/kyc/background-check"]}

# --- ======================================================= ---
# --- SERVIÇO 1: EXTRAÇÃO DE TEXTO (OCR)                      ---
# --- ======================================================= ---
@app.post("/ocr/extract-text", response_model=OCRResponse)
async def extract_text_from_image(file: UploadFile = File(...)):
    if reader is None: raise HTTPException(status_code=500, detail="Erro Interno: Leitor OCR não inicializado.")
    supported_types = ["image/jpeg", "image/png"]
    if file.content_type not in supported_types: raise HTTPException(status_code=400, detail=f"Tipo de ficheiro não suportado. (Recebido: {file.content_type})")
    try:
        contents = await file.read(); nparr = np.frombuffer(contents, np.uint8); img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None: raise HTTPException(status_code=400, detail="Não foi possível descodificar a imagem.")
        results = reader.readtext(img, detail=0, paragraph=False); full_text = "\n".join(results)
        if not full_text.strip(): full_text = "[Nenhum texto legível encontrado na imagem]"
        return OCRResponse(filename=file.filename, content_type=file.content_type, extracted_text=full_text)
    except Exception as e: raise HTTPException(status_code=500, detail=f"Erro no OCR: {str(e)}")


# --- ======================================================= ---
# --- SERVIÇO 2: BIOMETRIA FACIAL COM LIVENESS ATIVA           ---
# --- ======================================================= ---
@app.post("/biometry/perform-liveness-check", response_model=LivenessCheckResponse)
async def perform_liveness_check(
    doc_image: UploadFile = File(..., description="Foto do documento (ex: CNH)"),
    liveness_video: UploadFile = File(..., description="Vídeo curto (.mp4, .webm) do cliente a sorrir.")
):
    video_path = None
    try:
        # Salvar vídeo temporariamente
        video_suffix = os.path.splitext(liveness_video.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=video_suffix) as temp_video:
            content = await liveness_video.read()
            temp_video.write(content)
            video_path = temp_video.name

        # Análise de Liveness (Sorriso)
        video_capture = cv2.VideoCapture(video_path)
        if not video_capture.isOpened(): raise HTTPException(status_code=400, detail="Não foi possível abrir o vídeo.")

        smile_detected = False
        frame_count = 0
        mar_values = []
        best_liveness_encoding = None

        while frame_count < MAX_FRAMES_TO_PROCESS:
            ret, frame = video_capture.read()
            if not ret: break
            frame_count += 1
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame, model="hog")
            face_landmarks_list = face_recognition.face_landmarks(rgb_frame, face_locations)

            if face_landmarks_list:
                landmarks = face_landmarks_list[0]
                if 'mouth' in landmarks:
                    current_mar = calculate_mar(landmarks['mouth'])
                    mar_values.append(current_mar)
                    if current_mar > MAR_THRESHOLD:
                        smile_counter = sum(1 for mar in mar_values[-MAR_CONSECUTIVE_FRAMES:] if mar > MAR_THRESHOLD)
                        if smile_counter >= MAR_CONSECUTIVE_FRAMES:
                            smile_detected = True
                            face_encodings = face_recognition.face_encodings(rgb_frame, [face_locations[0]])
                            if face_encodings: best_liveness_encoding = face_encodings[0]
                            break # Sorriso detectado
                else: mar_values.append(0.0)
            else: mar_values.append(0.0)

        video_capture.release()

        if not smile_detected or best_liveness_encoding is None:
            return LivenessCheckResponse(liveness_check_passed=False, action_detected=None, face_match_passed=None, detail="Prova de vida falhou: Sorriso não detectado ou rosto não capturado.")

        # Comparação com Documento
        doc_contents = await doc_image.read()
        doc_pil_image = Image.open(io.BytesIO(doc_contents))
        doc_np_image = np.array(doc_pil_image)
        doc_face_encodings = face_recognition.face_encodings(doc_np_image)

        if not doc_face_encodings:
            return LivenessCheckResponse(liveness_check_passed=True, action_detected="smile", face_match_passed=False, detail="Prova de vida OK, mas rosto não encontrado no documento.")

        doc_encoding = doc_face_encodings[0]
        matches = face_recognition.compare_faces([doc_encoding], best_liveness_encoding, tolerance=0.6)
        is_match = bool(matches[0])
        face_distance = face_recognition.face_distance([doc_encoding], best_liveness_encoding)[0]
        confidence = max(0.0, (1.0 - face_distance)) * 100

        detail = f"Prova de vida confirmada (sorriso detectado) e rosto {'compatível' if is_match else 'INCOMPATÍVEL'} com documento. (Score: {confidence:.2f}%)"

        return LivenessCheckResponse(
            liveness_check_passed=True, action_detected="smile", face_match_passed=is_match,
            face_distance=round(face_distance, 4), confidence_score=round(confidence, 2), detail=detail
        )

    except Exception as e:
        return LivenessCheckResponse(liveness_check_passed=False, action_detected=None, face_match_passed=None, detail=f"Erro liveness: {str(e)}")
    finally:
        if video_path and os.path.exists(video_path): os.remove(video_path)


# --- ======================================================= ---
# --- SERVIÇO 3: ENRIQUECIMENTO DE CPF (KYC)                   ---
# --- ======================================================= ---
@app.get("/kyc/enrich-cpf/", response_model=KYCResponse)
async def enrich_cpf(cpf: str = Query(..., description="CPF (com ou sem pontos/traços)", min_length=11, max_length=14)):
    cpf_limpo = re.sub(r'\D', '', cpf);
    if len(cpf_limpo) != 11: raise HTTPException(status_code=400, detail="CPF inválido.")
    url = f"https://brasilapi.com.br/api/cpf/v2/{cpf_limpo}"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10.0); response.raise_for_status(); data = response.json()
            return KYCResponse(cpf=data.get("numero_de_cpf", cpf), situacao_cadastral=data.get("situacao_cadastrar", "DESCONHECIDA"), nome=data.get("nome", "N/A"), data_nascimento=data.get("data_nascimento", "N/A"), ano_obito=data.get("ano_obito"), source="BrasilAPI (Receita Federal v2)")
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404: raise HTTPException(status_code=404, detail=f"CPF {cpf_limpo} não encontrado (Base v2).")
        else: raise HTTPException(status_code=e.response.status_code, detail=f"Erro API externa: {e.response.text}")
    except httpx.RequestError as e: raise HTTPException(status_code=503, detail=f"Serviço BrasilAPI indisponível: {str(e)}")
    except Exception as e: raise HTTPException(status_code=500, detail=f"Erro interno KYC: {str(e)}")


# --- ======================================================= ---
# --- SERVIÇO 4: ENRIQUECIMENTO DE CNPJ (KYB)                  ---
# --- ======================================================= ---
@app.get("/kyc/enrich-cnpj/", response_model=KYBResponse)
async def enrich_cnpj(cnpj: str = Query(..., description="CNPJ (com ou sem pontos/traços/barras)", min_length=14, max_length=18)):
    cnpj_limpo = re.sub(r'\D', '', cnpj);
    if len(cnpj_limpo) != 14: raise HTTPException(status_code=400, detail="CNPJ inválido.")
    url = f"https://brasilapi.com.br/api/cnpj/v1/{cnpj_limpo}"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10.0); response.raise_for_status(); data = response.json()
            telefone_completo = f"{data.get('ddd_telefone_1') or ''} {data.get('telefone_1') or ''}".strip() or None
            return KYBResponse(cnpj=data.get("cnpj", cnpj), razao_social=data.get("razao_social", "N/A"), nome_fantasia=data.get("nome_fantasia"), situacao_cadastral=data.get("descricao_situacao_cadastral", "DESCONHECIDA"), data_situacao_cadastral=data.get("data_situacao_cadastral"), data_inicio_atividade=data.get("data_inicio_atividade"), cnae_fiscal_descricao=data.get("cnae_fiscal_descricao"), logradouro=data.get("logradouro"), numero=data.get("numero"), bairro=data.get("bairro"), municipio=data.get("municipio"), uf=data.get("uf"), cep=data.get("cep"), telefone=telefone_completo, email=data.get("email"), quadro_societario=data.get("qsa", []), source="BrasilAPI (Receita Federal v1)")
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404: raise HTTPException(status_code=404, detail=f"CNPJ {cnpj_limpo} não encontrado.")
        else: raise HTTPException(status_code=e.response.status_code, detail=f"Erro API externa: {e.response.text}")
    except httpx.RequestError as e: raise HTTPException(status_code=503, detail=f"Serviço BrasilAPI indisponível: {str(e)}")
    except Exception as e: raise HTTPException(status_code=500, detail=f"Erro interno KYB: {str(e)}")


# --- ======================================================= ---
# --- SERVIÇO 5: BACKGROUND CHECK (PEPs & SANÇÕES)             ---
# --- ======================================================= ---
@app.get("/kyc/background-check/", response_model=BackgroundCheckResponse)
async def background_check(cpf: str = Query(..., description="CPF a verificar (com ou sem pontos/traços)", min_length=11, max_length=14)):
    cpf_limpo = re.sub(r'\D', '', cpf);
    if len(cpf_limpo) != 11: raise HTTPException(status_code=400, detail="CPF inválido.")
    risk_profile = MOCK_RISK_LIST.get(cpf_limpo)
    if risk_profile:
        return BackgroundCheckResponse(cpf=cpf, is_pep=risk_profile.get("is_pep", False), on_sanctions_list=risk_profile.get("on_sanctions_list", False), risk_level=risk_profile.get("risk_level", "ALTO"), details=risk_profile.get("details", "Risco identificado."))
    else:
        return BackgroundCheckResponse(cpf=cpf, is_pep=False, on_sanctions_list=False, risk_level="BAIXO", details="Nenhum risco identificado.")