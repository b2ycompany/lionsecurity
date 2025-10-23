# --- Importações de Bibliotecas ---
import io
import face_recognition
import numpy as np
import cv2 # Essencial para processar vídeo
import easyocr
import httpx 
import re 
import tempfile # Para guardar o vídeo temporariamente
import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Path
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
# Mouth Aspect Ratio: Relação entre altura e largura da boca. Aumenta quando sorrimos.
MAR_THRESHOLD = 0.35 # Limiar para considerar um sorriso (ajustar experimentalmente)
MAR_CONSECUTIVE_FRAMES = 2 # Quantos frames seguidos acima do limiar para confirmar sorriso
MAX_FRAMES_TO_PROCESS = 60 # Analisar no máximo ~2 segundos de vídeo (a 30fps)

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

# NOVO MODELO DE RESPOSTA PARA LIVENESS
class LivenessCheckResponse(BaseModel):
    liveness_check_passed: bool = Field(..., description="O teste de Prova de Vida (ex: sorriso) passou?")
    action_detected: str | None = Field(None, example="smile", description="Qual ação foi detectada (se passou)?")
    face_match_passed: bool | None = Field(None, description="O rosto do vídeo bate com o do documento (se liveness passou)?")
    face_distance: float | None = Field(None, example=0.45, description="Distância entre os rostos (se comparados).")
    confidence_score: float | None = Field(None, example=55.0, description="Score de confiança da comparação (se comparados).")
    detail: str = Field(..., example="Prova de vida confirmada (sorriso detectado) e rosto compatível.")

class KYCResponse(BaseModel):
    cpf: str = Field(..., example="123.456.789-00")
    # ... (campos omitidos)
    source: str = Field(..., example="BrasilAPI (Receita Federal v2)")
    situacao_cadastral: str
    nome: str
    data_nascimento: str
    ano_obito: str | None

class KYBResponse(BaseModel):
    cnpj: str = Field(..., example="00.000.000/0001-00")
    # ... (campos omitidos)
    source: str = Field(..., example="BrasilAPI (Receita Federal v1)")
    razao_social: str
    nome_fantasia: str | None
    situacao_cadastral: str
    data_situacao_cadastral: str | None
    data_inicio_atividade: str | None
    cnae_fiscal_descricao: str
    logradouro: str | None
    numero: str | None
    bairro: str | None
    municipio: str | None
    uf: str
    cep: str | None
    telefone: str | None
    email: str | None
    quadro_societario: List[Dict[str, Any]]

class BackgroundCheckResponse(BaseModel):
    cpf: str = Field(..., example="111.111.111-11")
    is_pep: bool
    on_sanctions_list: bool
    risk_level: str
    details: str 


# --- Helper Function: Calcular o Mouth Aspect Ratio (MAR) ---
def calculate_mar(mouth_landmarks):
    # Pontos do contorno da boca (índices específicos do dlib)
    # Veja: https://pyimagesearch.com/wp-content/uploads/2017/04/facial_landmarks_68markup.jpg
    # Pontos verticais (lábio superior interno -> lábio inferior interno)
    A = dist.euclidean(mouth_landmarks[13], mouth_landmarks[19]) # Ponto 61 -> 67 (0-based) ? Dlib landmarks indexes are 1-based, face_recognition might be 0-based. Let's assume 0-based from face_recognition output. Points 51, 57 roughly? No, need the lip points. Let's use standard Dlib 68 points mapping. Mouth is 48-67. Inner lips points are 60-67.
    # Recalculating based on standard 68 points (0-indexed):
    # Inner upper lip: 61, 62, 63
    # Inner lower lip: 67, 66, 65
    # Corners: 60, 64
    # Let's use the distance between the midpoints of the inner upper and lower lips
    mid_upper = (np.array(mouth_landmarks[61]) + np.array(mouth_landmarks[62]) + np.array(mouth_landmarks[63])) / 3
    mid_lower = (np.array(mouth_landmarks[67]) + np.array(mouth_landmarks[66]) + np.array(mouth_landmarks[65])) / 3
    A = dist.euclidean(mid_upper, mid_lower)

    # Pontos horizontais (canto esquerdo -> canto direito da boca)
    C = dist.euclidean(mouth_landmarks[60], mouth_landmarks[64]) # Corner points

    # Calcular MAR
    if C == 0: return 0.0 # Evitar divisão por zero
    mar = A / C
    return mar

# --- Inicialização da Aplicação FastAPI ---
app = FastAPI(
    title="Plataforma de Onboarding - Engine de Serviços",
    description="Microserviços para OCR, Biometria com Liveness Ativa, KYC (CPF/CNPJ) e Background Check.",
    version="6.0.0" # Subimos para a v6.0! Liveness!
)

# --- Endpoint de "Saúde" (Raiz) ---
@app.get("/")
def read_root():
    return {"status": "Engine de Onboarding Online", "serviços": ["/ocr", "/biometry/liveness", "/kyc/enrich-cpf", "/kyc/enrich-cnpj", "/kyc/background-check"]}

# --- ======================================================= ---
# --- SERVIÇO 1: EXTRAÇÃO DE TEXTO (OCR)                      ---
# --- ======================================================= ---
# (Sem alterações no Serviço 1)
@app.post("/ocr/extract-text", response_model=OCRResponse)
async def extract_text_from_image(file: UploadFile = File(...)):
    if reader is None: raise HTTPException(status_code=500, detail="Erro Interno: Leitor OCR não inicializado.")
    # ... (restante do código OCR igual) ...
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
# Endpoint SUBSTITUÍDO!
@app.post("/biometry/perform-liveness-check", response_model=LivenessCheckResponse)
async def perform_liveness_check(
    doc_image: UploadFile = File(..., description="Foto do documento (ex: CNH)"),
    liveness_video: UploadFile = File(..., description="Vídeo curto (.mp4, .webm) do cliente a sorrir.")
):
    """
    Endpoint para realizar Prova de Vida Ativa (detecção de sorriso em vídeo)
    e comparar o rosto com a foto do documento.
    """
    
    video_path = None # Inicializa o caminho do vídeo
    
    try:
        # --- PASSO 1: Salvar o Vídeo Temporariamente ---
        # OpenCV precisa de um caminho de ficheiro, não pode ler bytes diretamente
        # Usamos 'tempfile' para criar um ficheiro temporário seguro
        video_suffix = os.path.splitext(liveness_video.filename)[1] # Pega a extensão (.mp4, .webm)
        with tempfile.NamedTemporaryFile(delete=False, suffix=video_suffix) as temp_video:
            content = await liveness_video.read()
            temp_video.write(content)
            video_path = temp_video.name
            
        # --- PASSO 2: Análise de Liveness (Detecção de Sorriso no Vídeo) ---
        
        video_capture = cv2.VideoCapture(video_path)
        if not video_capture.isOpened():
            raise HTTPException(status_code=400, detail="Não foi possível abrir o ficheiro de vídeo.")

        smile_detected = False
        frame_count = 0
        mar_values = []
        best_liveness_frame = None # Frame onde o sorriso foi detectado com rosto
        best_liveness_encoding = None # Encoding do rosto nesse frame
        
        while frame_count < MAX_FRAMES_TO_PROCESS:
            ret, frame = video_capture.read()
            if not ret: break # Fim do vídeo
            
            frame_count += 1
            
            # Converte frame para RGB (face_recognition usa RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detecta rostos no frame atual
            face_locations = face_recognition.face_locations(rgb_frame, model="hog") # 'hog' é mais rápido
            face_landmarks_list = face_recognition.face_landmarks(rgb_frame, face_locations)

            if face_landmarks_list: # Se encontrou um rosto com landmarks
                # Vamos assumir apenas o primeiro rosto encontrado
                landmarks = face_landmarks_list[0]
                
                # Verifica se temos os pontos da boca (precisamos do índice 60 a 67)
                if 'mouth' in landmarks and len(landmarks['mouth']) >= 8: # Precisa de pontos suficientes
                  # Assume 'mouth' contains points 48-67 from dlib 68 point model. We need 60-67.
                  # face_recognition returns dict keys like 'left_eye', 'right_eye', 'nose_bridge', 'mouth', etc.
                  # Let's assume the 'mouth' key maps directly to the 20 points (48-67).
                  # We need points 60, 61, 62, 63, 64, 65, 66, 67 (relative indices 12 to 19 if 'mouth' list starts at 48)
                  
                  # Recalculate MAR using points from 'mouth' list directly
                  # Indices within the 'mouth' list (assuming it corresponds to 48-67)
                  # Corners: 48 (idx 0), 54 (idx 6)
                  # Inner Upper Lip: 50(2), 51(3), 52(4)
                  # Inner Lower Lip: 58(10), 57(9), 56(8)
                  
                  mouth_pts = landmarks['mouth']
                  
                  # Check if we have enough points (at least 12 for these indices)
                  if len(mouth_pts) >= 13: # Points 48 to 60 -> indices 0 to 12 needed roughly
                    p51 = mouth_pts[3]  # point 51
                    p57 = mouth_pts[9]  # point 57
                    p48 = mouth_pts[0]  # point 48
                    p54 = mouth_pts[6]  # point 54

                    A = dist.euclidean(p51, p57) # Vertical distance mid lips
                    C = dist.euclidean(p48, p54) # Horizontal distance corners
                    
                    if C > 0:
                        current_mar = A / C
                        mar_values.append(current_mar)

                        # Detecção do Sorriso: MAR acima do limiar por frames consecutivos?
                        if current_mar > MAR_THRESHOLD:
                            # Contar quantos dos últimos frames também estavam acima
                            smile_counter = 0
                            for mar_val in reversed(mar_values[-MAR_CONSECUTIVE_FRAMES:]):
                                if mar_val > MAR_THRESHOLD:
                                    smile_counter += 1
                                else:
                                    break # Não são consecutivos
                            
                            if smile_counter >= MAR_CONSECUTIVE_FRAMES:
                                smile_detected = True
                                # Captura o encoding deste frame para comparação futura
                                face_encodings = face_recognition.face_encodings(rgb_frame, [face_locations[0]]) # Usa a location já encontrada
                                if face_encodings:
                                    best_liveness_encoding = face_encodings[0]
                                    best_liveness_frame = frame # Guarda o frame BGR
                                break # Sorriso detectado, pode parar de processar o vídeo
                        
                    else:
                      mar_values.append(0.0)
                  else:
                    mar_values.append(0.0) # Not enough points
                else:
                    mar_values.append(0.0) # Boca não detectada corretamente
            else:
                 mar_values.append(0.0) # Rosto não detectado no frame

        video_capture.release() # Fecha o ficheiro de vídeo

        # --- PASSO 3: Verificar Resultado do Liveness ---
        if not smile_detected or best_liveness_encoding is None:
            return LivenessCheckResponse(
                liveness_check_passed=False,
                action_detected=None,
                face_match_passed=None,
                detail="Prova de vida falhou: Sorriso não detectado ou rosto não capturado claramente no vídeo."
            )
            
        # --- PASSO 4: Carregar e Processar Imagem do Documento ---
        doc_contents = await doc_image.read()
        doc_pil_image = Image.open(io.BytesIO(doc_contents))
        doc_np_image = np.array(doc_pil_image)
        
        doc_face_encodings = face_recognition.face_encodings(doc_np_image)
        if not doc_face_encodings:
            return LivenessCheckResponse(
                liveness_check_passed=True, # Liveness passou, mas...
                action_detected="smile",
                face_match_passed=False,
                detail="Prova de vida OK, mas nenhum rosto encontrado na foto do documento para comparação."
            )
        doc_encoding = doc_face_encodings[0] # Pega o primeiro rosto do doc

        # --- PASSO 5: Comparar Rosto do Vídeo com Rosto do Documento ---
        matches = face_recognition.compare_faces([doc_encoding], best_liveness_encoding, tolerance=0.6)
        is_match = bool(matches[0]) 
        face_distance = face_recognition.face_distance([doc_encoding], best_liveness_encoding)[0]
        confidence = max(0.0, (1.0 - face_distance)) * 100

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

    except Exception as e:
        # Captura qualquer outro erro
        return LivenessCheckResponse(
            liveness_check_passed=False,
            action_detected=None,
            face_match_passed=None,
            detail=f"Erro durante o processamento de liveness: {str(e)}"
        )
    finally:
        # --- Limpeza: Apagar o ficheiro de vídeo temporário ---
        if video_path and os.path.exists(video_path):
            os.remove(video_path)


# --- ======================================================= ---
# --- SERVIÇO 3: ENRIQUECIMENTO DE CPF (KYC)                   ---
# --- ======================================================= ---
# (Sem alterações no Serviço 3)
@app.get("/kyc/enrich-cpf/", response_model=KYCResponse) 
async def enrich_cpf(cpf: str = Query(..., description="CPF (com ou sem pontos/traços)", min_length=11, max_length=14)):
    cpf_limpo = re.sub(r'\D', '', cpf); # ... (restante do código KYC igual) ...
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
# (Sem alterações no Serviço 4)
@app.get("/kyc/enrich-cnpj/", response_model=KYBResponse)
async def enrich_cnpj(cnpj: str = Query(..., description="CNPJ (com ou sem pontos/traços/barras)", min_length=14, max_length=18)):
    cnpj_limpo = re.sub(r'\D', '', cnpj); # ... (restante do código KYB igual) ...
    if len(cnpj_limpo) != 14: raise HTTPException(status_code=400, detail="CNPJ inválido.")
    url = f"https://brasilapi.com.br/api/cnpj/v1/{cnpj_limpo}"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10.0); response.raise_for_status(); data = response.json()
            # Ajuste no telefone para evitar "None"
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
# (Sem alterações no Serviço 5)
@app.get("/kyc/background-check/", response_model=BackgroundCheckResponse)
async def background_check(cpf: str = Query(..., description="CPF a verificar (com ou sem pontos/traços)", min_length=11, max_length=14)):
    cpf_limpo = re.sub(r'\D', '', cpf); # ... (restante do código Background Check igual) ...
    if len(cpf_limpo) != 11: raise HTTPException(status_code=400, detail="CPF inválido.")
    risk_profile = MOCK_RISK_LIST.get(cpf_limpo)
    if risk_profile:
        return BackgroundCheckResponse(cpf=cpf, is_pep=risk_profile.get("is_pep", False), on_sanctions_list=risk_profile.get("on_sanctions_list", False), risk_level=risk_profile.get("risk_level", "ALTO"), details=risk_profile.get("details", "Risco identificado."))
    else:
        return BackgroundCheckResponse(cpf=cpf, is_pep=False, on_sanctions_list=False, risk_level="BAIXO", details="Nenhum risco identificado.")