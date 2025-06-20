import cv2
import mediapipe as mp
import os
from utils import calcular_angulo
from db_conexao import salvar_dado_treinamento

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Caminhos dos vídeos organizados por rótulo
ROTULOS_VIDEOS = {
    r'C:\Users\gusta\OneDrive\Desktop\tcc\pratico\videos\correto': 'correto',
    r'C:\Users\gusta\OneDrive\Desktop\tcc\pratico\videos\incorreto': 'incorreto'
}

def processar_video(caminho_video, rotulo):
    cap = cv2.VideoCapture(caminho_video)
    print(f"Processando: {caminho_video} | Rótulo: {rotulo}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(frame_rgb)

        if result.pose_landmarks:
            lm = result.pose_landmarks.landmark

            # Direito
            ombro_d = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
                       lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
            cotovelo_d = [lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * w,
                          lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * h]
            punho_d = [lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w,
                       lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h]

            # Esquerdo
            ombro_e = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w,
                       lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h]
            cotovelo_e = [lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * w,
                          lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * h]
            punho_e = [lm[mp_pose.PoseLandmark.LEFT_WRIST.value].x * w,
                       lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y * h]

            angulo_d = calcular_angulo(ombro_d, cotovelo_d, punho_d)
            angulo_e = calcular_angulo(ombro_e, cotovelo_e, punho_e)

            y_punho_d = lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
            y_punho_e = lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y
            y_ombro_d = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
            y_ombro_e = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y

            dados = (
                angulo_d, y_punho_d, y_ombro_d,
                angulo_e, y_punho_e, y_ombro_e,
                rotulo
            )

            salvar_dado_treinamento(dados)

    cap.release()

# Processar todos os vídeos das pastas
for pasta, rotulo in ROTULOS_VIDEOS.items():
    for nome_arquivo in os.listdir(pasta):
        if nome_arquivo.endswith(('.mp4', '.avi', '.mov')):
            caminho = os.path.join(pasta, nome_arquivo)
            processar_video(caminho, rotulo)

print("✅ Coleta finalizada e dados salvos no banco.")
