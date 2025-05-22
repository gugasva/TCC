import cv2
import mediapipe as mp
import csv
import os
import numpy as np
from utils import calcular_angulo

# Caminho base dos vídeos
BASE_DIR = 'videos'  # deve conter 'correto/' e 'incorreto/'

# Saída do CSV
saida_csv = 'dados_treinamento.csv'
if not os.path.exists(saida_csv):
    with open(saida_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'angulo_braco_direito', 'y_punho_direito', 'y_ombro_direito',
            'angulo_braco_esquerdo', 'y_punho_esquerdo', 'y_ombro_esquerdo',
            'rotulo'])

# Inicialização do MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Varre os diretórios de vídeos
for rotulo in os.listdir(BASE_DIR):
    pasta = os.path.join(BASE_DIR, rotulo)
    if not os.path.isdir(pasta):
        continue

    for arquivo in os.listdir(pasta):
        if not arquivo.lower().endswith(('.mp4', '.avi', '.mov')):
            continue

        caminho_video = os.path.join(pasta, arquivo)
        print(f'Processando: {caminho_video} | Rótulo: {rotulo}')

        cap = cv2.VideoCapture(caminho_video)
        if not cap.isOpened():
            print(f'[ERRO] Não foi possível abrir o vídeo: {caminho_video}')
            continue

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(frame_rgb)

            if result.pose_landmarks:
                landmarks = result.pose_landmarks.landmark
                h, w, _ = frame.shape

                # Braço direito
                ombro_d = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
                           landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
                cotovelo_d = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * w,
                              landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * h]
                punho_d = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w,
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h]
                y_punho_d = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
                y_ombro_d = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
                angulo_d = calcular_angulo(ombro_d, cotovelo_d, punho_d)

                # Braço esquerdo
                ombro_e = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w,
                           landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h]
                cotovelo_e = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * w,
                              landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * h]
                punho_e = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * w,
                           landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * h]
                y_punho_e = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y
                y_ombro_e = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
                angulo_e = calcular_angulo(ombro_e, cotovelo_e, punho_e)

                # Salvar no CSV
                with open(saida_csv, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        angulo_d, y_punho_d, y_ombro_d,
                        angulo_e, y_punho_e, y_ombro_e,
                        rotulo
                    ])

        cap.release()

cv2.destroyAllWindows()
print("✅ Coleta concluída.")
