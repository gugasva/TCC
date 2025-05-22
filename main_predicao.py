import cv2
import mediapipe as mp
import joblib
from utils import calcular_angulo
import numpy as np

modelo = joblib.load('modelo_treinado.pkl')

cap = cv2.VideoCapture(0)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

while True:
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

        entrada = [[angulo_d, y_punho_d, y_ombro_d, angulo_e, y_punho_e, y_ombro_e]]
        predicao = modelo.predict(entrada)[0]

        cor = (0, 255, 0) if predicao == 'correto' else (0, 0, 255)
        cv2.putText(frame, f'Predicao: {predicao}', (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, cor, 2)
        cv2.putText(frame, f'Ang D: {int(angulo_d)} Y D: {round(y_punho_d, 2)} Omb D: {round(y_ombro_d, 2)}',
                    (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f'Ang E: {int(angulo_e)} Y E: {round(y_punho_e, 2)} Omb E: {round(y_ombro_e, 2)}',
                    (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow("IA - ambos os braços com y_ombro", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
