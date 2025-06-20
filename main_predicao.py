import cv2
import mediapipe as mp
import numpy as np
import joblib
from utils import calcular_angulo

# Carrega modelo treinado
modelo = joblib.load('modelo_treinado.pkl')

# Inicializa MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Controle de repetição com estado
estado_repeticao = 'esperando_descida'
contador_reps = 0
limite_descida = 0.6
limite_subida = 0.4

# Inicia webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(frame_rgb)

    classe = "Nenhum usuario detectado"
    confianca = 0.0

    if result.pose_landmarks:
        lm = result.pose_landmarks.landmark

        # Pontos chave
        ombro_d = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
                   lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
        cotovelo_d = [lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * w,
                      lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * h]
        punho_d = [lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w,
                   lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h]

        ombro_e = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w,
                   lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h]
        cotovelo_e = [lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * w,
                      lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * h]
        punho_e = [lm[mp_pose.PoseLandmark.LEFT_WRIST.value].x * w,
                   lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y * h]

        # Cálculos
        angulo_d = calcular_angulo(ombro_d, cotovelo_d, punho_d)
        angulo_e = calcular_angulo(ombro_e, cotovelo_e, punho_e)
        y_punho_d = lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
        y_punho_e = lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y
        y_ombro_d = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
        y_ombro_e = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y

        entrada = [[angulo_d, y_punho_d, y_ombro_d,
                    angulo_e, y_punho_e, y_ombro_e]]

        prob = modelo.predict_proba(entrada)[0]
        classe = modelo.classes_[np.argmax(prob)]
        confianca = np.max(prob)

        # Contagem de repetições usando média dos punhos
        media_punho_y = (y_punho_d + y_punho_e) / 2
        if estado_repeticao == 'esperando_descida' and media_punho_y > limite_descida:
            estado_repeticao = 'descendo'
        elif estado_repeticao == 'descendo' and media_punho_y < limite_subida:
            contador_reps += 1
            estado_repeticao = 'esperando_descida'

        # Desenhar landmarks
        mp_drawing.draw_landmarks(
            frame,
            result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
        )

    # Feedback visual
    cor = (0, 255, 0) if classe == 'correto' else (0, 0, 255)
    texto_classe = f"Movimento: {classe.upper()} ({confianca*100:.1f}%)"
    cv2.putText(frame, texto_classe, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, cor, 2)
    cv2.putText(frame, f"Repeticoes: {contador_reps}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow("Avaliacao em tempo real", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
