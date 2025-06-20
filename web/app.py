from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import mediapipe as mp
import joblib
from utils import calcular_angulo
from PIL import Image
import io

app = Flask(__name__)

# Carrega modelo treinado
modelo = joblib.load('modelo_treinado.pkl')

# Inicializa MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/teste')
def teste():
    return render_template('teste.html')

@app.route('/prever', methods=['POST'])
def prever():
    if 'frame' not in request.files:
        return jsonify({'erro': 'Nenhum frame enviado.'}), 400

    img = Image.open(request.files['frame']).convert('RGB')
    frame = np.array(img)

    h, w, _ = frame.shape
    result = pose.process(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    if not result.pose_landmarks:
        return jsonify({'classe': 'desconhecido', 'confianca': 0.0})

    lm = result.pose_landmarks.landmark
    h, w, _ = frame.shape

# Direito
    ombro_d = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
           lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
    cotovelo_d = [lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * w,
              lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * h]
    punho_d = [lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w,
           lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h]
    angulo_d = calcular_angulo(ombro_d, cotovelo_d, punho_d)
    y_punho_d = lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
    y_ombro_d = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y

# Esquerdo
    ombro_e = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w,
           lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h]
    cotovelo_e = [lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * w,
              lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * h]
    punho_e = [lm[mp_pose.PoseLandmark.LEFT_WRIST.value].x * w,
           lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y * h]
    angulo_e = calcular_angulo(ombro_e, cotovelo_e, punho_e)
    y_punho_e = lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y
    y_ombro_e = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y

# Entrada com 6 features
    entrada = [[
        angulo_d, y_punho_d, y_ombro_d,
        angulo_e, y_punho_e, y_ombro_e
]]
    prob = modelo.predict_proba(entrada)[0]
    classe = modelo.classes_[np.argmax(prob)]
    confianca = float(np.max(prob))

    return jsonify({
        'classe': classe,
        'confianca': confianca
    })

if __name__ == '__main__':
    app.run(debug=True)