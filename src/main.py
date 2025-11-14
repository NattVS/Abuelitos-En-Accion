import cv2
import mediapipe as mp
import joblib
import numpy as np
import warnings
from collections import deque

warnings.filterwarnings('ignore')

# --- 1. Cargar Artefactos del Modelo ---
try:
    # Usamos Random Forest por ser generalmente más rápido para inferencia
    pipeline = joblib.load('./models/best_random_forest_model.joblib')
    label_encoder = joblib.load('./models/label_encoder.joblib')
    print("Modelos (Temporal + Velocidad) cargados correctamente.")
except FileNotFoundError:
    print("Error: No se encontraron los archivos .joblib en ./models/")
    print("Asegúrate de ejecutar 'model_training.py' primero.")
    exit()

# --- 2. Funciones de Ingeniería de Características (Solo los 8 ángulos) ---
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    if cosine_angle > 1.0: cosine_angle = 1.0
    if cosine_angle < -1.0: cosine_angle = -1.0
    return np.degrees(np.arccos(cosine_angle))

def extract_angles(results):
    try:
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
        
        left_hip, right_hip = landmarks[23], landmarks[24]
        hip_center = (left_hip + right_hip) / 2.0
        normalized_landmarks = landmarks - hip_center
        
        left_shoulder, right_shoulder = normalized_landmarks[11], landmarks[12] # Bug sutil corregido
        shoulder_distance = np.linalg.norm(left_shoulder - right_shoulder)
        
        if shoulder_distance == 0: return None
        normalized_landmarks = normalized_landmarks / shoulder_distance
        
        l_sho, r_sho = normalized_landmarks[11], normalized_landmarks[12]
        l_elb, r_elb = normalized_landmarks[13], normalized_landmarks[14]
        l_wri, r_wri = normalized_landmarks[15], normalized_landmarks[16]
        l_hip, r_hip = normalized_landmarks[23], normalized_landmarks[24]
        l_knee, r_knee = normalized_landmarks[25], normalized_landmarks[26]
        l_ank, r_ank = normalized_landmarks[27], normalized_landmarks[28]

        angles = [
            calculate_angle(l_sho, l_elb, l_wri), calculate_angle(r_sho, r_elb, r_wri),
            calculate_angle(l_elb, l_sho, l_hip), calculate_angle(r_elb, r_sho, r_hip),
            calculate_angle(l_sho, l_hip, l_knee), calculate_angle(r_sho, r_hip, r_knee),
            calculate_angle(l_hip, l_knee, l_ank), calculate_angle(r_hip, r_knee, r_ank)
        ]
        return np.array(angles)
    except:
        return None

# --- 3. Inicializar MediaPipe, Cámara y Deque ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara web.")
    exit()

window_size = 15
angle_deque = deque(maxlen=window_size)

# --- NUEVA LÓGICA: Umbral de Quietud ---
# Este es el valor clave para sintonizar.
# Es la suma de la desviación estándar de los 8 ángulos.
# Si la suma es menor que esto, asumimos que está "quieto".
STILLNESS_THRESHOLD = 40.0 

print("Cámara iniciada. Presiona 'q' para salir.")
current_action = "---"
current_confidence = 0.0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        angles = extract_angles(results)
        
        if angles is not None:
            angle_deque.append(angles)
            
            if len(angle_deque) == window_size:
                try:
                    # --- 4. Calcular Características y aplicar "Gate de Quietud" ---
                    window_data = np.array(angle_deque) # (15, 8)
                    
                    # 4.1. Calcular STD de los ángulos PRIMERO
                    angle_std = np.std(window_data, axis=0)
                    
                    # 4.2. Calcular la magnitud total del movimiento
                    movement_magnitude = np.sum(angle_std)
                    
                    # 4.3. Decidir: ¿Está quieto o en movimiento?
                    if movement_magnitude < STILLNESS_THRESHOLD:
                        current_action = "quieto" # Clase "default"
                        current_confidence = 1.0
                    
                    else:
                        # 4.4. Hay movimiento, calcular el resto de características
                        angle_mean = np.mean(window_data, axis=0)
                        angle_min = np.min(window_data, axis=0)
                        angle_max = np.max(window_data, axis=0)
                        
                        # Velocidad (diferencia) dentro de la ventana
                        window_vel = np.diff(window_data, axis=0) # (14, 8)
                        vel_mean = np.mean(window_vel, axis=0)
                        vel_std = np.std(window_vel, axis=0)
                        
                        # 4.5. Concatenar (32 + 16 = 48 características)
                        features = np.concatenate([
                            angle_mean, angle_std, angle_min, angle_max,
                            vel_mean, vel_std
                        ])
                        
                        # --- 5. Predicción del Modelo (Experto en Movimiento) ---
                        features_2d = features.reshape(1, -1)
                        prediction = pipeline.predict(features_2d)
                        proba = pipeline.predict_proba(features_2d)
                        
                        current_action = label_encoder.inverse_transform(prediction)[0]
                        current_confidence = np.max(proba)

                except Exception as e:
                    # print(f"Error: {e}")
                    current_action = "Error"
                    current_confidence = 0.0
    else:
        # Si no se detectan landmarks, reseteamos
        angle_deque.clear()
        current_action = "---"
        current_confidence = 0.0

    # --- 6. Visualización ---
    cv2.rectangle(frame, (0, 0), (400, 70), (20, 20, 20), -1)
    cv2.putText(frame, 'ACCION', (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, current_action, (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, 'CONFIANZA', (250, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f'{current_confidence * 100:.1f}%', (250, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Reconocimiento de Movimiento', frame)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# --- 7. Limpieza ---
cap.release()
cv2.destroyAllWindows()
pose.close()
print("Script finalizado.")