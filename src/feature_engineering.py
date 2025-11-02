import pandas as pd
import numpy as np
from tqdm import tqdm

# Desactivar la advertencia SettingWithCopyWarning
pd.options.mode.chained_assignment = None

def calculate_angle(a, b, c):
    """Calcula el ángulo entre tres puntos (en el punto b)."""
    a = np.array(a) # Primer punto
    b = np.array(b) # Punto medio (vértice del ángulo)
    c = np.array(c) # Tercer punto
    
    # Vectores
    ba = a - b
    bc = c - b
    
    # Producto punto
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    
    # Manejar casos extremos de arcoseno
    if cosine_angle > 1.0:
        cosine_angle = 1.0
    if cosine_angle < -1.0:
        cosine_angle = -1.0
        
    # Calcular el ángulo en grados
    angle = np.arccos(cosine_angle)
    
    return np.degrees(angle)

def extract_features(row):
    """
    Extrae un vector de características normalizado de una sola fila (frame)
    que contiene las 33 coordenadas de landmarks.
    """
    landmarks = []
    for i in range(33):
        x = row[f'landmark_{i}_x']
        y = row[f'landmark_{i}_y']
        z = row[f'landmark_{i}_z']
        landmarks.append([x, y, z])
    
    landmarks = np.array(landmarks)
    
    # --- 1. Normalización ---
    # Centrar la pose usando el punto medio de la cadera
    left_hip = landmarks[23]
    right_hip = landmarks[24]
    hip_center = (left_hip + right_hip) / 2.0
    
    normalized_landmarks = landmarks - hip_center
    
    # Escalar la pose usando la distancia entre los hombros
    left_shoulder = normalized_landmarks[11]
    right_shoulder = normalized_landmarks[12]
    shoulder_distance = np.linalg.norm(left_shoulder - right_shoulder)
    
    # Evitar división por cero si los landmarks no se detectaron
    if shoulder_distance == 0:
        return None # Retorna None para filtrar este frame
        
    normalized_landmarks = normalized_landmarks / shoulder_distance
    
    # --- 2. Ingeniería de Características (Ángulos) ---
    features = {}
    
    # Índices de landmarks de MediaPipe
    # Hombros
    l_sho = normalized_landmarks[11]
    r_sho = normalized_landmarks[12]
    # Codos
    l_elb = normalized_landmarks[13]
    r_elb = normalized_landmarks[14]
    # Muñecas
    l_wri = normalized_landmarks[15]
    r_wri = normalized_landmarks[16]
    # Caderas
    l_hip = normalized_landmarks[23]
    r_hip = normalized_landmarks[24]
    # Rodillas
    l_knee = normalized_landmarks[25]
    r_knee = normalized_landmarks[26]
    # Tobillos
    l_ank = normalized_landmarks[27]
    r_ank = normalized_landmarks[28]

    # Calcular ángulos
    features['angle_l_elbow'] = calculate_angle(l_sho, l_elb, l_wri)
    features['angle_r_elbow'] = calculate_angle(r_sho, r_elb, r_wri)
    
    features['angle_l_shoulder'] = calculate_angle(l_elb, l_sho, l_hip)
    features['angle_r_shoulder'] = calculate_angle(r_elb, r_sho, r_hip)
    
    features['angle_l_hip'] = calculate_angle(l_sho, l_hip, l_knee)
    features['angle_r_hip'] = calculate_angle(r_sho, r_hip, r_knee)
    
    features['angle_l_knee'] = calculate_angle(l_hip, l_knee, l_ank)
    features['angle_r_knee'] = calculate_angle(r_hip, r_knee, r_ank)

    # --- 3. Aplanar características y añadir coordenadas normalizadas ---
    # Aplanar los landmarks normalizados para usarlos también como características
    flat_landmarks = normalized_landmarks.flatten()
    for i in range(len(flat_landmarks)):
        features[f'norm_lm_{i}'] = flat_landmarks[i]
        
    return features

def main():
    # Cargar los datos extraídos por process.py
    input_path = './data/processed/datosmediapipe.csv'
    output_path = './data/processed/model_features.csv'
    
    print(f"Cargando datos desde {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {input_path}")
        print("Asegúrate de ejecutar process.py primero.")
        return

    print("Datos cargados. Iniciando ingeniería de características...")
    
    # Aplicar la extracción de características a cada fila
    # tqdm es una barra de progreso
    tqdm.pandas(desc="Procesando frames")
    feature_list = df.progress_apply(extract_features, axis=1)
    
    # Combinar las características con la información original
    df_features = pd.DataFrame(feature_list.tolist())
    
    # Concatenar las etiquetas (action) y la información de video
    final_df = pd.concat([df[['action', 'video_filename', 'frame_idx']], df_features], axis=1)
    
    # Eliminar filas donde la extracción de características falló (p.ej., división por cero)
    final_df.dropna(inplace=True)
    
    # Guardar el dataset listo para el modelo
    final_df.to_csv(output_path, index=False)
    
    print(f"\n¡Ingeniería de características completa!")
    print(f"Dataset para el modelo guardado en: {output_path}")
    print(f"Forma del dataset de características: {final_df.shape}")

if __name__ == "__main__":
    main()
