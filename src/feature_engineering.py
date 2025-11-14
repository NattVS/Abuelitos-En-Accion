import pandas as pd
import numpy as np
from tqdm import tqdm

pd.options.mode.chained_assignment = None

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    if cosine_angle > 1.0: cosine_angle = 1.0
    if cosine_angle < -1.0: cosine_angle = -1.0
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def extract_features(row):
    # Esta función ahora solo extrae los 8 ángulos
    landmarks = []
    for i in range(33):
        x = row[f'landmark_{i}_x']
        y = row[f'landmark_{i}_y']
        z = row[f'landmark_{i}_z']
        landmarks.append([x, y, z])
    
    landmarks = np.array(landmarks)
    
    try:
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        hip_center = (left_hip + right_hip) / 2.0
        normalized_landmarks = landmarks - hip_center
        
        left_shoulder = normalized_landmarks[11]
        right_shoulder = normalized_landmarks[12]
        shoulder_distance = np.linalg.norm(left_shoulder - right_shoulder)
        
        if shoulder_distance == 0:
            return None
            
        normalized_landmarks = normalized_landmarks / shoulder_distance
        
        features = {}
        l_sho, r_sho = normalized_landmarks[11], normalized_landmarks[12]
        l_elb, r_elb = normalized_landmarks[13], normalized_landmarks[14]
        l_wri, r_wri = normalized_landmarks[15], normalized_landmarks[16]
        l_hip, r_hip = normalized_landmarks[23], normalized_landmarks[24]
        l_knee, r_knee = normalized_landmarks[25], normalized_landmarks[26]
        l_ank, r_ank = normalized_landmarks[27], normalized_landmarks[28]

        features['angle_l_elbow'] = calculate_angle(l_sho, l_elb, l_wri)
        features['angle_r_elbow'] = calculate_angle(r_sho, r_elb, r_wri)
        features['angle_l_shoulder'] = calculate_angle(l_elb, l_sho, l_hip)
        features['angle_r_shoulder'] = calculate_angle(r_elb, r_sho, r_hip)
        features['angle_l_hip'] = calculate_angle(l_sho, l_hip, l_knee)
        features['angle_r_hip'] = calculate_angle(r_sho, r_hip, r_knee)
        features['angle_l_knee'] = calculate_angle(l_hip, l_knee, l_ank)
        features['angle_r_knee'] = calculate_angle(r_hip, r_knee, r_ank)
            
        return features
    except Exception as e:
        return None

def create_temporal_features(df_group, window_size=15):
    """
    Toma un grupo de frames (un video) y crea características temporales
    usando una ventana deslizante.
    Calcula 4 estadísticas para ángulos y 2 para velocidad de ángulos.
    """
    # 1. Calcular los 8 ángulos para cada frame en el grupo
    tqdm.pandas(desc=f"Calculando ángulos para {df_group['video_filename'].iloc[0]}", leave=False)
    feature_list = df_group.progress_apply(extract_features, axis=1)
    df_angles = pd.DataFrame(feature_list.tolist())
    
    # Manejar frames donde extract_features falló (retornó None)
    df_angles.dropna(inplace=True)
    if df_angles.empty:
        return pd.DataFrame() # Retornar DF vacío si no hay datos buenos

    # 2. Calcular estadísticas de ÁNGULOS (Posición)
    angle_names = df_angles.columns
    rolling_stats = df_angles.rolling(window=window_size, min_periods=window_size)
    
    df_mean = rolling_stats.mean().add_suffix('_mean')
    df_std = rolling_stats.std().add_suffix('_std')
    df_min = rolling_stats.min().add_suffix('_min')
    df_max = rolling_stats.max().add_suffix('_max')
    
    # 3. Calcular estadísticas de VELOCIDAD (Movimiento)
    df_vel = df_angles.diff().fillna(0)
    rolling_vel_stats = df_vel.rolling(window=window_size, min_periods=window_size)
    
    df_vel_mean = rolling_vel_stats.mean().add_suffix('_vel_mean')
    df_vel_std = rolling_vel_stats.std().add_suffix('_vel_std')
    
    # 4. Concatenar todas las estadísticas (48 características)
    df_temporal = pd.concat([df_mean, df_std, df_min, df_max, df_vel_mean, df_vel_std], axis=1)
    
    # 5. Añadir de nuevo la información de 'action' y 'video_filename'
    # Lógica de join ROBUSTA:
    df_temporal.dropna(inplace=True) # Elimina NaNs de las ventanas iniciales
    df_temporal['action'] = df_group['action'].iloc[0]
    df_temporal['video_filename'] = df_group['video_filename'].iloc[0]
    df_temporal['frame_idx'] = df_temporal.index
    
    return df_temporal

def main():
    input_path = './data/processed/datosmediapipe.csv'
    output_path = './data/processed/model_features.csv'
    
    print(f"Cargando datos desde {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {input_path}")
        return

    print("Datos cargados. Agrupando por video...")
    
    video_groups = df.groupby('video_filename')
    all_temporal_features = []
    
    for group_name, df_group in tqdm(video_groups, desc="Procesando videos"):
        temporal_features = create_temporal_features(df_group, window_size=15)
        all_temporal_features.append(temporal_features)
    
    final_df = pd.concat(all_temporal_features, ignore_index=True)
    
    final_df.to_csv(output_path, index=False)
    
    print(f"\n¡Ingeniería de características (temporal + velocidad) completa!")
    print(f"Dataset para el modelo guardado en: {output_path}")
    print(f"Forma del dataset de características: {final_df.shape}")

if __name__ == "__main__":
    main()