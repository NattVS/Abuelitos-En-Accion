
import argparse
import os
import sys
import math
import glob
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import mediapipe as mp
mp_pose = mp.solutions.pose

# MediaPipe
try:
    import mediapipe as mp
except ImportError as e:
    print("ERROR: No se encontró mediapipe. Instala con: pip install mediapipe", file=sys.stderr)
    raise

mp_pose = mp.solutions.pose

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}

def is_video_file(p: Path) -> bool:
    return p.suffix.lower() in VIDEO_EXTS

def compute_luminance_stats(gray: np.ndarray) -> Tuple[float, float]:
    # gray: imagen 8-bit [0..255]
    mean = float(gray.mean())
    std = float(gray.std(ddof=0))
    return mean, std

def compute_motion_mad(prev_gray: Optional[np.ndarray], gray: np.ndarray) -> Optional[float]:
    """
    Movimiento simple: Mean Absolute Difference (MAD) entre este frame y el anterior (en escala 0-255).
    Retorna None para el primer frame (no hay anterior).
    """
    if prev_gray is None:
        return None
    # Igualar tamaño si hicimos algún resize en pipeline (aquí no lo hacemos)
    if prev_gray.shape != gray.shape:
        try:
            prev_gray = cv2.resize(prev_gray, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_AREA)
        except Exception:
            return None
    mad = float(np.mean(np.abs(gray.astype(np.int16) - prev_gray.astype(np.int16))))
    return mad

def extract_pose_landmarks(pose_results) -> Tuple[Optional[List[Tuple[int,float,float,float,float]]],
                                                  Optional[List[Tuple[int,float,float,float]]]]:
    """
    Devuelve:
      - landmarks_img: lista de (id, x, y, z, visibility) en coords normalizadas de imagen [0..1], z en m.relativos
      - landmarks_world: lista de (id, X, Y, Z) en coordenadas 'world' (metros) si existen
    """
    landmarks_img = None
    landmarks_world = None

    if pose_results.pose_landmarks:
        lms = pose_results.pose_landmarks.landmark
        landmarks_img = [(idx, lm.x, lm.y, lm.z, getattr(lm, "visibility", np.nan)) for idx, lm in enumerate(lms)]

    if hasattr(pose_results, "pose_world_landmarks") and pose_results.pose_world_landmarks:
        wlms = pose_results.pose_world_landmarks.landmark
        landmarks_world = [(idx, lm.x, lm.y, lm.z) for idx, lm in enumerate(wlms)]

    return landmarks_img, landmarks_world

def process_video(video_path: Path,
                  action: str,
                  pose: "mp.solutions.pose.Pose",
                  stride: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Procesa un video y genera:
      - df_pose: (WIDE FORMAT) una fila por frame, con columnas para cada coordenada de landmark
      - df_frame: filas por frame con métricas de análisis (luminancia, movimiento, etc.)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration_sec = frame_count / fps if fps > 0 else np.nan

    pose_rows = []
    frame_rows = []
    prev_gray = None
    frame_idx = 0
    pbar = tqdm(total=frame_count, desc=f"Procesando [{action}] {video_path.name}", unit="f")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if (frame_idx % stride) != 0:
            frame_idx += 1
            pbar.update(1)
            continue
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = pose.process(rgb)
        landmarks_img, landmarks_world = extract_pose_landmarks(results)
        ts = frame_idx / fps if fps > 0 else np.nan
        lum_mean, lum_std = compute_luminance_stats(gray)
        motion_mad = compute_motion_mad(prev_gray, gray)

        frame_rows.append({
            "action": action, "video_filename": video_path.name, "frame_idx": frame_idx,
            "timestamp_sec": ts, "width": width, "height": height, "fps": fps,
            "total_frames": frame_count, "duration_sec": duration_sec, "luminance_mean": lum_mean,
            "luminance_std": lum_std, "motion_mad": motion_mad,
        })

        if landmarks_img is not None:
            # 1. Start with a base dictionary for the frame
            frame_pose_data = {
                "action": action,
                "video_filename": video_path.name,
                "frame_idx": frame_idx,
                "timestamp_sec": ts,
            }

            # 2. Add each landmark's coordinates as new columns
            for (lid, x, y, z, vis) in landmarks_img:
                frame_pose_data[f'landmark_{lid}_x'] = x
                frame_pose_data[f'landmark_{lid}_y'] = y
                frame_pose_data[f'landmark_{lid}_z'] = z
                frame_pose_data[f'landmark_{lid}_vis'] = vis
            
            pose_rows.append(frame_pose_data)

        prev_gray = gray
        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    
    # --- DataFrame creation (no change needed here) ---
    df_pose = pd.DataFrame(pose_rows)
    df_frame = pd.DataFrame(frame_rows)
    return df_pose, df_frame

def main():
    parser = argparse.ArgumentParser(description="Analiza videos: frames, iluminación, duración y 33 landmarks de MediaPipe Pose.")
    parser.add_argument("--stride", type=int, default=1, help="Procesar cada N frames (por defecto 1).")
    parser.add_argument("--static_image_mode", action="store_true", help="MediaPipe en modo estático (más robusto por frame, más lento).")
    parser.add_argument("--min_detection_confidence", type=float, default=0.5, help="Confianza mínima de detección (pose).")
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5, help="Confianza mínima de tracking (pose).")
    args = parser.parse_args()

    base_dir = Path("./data/raw/videos") # Use a relative path from your project root for clarity
    out_dir = Path("./data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    action_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    if not action_dirs:
        print(f"No se encontraron carpetas de acciones en: {base_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Encontradas {len(action_dirs)} acciones: {[d.name for d in action_dirs]}")

    all_pose = []
    all_frames = []

    with mp_pose.Pose(
        static_image_mode=args.static_image_mode,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        smooth_landmarks=True,
    ) as pose:

        # Loop through each action directory
        for action_dir in action_dirs:
            action_name = action_dir.name
            
            # Now find all video files inside this specific action directory
            # The "**/*" pattern finds all files recursively
            videos_in_action = [p for p in action_dir.glob("**/*") if is_video_file(p)]

            for vp in videos_in_action:
                try:
                    # Pass the action name to the processing function
                    df_pose, df_frame = process_video(vp, action_name, pose, stride=args.stride)
                    all_pose.append(df_pose)
                    all_frames.append(df_frame)
                except Exception as ex:
                    print(f"[ADVERTENCIA] Falló {vp.name}: {ex}", file=sys.stderr)
                    # You might want to log the error to the dataframe as you did before


    datosmediapipe_csv = out_dir / "datosmediapipe.csv"
    datos_analisis_csv = out_dir / "datos_analisis.csv"
    if all_pose:
        pd.concat(all_pose, ignore_index=True).to_csv(datosmediapipe_csv, index=False)
    else:

        headers = ["action", "video_filename", "frame_idx", "timestamp_sec"]
        for i in range(33): # 33 landmarks
            headers.extend([f'landmark_{i}_x', f'landmark_{i}_y', f'landmark_{i}_z', f'landmark_{i}_vis'])
        pd.DataFrame(columns=headers).to_csv(datosmediapipe_csv, index=False)
    
    if all_frames:
        pd.concat(all_frames, ignore_index=True).to_csv(datos_analisis_csv, index=False)
    else:
        pd.DataFrame(columns=[
            "action","video_filename","frame_idx","timestamp_sec", "width","height",
            "fps","total_frames","duration_sec", "luminance_mean","luminance_std","motion_mad"
        ]).to_csv(datos_analisis_csv, index=False)
    
    print(f"\n¡Proceso completado!")
    print(f"Guardado: {datosmediapipe_csv}")
    print(f"Guardado: {datos_analisis_csv}")

if __name__ == "__main__":
    main()
