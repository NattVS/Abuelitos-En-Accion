import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import warnings
import joblib
import os

warnings.filterwarnings('ignore')

# --- Funciones de Ploteo ---
def plot_confusion_matrix(y_true, y_pred, classes, model_name):
    os.makedirs('./reports', exist_ok=True)
    filename = f'./reports/cm_{model_name.lower().replace(" ", "_")}.png'
    
    # Añadir 'unknown' a las etiquetas por si acaso, aunque no debería aparecer
    cm_labels = list(classes)
    
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)), normalize='true')
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=cm_labels, yticklabels=cm_labels)
    plt.title(f'Matriz de Confusión - {model_name}')
    plt.xlabel('Predicción')
    plt.ylabel('Valor Real')
    plt.savefig(filename)
    print(f"Matriz de confusión guardada en: {filename}")
    plt.close()

def plot_feature_importance(model, feature_names, model_name):
    os.makedirs('./reports', exist_ok=True)
    filename = f'./reports/fi_{model_name.lower().replace(" ", "_")}.png'
    
    if not hasattr(model, 'feature_importances_'):
        print(f"El modelo {model_name} no tiene 'feature_importances_'.")
        return
        
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)
    
    print(f"\nTop 20 características más importantes ({model_name}):")
    print(feature_importance_df.head(20))
    
    plt.figure(figsize=(10, 12))
    sns.barplot(x='importance', y='feature', data=feature_importance_df.head(20), palette='viridis')
    plt.title(f'Top 20 Características más Importantes ({model_name})')
    plt.savefig(filename)
    print(f"Gráfico de importancia de características guardado en: {filename}")
    plt.close()
# --- Fin de Funciones de Ploteo ---

def main():
    print("Iniciando el script de ENTRENAMIENTO (Temporal + Velocidad)...")
    
    # --- 1. Cargar y Preparar los Datos ---
    data_path = './data/processed/model_features.csv'
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: No se encontró '{data_path}'.")
        print("Asegúrate de ejecutar 'feature_engineering.py' primero.")
        return

    print(f"Dataset cargado. Forma: {df.shape}")

    # Preparar X (características) y y (etiquetas)
    X = df.drop(columns=['action', 'video_filename', 'frame_idx'])
    y_raw = df['action']

    # Codificar las etiquetas
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    os.makedirs('./models', exist_ok=True)
    joblib.dump(le, './models/label_encoder.joblib')
    print(f"LabelEncoder guardado en './models/label_encoder.joblib'")

    print(f"Forma de X: {X.shape}") # Debería tener 48 columnas
    print(f"Forma de y: {y.shape}")
    print(f"Clases: {le.classes_}") # Deberías ver tus 5 clases

    if len(le.classes_) < 2:
        print("Error: El dataset solo tiene 1 clase. El entrenamiento fallará.")
        print("Por favor, verifica que 'process.py' haya procesado todas tus carpetas de video.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"Muestras de entrenamiento: {X_train.shape[0]}")
    print(f"Muestras de prueba: {X_test.shape[0]}")

    # --- 2. Construir Pipelines (con PCA) ---
    print("\nConstruyendo pipelines con StandardScaler y PCA...")
    
    pipe_rf = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95)),
        ('rf', RandomForestClassifier(random_state=42))
    ])

    pipe_xgb = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95)),
        ('xgb', XGBClassifier(random_state=42, eval_metric='mlogloss'))
    ])

    param_grid_rf = {
        'pca__n_components': [0.95, None], # Probar con y sin PCA
        'rf__n_estimators': [100, 200],
        'rf__max_depth': [20, None]
    }

    param_grid_xgb = {
        'pca__n_components': [0.95, None],
        'xgb__n_estimators': [100, 200],
        'xgb__learning_rate': [0.1, 0.3],
    }

    # --- 3. Entrenamiento y Ajuste de Hiperparámetros ---
    print("\nIniciando ajuste de Random Forest...")
    grid_rf = GridSearchCV(pipe_rf, param_grid_rf, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_rf.fit(X_train, y_train)
    print(f"Mejores parámetros para RF: {grid_rf.best_params_}")
    print(f"Mejor score (accuracy) en CV: {grid_rf.best_score_:.4f}")

    print("\nIniciando ajuste de XGBoost...")
    grid_xgb = GridSearchCV(pipe_xgb, param_grid_xgb, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_xgb.fit(X_train, y_train)
    print(f"Mejores parámetros para XGB: {grid_xgb.best_params_}")
    print(f"Mejor score (accuracy) en CV: {grid_xgb.best_score_:.4f}")

    # --- 4. Evaluación de Modelos en el Conjunto de Prueba ---
    print("\nEvaluando modelos en el conjunto de prueba...")
    best_rf = grid_rf.best_estimator_
    best_xgb = grid_xgb.best_estimator_
    target_names = le.classes_

    y_pred_rf = best_rf.predict(X_test)
    y_pred_xgb = best_xgb.predict(X_test)

    print("\n========= REPORTE DE RANDOM FOREST =========")
    print(f"Accuracy en Test: {accuracy_score(y_test, y_pred_rf):.4f}\n")
    print(classification_report(y_test, y_pred_rf, target_names=target_names))
    plot_confusion_matrix(y_test, y_pred_rf, target_names, "Random Forest")

    print("\n========= REPORTE DE XGBOOST =========")
    print(f"Accuracy en Test: {accuracy_score(y_test, y_pred_xgb):.4f}\n")
    print(classification_report(y_test, y_pred_xgb, target_names=target_names))
    plot_confusion_matrix(y_test, y_pred_xgb, target_names, "XGBoost")

    # --- 5. Análisis de Importancia de Características ---
    if 'rf' in best_rf.named_steps:
        model_rf = best_rf.named_steps['rf']
        if hasattr(best_rf.named_steps['pca'], 'n_components_') and best_rf.named_steps['pca'].n_components_ is not None:
            n_comps = best_rf.named_steps['pca'].n_components_
            feature_names = [f'Componente_{i+1}' for i in range(n_comps)]
        else:
            feature_names = X.columns
            
        plot_feature_importance(model_rf, feature_names, "RF")
    
    # --- 6. Guardar los modelos finales ---
    print("\nGuardando los mejores modelos entrenados...")
    joblib.dump(best_rf, './models/best_random_forest_model.joblib')
    joblib.dump(best_xgb, './models/best_xgboost_model.joblib')
    print("Modelos guardados en la carpeta './models'")
    
    print("\nScript de entrenamiento completado.")

if __name__ == "__main__":
    main()