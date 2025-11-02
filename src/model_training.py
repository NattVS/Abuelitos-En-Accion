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

def plot_confusion_matrix(y_true, y_pred, classes, model_name):
    """Guarda una matriz de confusión en un archivo."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(f'Matriz de Confusión - {model_name}')
    plt.xlabel('Predicción')
    plt.ylabel('Valor Real')
    filename = f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
    plt.savefig(filename)
    print(f"Matriz de confusión guardada en: {filename}")
    plt.close()

def plot_feature_importance(model, feature_names, model_name):
    """Guarda un gráfico de importancia de características (si aplica)."""
    if not hasattr(model, 'feature_importances_'):
        print(f"El modelo {model_name} no tiene 'feature_importances_'. Omitiendo gráfico.")
        return
        
    importances = model.feature_importances_
    
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)
    
    # Mostrar las 20 características más importantes
    print(f"\nTop 20 características más importantes ({model_name}):")
    print(feature_importance_df.head(20))
    
    # Graficar las 20 más importantes
    plt.figure(figsize=(10, 12))
    sns.barplot(x='importance', y='feature', data=feature_importance_df.head(20), palette='viridis')
    plt.title(f'Top 20 Características más Importantes ({model_name})')
    filename = f'feature_importance_{model_name.lower().replace(" ", "_")}.png'
    plt.savefig(filename)
    print(f"Gráfico de importancia de características guardado en: {filename}")
    plt.close()

def main():
    print("Iniciando el script de entrenamiento de modelos...")
    
    # --- 1. Cargar y Preparar los Datos ---
    data_path = './data/processed/model_features.csv'
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: No se encontró '{data_path}'.")
        print("Asegúrate de haber ejecutado 'feature_engineering.py' primero.")
        return

    print(f"Dataset cargado. Forma: {df.shape}")

    # Preparar X (características) y y (etiquetas)
    X = df.drop(columns=['action', 'video_filename', 'frame_idx'])
    y_raw = df['action']

    # Codificar las etiquetas (de texto a números)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    # Guardar el LabelEncoder para usarlo en el despliegue
    os.makedirs('./models', exist_ok=True)
    joblib.dump(le, './models/label_encoder.joblib')
    print("LabelEncoder guardado en './models/label_encoder.joblib'")

    print(f"Forma de X: {X.shape}")
    print(f"Forma de y: {y.shape}")
    print(f"Clases: {le.classes_}")

    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"Muestras de entrenamiento: {X_train.shape[0]}")
    print(f"Muestras de prueba: {X_test.shape[0]}")

    # --- 2. Construir Pipelines (AHORA CON PCA) ---
    print("\nConstruyendo pipelines con StandardScaler y PCA...")

    # Usamos n_components=0.95 para que PCA seleccione automáticamente
    # los componentes necesarios para explicar el 95% de la varianza.
    
    # Pipeline para Random Forest
    pipe_rf = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95)),  
        ('rf', RandomForestClassifier(random_state=42))
    ])

    # Pipeline para XGBoost
    pipe_xgb = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95)),  
        ('xgb', XGBClassifier(random_state=42, eval_metric='mlogloss')) 
    ])

    # Definir cuadrícula de hiperparámetros
    param_grid_rf = {
        'pca__n_components': [0.90, 0.95, 0.99],
        'rf__n_estimators': [100, 200],
        'rf__max_depth': [10, 20]
    }

    param_grid_xgb = {
        'pca__n_components': [0.90, 0.95],
        'xgb__n_estimators': [100, 200],
        'xgb__learning_rate': [0.1, 0.3],
        'xgb__max_depth': [5, 7]
    }

    # --- 3. Entrenamiento y Ajuste de Hiperparámetros ---
    print("\nIniciando ajuste de Random Forest (esto puede tardar)...")
    grid_rf = GridSearchCV(pipe_rf, param_grid_rf, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_rf.fit(X_train, y_train)
    print(f"Mejores parámetros para RF: {grid_rf.best_params_}")
    print(f"Mejor score (accuracy) en CV: {grid_rf.best_score_:.4f}")

    print("\nIniciando ajuste de XGBoost (esto puede tardar)...")
    grid_xgb = GridSearchCV(pipe_xgb, param_grid_xgb, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_xgb.fit(X_train, y_train)
    print(f"Mejores parámetros para XGB: {grid_xgb.best_params_}")
    print(f"Mejor score (accuracy) en CV: {grid_xgb.best_score_:.4f}")

    # --- 4. Evaluación de Modelos en el Conjunto de Prueba ---
    print("\nEvaluando modelos en el conjunto de prueba...")
    best_rf = grid_rf.best_estimator_
    best_xgb = grid_xgb.best_estimator_

    y_pred_rf = best_rf.predict(X_test)
    y_pred_xgb = best_xgb.predict(X_test)

    target_names = le.classes_

    # --- Resultados de Random Forest ---
    print("\n========= REPORTE DE RANDOM FOREST =========")
    print(f"Accuracy en Test: {accuracy_score(y_test, y_pred_rf):.4f}\n")
    print(classification_report(y_test, y_pred_rf, target_names=target_names))
    plot_confusion_matrix(y_test, y_pred_rf, target_names, "Random Forest")

    # --- Resultados de XGBoost ---
    print("\n========= REPORTE DE XGBOOST =========")
    print(f"Accuracy en Test: {accuracy_score(y_test, y_pred_xgb):.4f}\n")
    print(classification_report(y_test, y_pred_xgb, target_names=target_names))
    plot_confusion_matrix(y_test, y_pred_xgb, target_names, "XGBoost")

    # --- 5. Análisis de Importancia de Características ---
    if 'rf' in best_rf.named_steps:
        # Obtener los nombres de los componentes de PCA
        n_comps = best_rf.named_steps['pca'].n_components_
        pca_feature_names = [f'Componente_{i+1}' for i in range(n_comps)]
        plot_feature_importance(best_rf.named_steps['rf'], pca_feature_names, "RF (Componentes PCA)")
    
    # --- 6. Guardar los modelos finales ---
    print("\nGuardando los mejores modelos entrenados...")
    joblib.dump(best_rf, './models/best_random_forest_model.joblib')
    joblib.dump(best_xgb, './models/best_xgboost_model.joblib')
    print("Modelos guardados en la carpeta './models'")
    
    print("\nScript de entrenamiento completado.")

if __name__ == "__main__":
    main()

