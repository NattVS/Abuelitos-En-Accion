# Proyecto Final APO3

**Universidad Icesi – 2025-2**
**Curso: Algoritmos y Programación III**

**Entregas 1 y 2 – EDA y Modelado**

---

## Objetivo General

Desarrollar una herramienta capaz de analizar actividades humanas específicas (*caminar hacia la cámara, caminar hacia atrás, girar, sentarse y levantarse*) mediante el seguimiento de *landmarks* corporales detectados con **MediaPipe**.

El propósito del sistema es apoyar el monitoreo de movilidad en adultos mayores, clasificando sus movimientos en tiempo real.

---

## Estructura del Proyecto

```text
.
│
├── data/
│   ├── raw/
│   │   └── videos/
│   │       ├── caminar-adelante/
│   │       ├── ... (4 acciones más)
│   │
│   └── processed/
│       ├── datos_analisis.csv
│       ├── datosmediapipe.csv
│       └── model_features.csv
│
├── models/
│   ├── best_random_forest_model.joblib
│   ├── best_xgboost_model.joblib
│   └── label_encoder.joblib
│
├── notebooks/
│   └── EDA.ipynb
│
├── reports/
│   ├── APO3_Informe_Entrega1.pdf
│   ├── APO3_Informe_Entrega2.pdf
│   ├── confusion_matrix_random_forest.png
│   ├── confusion_matrix_xgboost.png
│   └── feature_importance_rf_(componentes_pca).png
│
├── src/
│   ├── process.py                 # (Paso 1) Extrae landmarks de videos a CSV
│   ├── feature_engineering.py     # (Paso 2) Crea características (ángulos) desde CSV
│   └── model_training.py          # (Paso 3) Entrena modelos con PCA y guarda en .joblib
│
├── requirements.txt
└── README.md
```

---

## Requisitos del entorno

Asegúrate de tener instalado **Python 3.10+** y las siguientes dependencias:

> Se recomienda crear un entorno virtual:

```bash
python -m venv venv
# (Windows)
venv\Scripts\activate
# (macOS / Linux)
source venv/bin/activate
```

Instalación rápida:

```bash
pip install opencv-python-headless mediapipe pandas numpy tqdm
pip install scikit-learn xgboost matplotlib seaborn joblib
```

> También puedes usar el archivo `requirements.txt`.

---

## Flujo de Trabajo y Ejecución

El proyecto se ejecuta en **3 pasos** principales para pasar de los videos en crudo a un modelo entrenado.

### Paso 1: Extracción de Landmarks (`process.py`)

Este script recorre todos los videos en `data/raw/videos/`, extrae los **33 landmarks** de MediaPipe de cada *frame* y genera dos archivos CSV en `data/processed/`:

* `datosmediapipe.csv`: Datos *wide-format* con las coordenadas (`x, y, z, vis`) de los 33 landmarks por *frame*.
* `datos_analisis.csv`: Metadatos de los videos (duración, luminancia, etc.).

**Ejecución:**

```bash
python src/process.py
```

---

### Paso 2: Ingeniería de Características (`feature_engineering.py`)

Este script toma `datosmediapipe.csv` y prepara el dataset para el modelo:

* **Normalización:** Centra la pose usando la cadera y escala usando la distancia de los hombros.
* **Cálculo de Ángulos:** Calcula 8 ángulos corporales clave (codos, hombros, caderas, rodillas).
* **Aplanamiento:** Combina ángulos y coordenadas normalizadas en un vector de **107 características**.

**Ejecución:**

```bash
python src/feature_engineering.py
```

**Salida:** `data/processed/model_features.csv`

---

### Paso 3: Entrenamiento de Modelos (`model_training.py`)

Script para entrenar y evaluar modelos de *Machine Learning*.

* Carga `model_features.csv`.
* Codifica etiquetas de texto (ej. `sentarse`) a números.
* Construye un **Pipeline** de Scikit-learn que incluye:

  * `StandardScaler`
  * `PCA` (95% de varianza)
  * Clasificador (**Random Forest** o **XGBoost**)
* Usa `GridSearchCV` para mejores hiperparámetros.
* Evalúa en un *hold-out* de **30%**.

**Ejecución:**

```bash
python src/model_training.py
```

**Salidas:**

* Modelos (`.joblib`) en `models/`.
* Gráficos (`.png`) en `reports/`.
* Reportes de clasificación en la terminal.

---

## 4. Análisis Exploratorio (EDA)

El notebook `notebooks/EDA.ipynb` contiene el análisis exploratorio inicial (Entrega 1) que validó la calidad de los datos y la viabilidad del proyecto.

* Dataset balanceado (**18 videos por clase**).
* Alta calidad de detección (**visibilidad promedio > 0.9**).
* Clara separabilidad de clases (ej. posición de la cadera).

---

## 5. Resultados del Modelo (Entrega 2)

> *(Pega aquí los resultados finales de tus modelos. Ejemplo):*

Ambos modelos alcanzaron un rendimiento excelente en el conjunto de prueba:

* **Random Forest:** *Accuracy* de **96.69%**
* **XGBoost:** *Accuracy* de **96.26%**

Las matrices de confusión (guardadas en `reports/`) muestran que la mayoría de los errores ocurren entre clases lógicamente similares (ej. *caminar-adelante* vs *caminar-atras*), lo cual es esperado.

---

## 6. Próximos Pasos (Despliegue)

Siguiente paso: usar los modelos guardados en `models/` para crear una aplicación **en tiempo real**.

* Crear `src/real_time_inference.py`.
* El script:

  * Carga la cámara web con **OpenCV**.
  * Carga el *pipeline* `.joblib` (ej. `best_random_forest_model.joblib`).
  * En un bucle, captura *frames*, aplica la misma función `extract_features` y usa el *pipeline* para predecir la acción.
  * Muestra la predicción en pantalla con `cv2.putText()`.

---

## 7. Aspectos Éticos

* Se garantiza el **consentimiento informado** y el uso **exclusivamente académico** de las grabaciones.
* Las imágenes se almacenan en **carpetas locales seguras**, no en servicios en la nube.
* El modelo se diseña para **minimizar sesgos** (edad, género, iluminación, vestimenta).

---

## Autoras

**Mariana Agudelo Salazar** y **Natalia Vargas**
Universidad Icesi – Facultad Barberi de Ingeniería, Diseño y Ciencias Aplicadas
Departamento de Computación y Sistemas Inteligentes
