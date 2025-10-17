
# 🧠 Proyecto Final APO3
**Universidad Icesi – 2025-2**  
**Curso:** Algoritmos y Programación III  
**Entrega 1 – Análisis Exploratorio de Datos (EDA)**  

## 🎯 Objetivo General

Desarrollar una herramienta capaz de **analizar actividades humanas específicas** (caminar hacia la cámara, caminar hacia atrás, girar, sentarse y levantarse) mediante el seguimiento de **landmarks corporales** detectados con **MediaPipe**.  
El propósito del sistema es apoyar el monitoreo de **movilidad en adultos mayores**, clasificando sus movimientos en tiempo real.

---

## 📁 Estructura del Proyecto

```
.
│
├── data/
│   ├── raw/
│   │   └── videos/
│   │       ├── caminar-adelante/
│   │       ├── caminar-atras/
│   │       ├── girar/
│   │       ├── sentarse/
│   │       └── pararse/
│   │
│   └── processed/
│       ├── datos_analisis.csv
│       └── datosmediapipe.csv
│
├── src/
│   ├── process.py        # Script de extracción de landmarks y métricas de video
│   └── eda.py (o EDA.ipynb)  # Notebook de análisis exploratorio de los datos
├── reports/
│   ├── APO3_Informe_Entrega1.pdf
└── README.md

````

---

## Requisitos del entorno

Asegúrate de tener instalado **Python 3.10+** y las siguientes dependencias:

```bash
pip install -r requirements
````

*(Recomendado: crear un entorno virtual con `python -m venv venv` y activarlo antes de instalar)*

---

## 1. Generación de Datos con `process.py`

El script `process.py` recorre todas las carpetas de acciones dentro de `data/raw/videos/`, analiza los videos y genera dos archivos CSV:

* **`datosmediapipe.csv`** → 33 landmarks de MediaPipe (x, y, z, visibility) por frame.
* **`datos_analisis.csv`** → Información general de los frames: duración, FPS, luminancia, movimiento, etc.

### ▶️ Ejecución:

```bash
python src/process.py
```

**Parámetros opcionales:**

* `--stride N` → Procesa cada N fotogramas (para acelerar el análisis).
* `--static_image_mode` → Modo estático (más preciso, más lento).
* `--min_detection_confidence` → Confianza mínima para detección de pose.
* `--min_tracking_confidence` → Confianza mínima para seguimiento.

📦 Al finalizar, se crearán los CSV en `data/processed/`.

---

## 2. Análisis Exploratorio (EDA)

El notebook `EDA.ipynb` analiza los dos datasets generados:

### Contenidos principales:

1. **Carga y validación de datos**

   * Tamaño de los datasets y tipos de variables.
   * Conteo de valores nulos.
2. **Análisis de metadatos de video (`datos_analisis.csv`)**

   * Balance de clases (número de videos y frames por acción).
   * Distribución de duraciones de video y luminancia media.
   * Validación de condiciones homogéneas de grabación.
3. **Análisis de landmarks (`datosmediapipe.csv`)**

   * Evaluación de visibilidad promedio de los 33 puntos.
   * Ejemplos visuales del esqueleto humano (MediaPipe).
   * Distribuciones de posición de caderas y rodillas por acción.
4. **Conclusiones y próximos pasos**

   * Confirmación de calidad y balance de datos.
   * Identificación de características clave (`hip_y_avg`).
   * Recomendaciones para ingeniería de características y modelado supervisado.


## 3. Resultados Principales

* Dataset **balanceado** entre las 5 clases de acción.
* **Alta calidad** de detección (visibilidad promedio ≈ 0.9).
* Claras diferencias en la posición vertical de caderas entre *sentarse* y *pararse*.
* Luminancia y duración de videos **controladas** (homogeneidad experimental).

---

## 4. Aspectos Éticos

* Se garantiza el **consentimiento informado** y el **uso exclusivo académico** de las grabaciones.
* Las imágenes se almacenan en carpetas locales seguras, no en servicios en la nube.
* El modelo se diseñará para minimizar sesgos (edad, género, iluminación, vestimenta).


## Próximos Pasos

1. **Ingeniería de características:**

   * Cálculo de ángulos, velocidades y distancias entre articulaciones.
2. **Entrenamiento de modelos supervisados:**

   * Comparar desempeño entre Random Forest, SVM y XGBoost.
3. **Evaluación de rendimiento:**

   * Métricas: Accuracy, Precision, Recall y F1-Score.
4. **Despliegue:**

   * Interfaz simple para visualizar resultados en tiempo real con MediaPipe.


**Autoras:**

> Mariana Agudelo Salazar y Natalia Vargas
> Universidad Icesi – Facultad Barberi de Ingeniería, Diseño y Ciencias Aplicadas
> Departamento de Computación y Sistemas Inteligentes

