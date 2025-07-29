# 🔍 Clasificador de Paneles Solares por Imágenes Infrarrojas

Este proyecto implementa un modelo de red neuronal convolucional (CNN) entrenado para detectar **anomalías en paneles solares** utilizando imágenes térmicas infrarrojas. Las imágenes son procesadas y clasificadas en distintas categorías según su estado (normal o defectuoso). Además, se incluye una interfaz gráfica interactiva para realizar predicciones fácilmente desde archivos locales.

---

## 📌 Tabla de Contenidos

- [🧠 Modelo](#-modelo)
- [📁 Estructura del Repositorio](#-estructura-del-repositorio)
- [⚙️ Requisitos](#️-requisitos)
- [🚀 Entrenamiento](#-entrenamiento)
- [🧪 Resultados](#-resultados)
- [🖼️ Interfaz Gráfica](#-interfaz-gráfica)
- [📦 Despliegue](#-despliegue)
- [📤 Input Esperado](#-input-esperado)
- [📄 Licencia](#-licencia)

---

## 🧠 Modelo

Se utilizó una red neuronal convolucional (CNN) construida con **TensorFlow/Keras**, compuesta por:

- 3 capas `Conv2D` con `ReLU` y `MaxPooling2D`
- Regularización `Dropout` y `L1_L2`
- Capa `Dense` intermedia con 120 unidades
- Capa `Dense` final `softmax` para clasificación multiclase

> **Precisión alcanzada**:  
> - Entrenamiento: ~57%  
> - Validación: ~64%

---

## 📁 Estructura del Repositorio

```bash
├── main.py                  # Script de entrenamiento
├── myapp_ui.py             # Interfaz gráfica de usuario (UI)
├── cambio_aumento_datos.py # Aumento de datos y ajustes
├── cargar_datos.py         # Carga y procesamiento de datos
├── remapeo_unir_grupos.py  # Preprocesamiento y agrupamiento
├── model_066.h5            # Modelo entrenado guardado
├── model_params.pkl        # Pesos del modelo
├── label_encoder.pkl       # Codificador de etiquetas
├── logo.png / LOGOICO.ico  # Iconos de la app
├── utils/                  # Utilidades y funciones auxiliares
├── README.md               # Este archivo
├── .gitignore              # Archivos ignorados por Git
└── MyApp.spec              # Configuración de pyinstaller
