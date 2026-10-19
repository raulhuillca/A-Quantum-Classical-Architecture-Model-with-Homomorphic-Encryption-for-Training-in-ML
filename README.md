# Modelo de Arquitectura Cuántica-Clásica con Encriptación Homomórfica para Entrenamiento y Preservación de la Privacidad en el Aprendizaje Automático

Este repositorio contiene la implementación experimental del proyecto de investigación:
**"Modelo de Arquitectura Cuántica-Clásica con Encriptación Homomórfica para Entrenamiento y Preservación de la Privacidad en el Aprendizaje Automático".**

La arquitectura propuesta integra la **Transformada Cuántica de Fourier (QFT)** con el **esquema de encriptación homomórfica CKKS** para investigar el aprendizaje automático que preserva la privacidad sobre representaciones en el dominio de la frecuencia.

El flujo de trabajo experimental combina el preprocesamiento clásico, la transformación de características cuánticas, la encriptación homomórfica, el aprendizaje automático y la evaluación cuantitativa.

## 📌 Tabla de Contenidos
- [Descripción General](#-descripción-general)
- [Objetivo de la Investigación](#-objetivo-de-la-investigación)
- [Arquitectura del Sistema](#-arquitectura-del-sistema)
- [Fundamentos Matemáticos](#-fundamentos-matemáticos)
- [Canalización y Procesamiento de Datos](#-canalización-y-procesamiento-de-datos)
- [Transformada Cuántica de Fourier](#️-transformada-cuántica-de-fourier)
- [Canal de Encriptación Homomórfica](#-canal-de-encriptación-homomórfica)
- [Modelos de Aprendizaje Automático](#-modelos-de-aprendizaje-automático)
- [Configuración Experimental](#-configuración-experimental)
- [Métricas de Evaluación](#-métricas-de-evaluación)
- [Resultados Experimentales](#-resultados-experimentales)
- [Estructura del Repositorio](#-estructura-del-repositorio)
- [Instalación](#-instalación)
- [Flujo de Ejecución](#️-flujo-de-ejecución)
- [Hardware y Entorno](#-hardware-y-entorno)
- [Reproducibilidad](#-reproducibilidad)
- [Consideraciones de Seguridad](#-consideraciones-de-seguridad)
- [Limitaciones](#limitaciones)
- [Citación](#citación)
- [Autor](#autor)
- [Licencia](#licencia)

---

## 🔬 Descripción General

La arquitectura propuesta investiga un canal computacional híbrido en el cual los datos clásicos se transforman a una representación en el dominio de la frecuencia utilizando la Transformada Cuántica de Fourier (QFT) y, posteriormente, se protegen mediante la encriptación homomórfica CKKS.

El flujo de trabajo general es:

```text
Datos Clásicos
      │
      ▼
Generación de Conjunto de Datos Sintético
      │
      ▼
Preprocesamiento de Datos
      │
      ▼
Normalización
      │
      ▼
Codificación de Amplitud
      │
      ▼
Transformada Cuántica de Fourier (QFT)
      │
      ▼
Representación en el Dominio de la Frecuencia
      │
      ▼
Encriptación Homomórfica CKKS
      │
      ▼
Aprendizaje Automático
      │
      ├── Regresión Lineal
      ├── Regresión Logística
      ├── LS-SVM
      └── MLP
      │
      ▼
Evaluación
      │
      ├── MSE
      ├── RMSE
      ├── MAE
      ├── R²
      └── Precisión (Accuracy)
```

El propósito no es afirmar que la computación cuántica reemplaza al aprendizaje automático clásico, sino estudiar experimentalmente la integración de:

- Transformación de características cuánticas.
- Encriptación homomórfica.
- Aprendizaje automático.
- Preservación de la privacidad.
- Escalabilidad computacional.

### 🎯 Objetivo de la Investigación

El objetivo principal es desarrollar y evaluar experimentalmente una arquitectura cuántica-clásica con encriptación homomórfica para aplicaciones de aprendizaje automático donde la privacidad de los datos es un requisito importante.

La arquitectura evalúa la interacción entre:

```plaintext
Transformada Cuántica de Fourier
              +
Encriptación Homomórfica
              +
Aprendizaje Automático
              +
Preservación de la Privacidad
```

### 🏗️ Arquitectura del Sistema

La arquitectura propuesta consta de cuatro fases principales.

```plaintext
┌──────────────────────────────────────────────────┐
│                     FASE 1                       │
│         PROCESAMIENTO DE DATOS CLÁSICOS          │
│                                                  │
│ Datos Sintéticos → Normalización → Codificación  │
└───────────────────────┬──────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────┐
│                     FASE 2                       │
│             TRANSFORMACIÓN CUÁNTICA              │
│                                                  │
│           Codificación de Amplitud → QFT         │
└───────────────────────┬──────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────┐
│                     FASE 3                       │
│             PROTECCIÓN HOMOMÓRFICA               │
│                                                  │
│       Representación de Frecuencia → CKKS        │
└───────────────────────┬──────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────┐
│                     FASE 4                       │
│             APRENDIZAJE AUTOMÁTICO               │
│                                                  │
│     RL → Regresión Logística → LS-SVM → MLP      │
└──────────────────────────────────────────────────┘
```

### 📐 Fundamentos Matemáticos

#### 1. Codificación de Amplitud

Sea:

$$x=[x_0,x_1,\ldots,x_{N-1}]$$

un vector clásico normalizado. El vector puede representarse como un estado cuántico:

$$\vert{}x\rangle = \sum_{k=0}^{N-1}x_k\vert{}k\rangle$$

sujeto a:

$$\sum_{k=0}^{N-1}\vert{}x_k\vert{}^2=1$$

La codificación de amplitud permite que un vector clásico sea representado a través de las amplitudes de un estado cuántico.

#### 2. Transformada Cuántica de Fourier

La Transformada Cuántica de Fourier se define como:

$$QFT\vert{}k\rangle = \frac{1}{\sqrt{N}} \sum_{j=0}^{N-1} e^{2\pi i kj/N}\vert{}j\rangle$$

Para un estado general:

$$\vert{}x\rangle = \sum_{k=0}^{N-1}x_k\vert{}k\rangle$$

el estado transformado resulta en:

$$QFT\vert{}x\rangle = \frac{1}{\sqrt{N}} \sum_{j=0}^{N-1} \left( \sum_{k=0}^{N-1} x_k e^{2\pi i kj/N} \right) \vert{}j\rangle$$

Las amplitudes resultantes contienen información sobre la representación en el dominio de la frecuencia de la entrada.


#### 3. Encriptación Homomórfica

La arquitectura utiliza CKKS (Cheon-Kim-Kim-Song) para aritmética aproximada sobre datos numéricos encriptados. CKKS se basa en construcciones criptográficas basadas en retículos (lattices) relacionadas con el problema de Aprendizaje con Errores en Anillos (RLWE).

La cadena de procesamiento conceptual es:

```plaintext
  Texto Plano (Plaintext)
             │
             ▼
     Codificación CKKS
             │
             ▼
       Encriptación
             │
             ▼
 Texto Cifrado (Ciphertext)
             │
             ▼
  Operaciones Homomórficas
             │
             ▼
      Desencriptación
             │
             ▼
   Texto Plano Aproximado
```

### 📊 Canalización y Procesamiento de Datos

El flujo experimental comienza con conjuntos de datos sintéticos que representan datos de consumo de energía. Las variables principales son:

| Variable | Descripción |
| :--- | :--- |
| `temp_ambiente` | Temperatura ambiente |
| `num_personas` | Número de personas |
| `eficiencia_maquina` | Eficiencia de la máquina |
| `consumo_kwh` | Objetivo de consumo de energía |

Los experimentos consideran:

- 1,000 registros
- 10,000 registros
- 100,000 registros
- 500,000 registros
- 1,000,000 registros

#### 1. Generación de Datos Sintéticos
##### El Algoritmo (TVAE Synthetic Data Generation)

**Algoritmo: Massive Synthetic Data Generation via TVAE and Batch Processing for CKKS Homomorphic Encryption**
<hr>

$$
\begin{array}{rl}
& \text{Input: Real tabular dataset } D_{\text{real}}\text{, Total target synthetic rows } N\text{, Batch size } B\text{,} \\
& \quad \text{Training epochs } E\text{, Output filename } F_{\text{out}} \\
& \text{Output: Exported CSV dataset } F_{\text{out}} \text{ optimized for CKKSEncoder} \\
1 & M \leftarrow \text{SingleTableMetadata}() \\
2 & M.\text{detect\\_from\\_dataframe}(D_{\text{real}}) \\
3 & \mathcal{M} \leftarrow \text{TVAESynthesizer}(M, \text{enforce\\_min\\_max\\_values}=\text{True}, \text{epochs}=E) \\
4 & \mathcal{M}.\text{fit}(D_{\text{real}}) \hspace{4.8cm} \text{// Train TVAE on latent feature distributions} \\
5 & Chunks \leftarrow [\,] \\
6 & N_{\text{generated}} \leftarrow 0 \\
7 & \textbf{while } N_{\text{generated}} < N \textbf{ do} \\
8 & \quad b_{\text{current}} \leftarrow \min(B, N - N_{\text{generated}}) \hspace{1.6cm} \text{// Dynamic batch boundary calculation} \\
9 & \quad C_k \leftarrow \mathcal{M}.\text{sample}(\text{num\\_rows}=b_{\text{current}}) \hspace{0.9cm} \text{// Sample vectors from TVAE latent space} \\
10 & \quad \text{Append } C_k \text{ to } Chunks \\
11 & \quad N_{\text{generated}} \leftarrow N_{\text{generated}} + b_{\text{current}} \\
12 & D_{\text{synth}} \leftarrow \text{Concat}(Chunks, \text{ignore\\_index}=\text{True}) \\
13 & \textbf{if } \text{HasNullValues}(D_{\text{synth}}) \textbf{ then} \\
14 & \quad D_{\text{synth}} \leftarrow \text{ImputeWithColumnMean}(D_{\text{synth}}) \hspace{0.5cm} \text{// Prevent CKKS encoding faults} \\
15 & \text{ExportToCSV}(D_{\text{synth}}, \text{path}=F_{\text{out}}, \text{float\\_format}=\text{'\%.10f'}) \hspace{0.2cm} \text{// Maintain 10-decimal float accuracy} \\
16 & \textbf{return } F_{\text{out}}
\end{array}
$$

<hr>

La primera etapa genera los conjuntos de datos experimentales. Ejemplo:

- `dataset_sintetico_1k.csv`
- `dataset_sintetico_10k.csv`
- `dataset_sintetico_100k.csv`
- `dataset_sintetico_500k.csv`
- `dataset_sintetico_1m.csv`

Los datos generados son posteriormente normalizados antes del procesamiento cuántico.

<details>
<summary><b>💻 Haz clic aquí para desplegar el Código Fuente (Python)</b></summary>

```python
import pandas as pd
import numpy as np
import os
import warnings

# =============================================================================
# PASO 1: CONFIGURACIÓN DEL ENTORNO Y LIBRERÍAS
# =============================================================================
# Forzamos el uso de la CPU para evitar errores de falta de memoria (OOM) 
# al generar grandes volúmenes de datos sintéticos con la GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = "" 
warnings.filterwarnings('ignore')

from sdv.metadata import SingleTableMetadata
from sdv.single_table import TVAESynthesizer
from sdv.evaluation.single_table import evaluate_quality
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# =============================================================================
# PASO 2: GENERACIÓN DE DATOS SEMILLA (GROUND TRUTH)
# =============================================================================
# Fijamos la semilla aleatoria para asegurar la reproducibilidad del experimento.
np.random.seed(42)
n_rows_seed = 1000

# Creamos las variables independientes (Matriz X) con diferentes distribuciones
data = pd.DataFrame({
    'temp_ambiente': np.random.normal(25, 5, n_rows_seed),
    'num_personas': np.random.poisson(10, n_rows_seed).astype(float),
    'eficiencia_maquina': np.random.uniform(0.5, 0.95, n_rows_seed),
})

# Definimos la relación lineal exacta (Variable objetivo Y) introduciendo ruido blanco
# Ecuación: y = 10 + 2.5*x1 + 4.0*x2 - 12.0*x3 + ruido
data['consumo_kwh'] = (
    10 + 
    (2.5 * data['temp_ambiente']) + 
    (4.0 * data['num_personas']) - 
    (12.0 * data['eficiencia_maquina']) + 
    np.random.normal(0, 1, n_rows_seed)
)

print(f"[INFO] Datos semilla generados: {data.shape[0]} registros.")

# =============================================================================
# PASO 3: EXTRACCIÓN DE METADATOS Y ENTRENAMIENTO DEL MODELO TVAE
# =============================================================================
# SDV requiere conocer la estructura de los datos (tipos, límites) antes de modelar.
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data)

# Instanciamos el Autoencoder Variacional (TVAE). 
# enforce_min_max_values=True evita que genere valores lógicamente imposibles.
synthesizer = TVAESynthesizer(
    metadata,
    enforce_min_max_values=True,
    epochs=500
)

print("[INFO] Entrenando TVAE (Variational Autoencoder) para preservar relaciones lineales...")
synthesizer.fit(data)
print("[INFO] Entrenamiento completado.")

# =============================================================================
# PASO 4: FUNCIÓN DE GENERACIÓN MASIVA POR LOTES (BATCHING)
# =============================================================================
def generar_sinteticos_masivos(synthesizer, total_rows, batch_size=10000):
    """
    Genera grandes volúmenes de datos sintéticos dividiendo la carga en lotes.
    Esto previene el colapso de la memoria RAM/VRAM al escalar a millones de registros.
    """
    print(f"\n[PROCESO] Generando {total_rows} datos sintéticos en lotes de {batch_size}...")
    chunks = []
    generados = 0
    
    while generados < total_rows:
        # Calculamos el tamaño del lote actual (el último lote podría ser menor)
        current_batch = min(batch_size, total_rows - generados)
        
        # Generamos la muestra del lote
        chunk = synthesizer.sample(num_rows=current_batch)
        chunks.append(chunk)
        
        # Actualizamos contadores
        generados += current_batch
        print(f"  -> Progreso: {generados} / {total_rows} registros generados...")
        
    # Concatenamos todos los lotes en un único DataFrame consolidado
    df_masivo = pd.concat(chunks, ignore_index=True)
    return df_masivo

# =============================================================================
# PASO 5: EJECUCIÓN DE LA GENERACIÓN Y VALIDACIÓN (TEST DE UTILIDAD)
# =============================================================================
# Para la prueba de validación, generaremos 10,000 datos usando nuestra nueva función
volumen_prueba = 10000
synthetic_data = generar_sinteticos_masivos(synthesizer, total_rows=volumen_prueba, batch_size=5000)

print("\n--- TEST DE UTILIDAD: REGRESIÓN LINEAL ---")

# Entrenamos un modelo de regresión con los datos reales originales
X_real = data[['temp_ambiente', 'num_personas', 'eficiencia_maquina']]
y_real = data['consumo_kwh']
reg_real = LinearRegression().fit(X_real, y_real)

# Entrenamos un modelo de regresión con los datos sintéticos generados
X_sint = synthetic_data[['temp_ambiente', 'num_personas', 'eficiencia_maquina']]
y_sint = synthetic_data['consumo_kwh']
reg_sint = LinearRegression().fit(X_sint, y_sint)

# Comparamos si el TVAE logró aprender y replicar los coeficientes originales
print(f"Coeficientes Reales:     {reg_real.coef_}")
print(f"Coeficientes Sintéticos: {reg_sint.coef_}")

# Evaluamos la varianza explicada (R2) en ambos conjuntos
r2_r = r2_score(y_real, reg_real.predict(X_real))
r2_s = r2_score(y_sint, reg_sint.predict(X_sint))

print(f"\nR2 Real:       {r2_r:.4f}")
print(f"R2 Sintético:  {r2_s:.4f}")

# Evaluación general de la fidelidad de la distribución multivariada usando SDV
report = evaluate_quality(data, synthetic_data, metadata)
print(f"\nPuntaje de Calidad General SDV: {report.get_score() * 100:.2f}%")


# =============================================================================
# PASO 6: EXPORTACIÓN PARA ENTORNOS DE CIFRADO (SEAL / CKKS)
# =============================================================================
def exportar_para_seal(df, filename="dataset_para_seal.csv"):
    """
    Acondiciona y exporta los datos para operaciones sobre esquemas encriptados (como CKKS).
    La alta precisión decimal y la ausencia de valores nulos son críticos para 
    multiplicar datos cifrados manteniendo la seguridad homomórfica y la estabilidad del ruido.
    """
    # Verificación de integridad: los esquemas criptográficos fallan con datos NaN
    if df.isnull().values.any():
        print("[ADVERTENCIA] Se detectaron valores nulos. Imputando con la media de la columna...")
        df = df.fillna(df.mean())
    
    # Exportación asegurando una representación de punto flotante extendida (10 decimales)
    # vital para la codificación polinomial de CKKS.
    df.to_csv(filename, index=False, float_format='%.10f')
    print(f"\n[INFO] Dataset exportado con éxito: {filename}")
    print("[INFO] Estructura lista para ser cargada por CKKSEncoder en C++ o Python.")

# Exportamos el conjunto de prueba generado
exportar_para_seal(synthetic_data, filename=f"dataset_sintetico_{volumen_prueba}.csv")

# =============================================================================
# INSTRUCCIONES PARA GENERAR MAYORES VOLÚMENES (100k, 500k, 1M)
# =============================================================================
###########################################
# =============================================================================
# GENERACIÓN DE 1,000 DATOS (1K)
# =============================================================================
print("\n--- INICIANDO GENERACIÓN: 1,000 DATOS ---")
# Como 1,000 es un volumen manejable, procesamos todo en un solo lote (batch_size=1000)
#synthetic_data_1k = generar_sinteticos_masivos(synthesizer, total_rows=1000, batch_size=1000)

# Exportamos el archivo asegurando compatibilidad con SEAL/CKKS
#exportar_para_seal(synthetic_data_1k, filename="dataset_sintetico_1k.csv")


# =============================================================================
# GENERACIÓN DE 10,000 DATOS (10K)
# =============================================================================
print("\n--- INICIANDO GENERACIÓN: 10,000 DATOS ---")
# Para 10,000, podemos dividirlo en 2 lotes de 5,000 para mantener la memoria liberada
#synthetic_data_10k = generar_sinteticos_masivos(synthesizer, total_rows=10000, batch_size=5000)

# Exportamos el archivo asegurando compatibilidad con SEAL/CKKS
#exportar_para_seal(synthetic_data_10k, filename="dataset_sintetico_10k.csv")
###########################################


# Para generar y exportar el resto de los volúmenes, simplemente descomenta y ajusta 
# las siguientes líneas según la capacidad de tu equipo de cómputo.
#
#synthetic_data_100k = generar_sinteticos_masivos(synthesizer, total_rows=100000, batch_size=10000)
#exportar_para_seal(synthetic_data_100k, "dataset_sintetico_100k.csv")
#
#synthetic_data_500k = generar_sinteticos_masivos(synthesizer, total_rows=500000, batch_size=25000)
#exportar_para_seal(synthetic_data_500k, "dataset_sintetico_500k.csv")
#
synthetic_data_1M = generar_sinteticos_masivos(synthesizer, total_rows=1000000, batch_size=50000)
exportar_para_seal(synthetic_data_1M, "dataset_sintetico_1M.csv")
```

</details>

<details>
<summary><b>💻 Haz clic aquí para desplegar el Código Fuente (Python)</b></summary>

```python
# Tu código Python aquí
print("Hola Mundo")
```

</details>
### ⚛️ Transformada Cuántica de Fourier

El módulo QFT recibe un vector de características y ajusta su dimensión a la potencia de dos más cercana.

La implementación conceptual es:

```python
def aplicar_qft_completa(data_vector):

    n_qubits = int(np.ceil(np.log2(len(data_vector))))
    target_len = 2**n_qubits

    padded_data = np.pad(
        data_vector,
        (0, target_len - len(data_vector)),
        'constant'
    )

    norm = np.linalg.norm(padded_data)

    if norm == 0:
        return np.zeros(target_len, dtype=complex)

    # Preparación del estado cuántico
    # seguido de la ejecución del circuito QFT
```
#### 🔒 Algoritmo

**Algoritmo: Quantum Fourier Transform via Amplitude Encoding with GPU Batch Processing**
<hr>

$$
\begin{array}{rl}
& \text{Input: Batch of classical feature vectors } X = \{V_1, \dots, V_B\} \text{ of length } L \\
& \text{Output: Batch of QFT-transformed vectors } X_{QFT} = \{F_1, \dots, F_B\} \\
1 & \mathcal{S} \leftarrow \text{AerSimulator}(\text{method}=\text{'statevector'}, \text{device}=\text{'GPU'}) \\
2 & Circuits \leftarrow [\,] \\
3 & Norms \leftarrow [\,] \\
4 & n_q \leftarrow \lceil \log_2(L) \rceil \hspace{6cm} \text{// Calculate required number of qubits} \\
5 & T \leftarrow 2^{n_q} \hspace{7.3cm} \text{// Target length (power of 2)} \\
6 & \textbf{for } i = 1 \textbf{ to } B \textbf{ do} \\
7 & \quad V_{pad} \leftarrow \text{Pad}(V_i, \text{length}=T, \text{value}=0) \\
8 & \quad norm_i \leftarrow \Vert V_{pad} \Vert_2 \\
9 & \quad \text{Append } norm_i \text{ to } Norms \\
10 & \quad \textbf{if } norm_i = 0 \textbf{ then} \\
11 & \quad \quad \text{Append Null to } Circuits \\
12 & \quad \quad \textbf{continue} \\
13 & \quad qc \leftarrow \text{QuantumCircuit}(n_q) \\
14 & \quad qc.\text{initialize}(V_{pad} / norm_i) \hspace{4cm} \text{// Amplitude Encoding} \\
15 & \quad \textbf{for } j = 0 \textbf{ to } n_q - 1 \textbf{ do} \\
16 & \quad \quad \textbf{for } k = 0 \textbf{ to } j - 1 \textbf{ do} \\
17 & \quad \quad \quad qc.\text{cp}(\pi / 2^{j-k}, k, j) \hspace{3cm} \text{// Controlled-Phase gate} \\
18 & \quad \quad qc.\text{h}(j) \hspace{5.2cm} \text{// Hadamard gate} \\
19 & \quad qc.\text{save\\_statevector}() \\
20 & \quad \text{Append } qc \text{ to } Circuits \\
21 & Circuits_{transpiled} \leftarrow \text{transpile}(Circuits, \mathcal{S}, \text{optimization\\_level}=0) \\
22 & \mathcal{R} \leftarrow \mathcal{S}.\text{run}(Circuits_{transpiled}).\text{result}() \\
23 & X_{QFT} \leftarrow [\,] \\
24 & idx \leftarrow 0 \\
25 & \textbf{for } i = 1 \textbf{ to } B \textbf{ do} \\
26 & \quad \textbf{if } Norms[i] = 0 \textbf{ then} \\
27 & \quad \quad F_i \leftarrow \{0\}^L \hspace{5.5cm} \text{// Handle zero-vectors} \\
28 & \quad \textbf{else} \\
29 & \quad \quad sv \leftarrow \mathcal{R}.\text{get\\_statevector}(idx) \\
30 & \quad \quad sv_{scaled} \leftarrow sv \cdot Norms[i] \hspace{3.7cm} \text{// Rescale by original norm} \\
31 & \quad \quad F_i \leftarrow \text{Real}(sv_{scaled})[0 \dots L-1] \hspace{2cm} \text{// Extract real part and truncate} \\
32 & \quad \quad idx \leftarrow idx + 1 \\
33 & \quad \text{Append } F_i \text{ to } X_{QFT} \\
34 & \textbf{return } X_{QFT}
\end{array}
$$

<hr>

### 🔐 Canal de Encriptación Homomórfica

El módulo criptográfico se implementa utilizando TenSEAL, que proporciona integraciones en Python para la encriptación homomórfica basada en Microsoft SEAL.

La configuración experimental de CKKS es:

| Parámetro | Valor |
| :--- | :--- |
| Esquema | CKKS |
| Grado del módulo polinomial | 8192 |
| Módulo de coeficientes | [60, 40, 40, 60] |
| Escala global | $2^{40}$ |
| Ranuras (slots) CKKS | 4096 |
| Parámetro experimental BKZ | 450 |

#### 🔒 Flujo de Datos CKKS

```plaintext
      Características QFT
               │
               ▼
       Codificación CKKS
               │
               ▼
       Encriptación CKKS
               │
               ▼
      Vectores Encriptados
               │
               ▼
    Operaciones Homomórficas
               │
               ▼
  Cálculo de Modelo Encriptado
               │
               ▼
    Desencriptación Controlada
               │
               ▼
           Evaluación
```
#### 🔒 Algoritmo
**Algoritmo: CKKS Homomorphic Encryption with Automatic Chunking**
<hr>

$$
\begin{array}{rl}
& \text{Input: Plaintext feature vector } V \text{ of size } n\text{, Polynomial modulus degree } N=8192\text{,} \\
& \quad \text{Coefficient modulus bit sizes } Q=[60, 40, 40, 60]\text{, Global scale } \Delta=2^{40} \\
& \text{Output: List of serialized encrypted chunks } C_{enc} = \{⟦ V ⟧_1, \dots, ⟦ V ⟧_C\} \\
1 & S \leftarrow N / 2 \hspace{7.5cm} \text{// Calculate available CKKS slots} \\
2 & \mathcal{C} \leftarrow \text{Context}(\text{CKKS}, N, Q) \\
3 & \mathcal{C}.\text{global\\_scale} \leftarrow \Delta \\
4 & \mathcal{C}.\text{generate\\_galois\\_keys}() \\
5 & C \leftarrow \lceil n / S \rceil \\
6 & C_{enc} \leftarrow [\,] \hspace{7.6cm} \text{// Initialize empty list for encrypted chunks} \\
7 & \textbf{for } i = 0 \textbf{ to } C-1 \textbf{ do} \\
8 & \quad start \leftarrow i \cdot S \\
9 & \quad end \leftarrow \min(start + S, n) \\
10 & \quad V_{chunk} \leftarrow V[start \dots end-1] \hspace{4cm} \text{// Extract data block} \\
11 & \quad ⟦ V ⟧_i \leftarrow \text{CKKS\\_Encrypt}(\mathcal{C}, V_{chunk}) \\
12 & \quad B_i \leftarrow \text{Serialize}(⟦ V ⟧_i) \hspace{4.4cm} \text{// Convert to bytes for storage} \\
13 & \quad \text{Append } B_i \text{ to } C_{enc} \\
14 & \textbf{return } C_{enc}
\end{array}
$$

<hr>

### 🧠 Modelos de Aprendizaje Automático

La arquitectura soporta varios modelos de aprendizaje automático adaptados a las características numéricas del cálculo encriptado.

#### 1. Regresión Lineal

El modelo de regresión lineal se define como:

$$\hat{y}=X\beta$$

El objetivo de optimización se basa en el Error Cuadrático Medio:

$$MSE=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2$$

La implementación experimental utiliza optimización basada en gradiente sobre la representación transformada por QFT.
##### 1.1. El Algoritmo
**Algoritmo: Hybrid Homomorphic Multiple Linear Regression Training (CKKS)**
<hr>

$$
\begin{array}{rl}
& \text{Input: Encrypted QFT-transformed feature chunks } ⟦ X ⟧ = \{⟦ X ⟧_1, \dots, ⟦ X ⟧_C\}\text{, Plaintext feature} \\
& \quad \text{matrix } X_{claro}\text{, Target vector } Y=\{y_1,\dots,y_n\}\text{, learning rate } \alpha\text{, epochs } EPOCHS\text{, total chunks } C \\
& \text{Output: Encrypted model coefficients per chunk } ⟦ \beta_0 ⟧_c, \dots, ⟦ \beta_m ⟧_c \\
1 & \textbf{for } k = 0 \textbf{ to } m \textbf{ do} \\
2 & \quad \textbf{for } c = 1 \textbf{ to } C \textbf{ do} \\
3 & \quad \quad ⟦ \beta_k ⟧_c \leftarrow ⟦ 0.01 ⟧ \\
4 & \textbf{for } epoch = 1 \textbf{ to } EPOCHS \textbf{ do} \\
5 & \quad \hat{Y} \leftarrow [\,] \\
6 & \quad \textbf{for } c = 1 \textbf{ to } C \textbf{ do} \\
7 & \quad \quad ⟦ \hat{Y} ⟧_c \leftarrow ⟦ \beta_0 ⟧_c \cdot ⟦ X_0 ⟧_c + \sum_{k=1}^{m} ⟦ \beta_k ⟧_c \cdot ⟦ X_k ⟧_c \\
8 & \quad \quad \hat{Y}_c \leftarrow \text{Decrypt}(⟦ \hat{Y} ⟧_c) \\
9 & \quad \quad \text{Append } \hat{Y}_c \text{ to } \hat{Y} \\
10 & \quad \textbf{for } j = 1 \textbf{ to } n \textbf{ do} \\
11 & \quad \quad e_j \leftarrow \hat{Y}_j - y_j \\
12 & \quad \textbf{for } k = 0 \textbf{ to } m \textbf{ do} \\
13 & \quad \quad grad_k \leftarrow \frac{1}{n} \sum_{j=1}^{n} e_j \cdot X_{claro, jk} \hspace{3cm} \text{// } X_{claro, j0} = 1 \\
14 & \quad \quad \Delta_k \leftarrow \alpha \cdot grad_k \\
15 & \quad \quad \textbf{for } c = 1 \textbf{ to } C \textbf{ do} \\
16 & \quad \quad \quad ⟦ \beta_k ⟧_c \leftarrow ⟦ \beta_k ⟧_c - \Delta_k \\
17 & \textbf{return } ⟦ \beta_0 ⟧_c, ⟦ \beta_1 ⟧_c, \dots, ⟦ \beta_m ⟧_c \quad \forall c \in \{1,\dots,C\}
\end{array}
$$

<hr>

##### 1.2. Código Fuente
##### 1.3. Resultados: Tablas
##### 1.4. Resultados: Figuras

#### 2. Regresión Logística

El modelo de regresión logística utiliza la función sigmoide:

$$\sigma(z)=\frac{1}{1+e^{-z}}$$

Dado que la evaluación directa de la función exponencial no es adecuada para el modelo aritmético de CKKS, se emplea una aproximación polinomial. La aproximación de Maclaurin de tercer grado es:

$$\sigma(z)\approx 0.5+0.25z-\frac{1}{48}z^3$$

Esta aproximación reduce la complejidad computacional asociada con las operaciones no lineales.

##### 2.1. El Algoritmo
#### 1.1. El Algoritmo (Regresión Logística)

**Algoritmo: Hybrid Homomorphic Logistic Regression Training (CKKS with Maclaurin Approximation)**
<hr>

$$
\begin{array}{rl}
& \text{Input: Encrypted QFT-transformed feature chunks } ⟦ X ⟧ = \{⟦ X ⟧_1, \dots, ⟦ X ⟧_C\}\text{, Plaintext feature} \\
& \quad \text{matrix } X_{claro}\text{, Target classes } Y=\{y_1,\dots,y_n\} \in \{0,1\}^n\text{, learning rate } \alpha\text{, epochs } EPOCHS\text{, chunks } C \\
& \text{Output: Encrypted logistic model coefficients per chunk } ⟦ \beta_0 ⟧_c, \dots, ⟦ \beta_m ⟧_c \\
1 & \textbf{for } k = 0 \textbf{ to } m \textbf{ do} \\
2 & \quad \textbf{for } c = 1 \textbf{ to } C \textbf{ do} \\
3 & \quad \quad ⟦ \beta_k ⟧_c \leftarrow ⟦ 0.01 ⟧ \\
4 & \textbf{for } epoch = 1 \textbf{ to } EPOCHS \textbf{ do} \\
5 & \quad P \leftarrow [\,] \\
6 & \quad \textbf{for } c = 1 \textbf{ to } C \textbf{ do} \\
7 & \quad \quad ⟦ Z ⟧_c \leftarrow ⟦ \beta_0 ⟧_c \cdot ⟦ X_0 ⟧_c + \sum_{k=1}^{m} ⟦ \beta_k ⟧_c \cdot ⟦ X_k ⟧_c \\
8 & \quad \quad Z_c \leftarrow \text{Decrypt}(⟦ Z ⟧_c) \\
9 & \quad \quad P_c \leftarrow 0.5 + \frac{Z_c}{4} - \frac{Z_c^3}{48} \hspace{5.5cm} \text{// Degree 3 Maclaurin Sigmoid Approximation} \\
10 & \quad \quad \text{Append } P_c \text{ to } P \\
11 & \quad \textbf{for } j = 1 \textbf{ to } n \textbf{ do} \\
12 & \quad \quad e_j \leftarrow P_j - y_j \\
13 & \quad \textbf{for } k = 0 \textbf{ to } m \textbf{ do} \\
14 & \quad \quad grad_k \leftarrow \frac{1}{n} \sum_{j=1}^{n} e_j \cdot X_{claro, jk} \hspace{4cm} \text{// } X_{claro, j0} = 1 \\
15 & \quad \quad \Delta_k \leftarrow \alpha \cdot grad_k \\
16 & \quad \textbf{for } c = 1 \textbf{ to } C \textbf{ do} \\
17 & \quad \quad \quad ⟦ \beta_k ⟧_c \leftarrow ⟦ \beta_k ⟧_c - \Delta_k \\
18 & \textbf{return } ⟦ \beta_0 ⟧_c, ⟦ \beta_1 ⟧_c, \dots, ⟦ \beta_m ⟧_c \quad \forall c \in \{1,\dots,C\}
\end{array}
$$

<hr>

##### 2.2. Código Fuente
##### 2.3. Resultados: Tablas
##### 2.4. Resultados: Figuras

#### 3. LS-SVM

La Máquina de Vectores de Soporte de Mínimos Cuadrados (LS-SVM) se formula utilizando un objetivo de optimización regularizado. La implementación utiliza una formulación lineal primal con:

- Gamma = 1.0
- Tasa de aprendizaje = 0.05
- Épocas = 150

El modelo se evalúa utilizando la precisión de clasificación.

##### 3.1. El Algoritmo

**Algoritmo: Hybrid Homomorphic Least Squares Support Vector Machine (LS-SVM) Training**
<hr>

$$
\begin{array}{rl}
& \text{Input: Encrypted QFT-transformed feature chunks } ⟦ X ⟧ = \{⟦ X ⟧_1, \dots, ⟦ X ⟧_C\}\text{, Plaintext feature} \\
& \quad \text{matrix } X_{claro}\text{, Target labels } Y=\{y_1,\dots,y_n\} \in \{-1,1\}^n\text{, learning rate } \alpha\text{, epochs } EPOCHS\text{,} \\
& \quad \text{regularization parameter } \gamma\text{, total chunks } C \\
& \text{Output: Encrypted LS-SVM model coefficients per chunk } ⟦ \beta_0 ⟧_c, \dots, ⟦ \beta_m ⟧_c \\
1 & \textbf{for } k = 0 \textbf{ to } m \textbf{ do} \\
2 & \quad \beta_{claro, k} \leftarrow 0.01 \hspace{5.5cm} \text{// Plaintext tracker for } L_2 \text{ regularization} \\
3 & \quad \textbf{for } c = 1 \textbf{ to } C \textbf{ do} \\
4 & \quad \quad ⟦ \beta_k ⟧_c \leftarrow ⟦ 0.01 ⟧ \\
5 & \textbf{for } epoch = 1 \textbf{ to } EPOCHS \textbf{ do} \\
6 & \quad S \leftarrow [\,] \\
7 & \quad \textbf{for } c = 1 \textbf{ to } C \textbf{ do} \\
8 & \quad \quad ⟦ S ⟧_c \leftarrow ⟦ \beta_0 ⟧_c \cdot ⟦ X_0 ⟧_c + \sum_{k=1}^{m} ⟦ \beta_k ⟧_c \cdot ⟦ X_k ⟧_c \\
9 & \quad \quad S_c \leftarrow \text{Decrypt}(⟦ S ⟧_c) \\
10 & \quad \quad \text{Append } S_c \text{ to } S \\
11 & \quad \textbf{for } k = 0 \textbf{ to } m \textbf{ do} \\
12 & \quad \quad \textbf{if } k = 0 \textbf{ then} \\
13 & \quad \quad \quad grad_0 \leftarrow \frac{\gamma}{n} \sum_{j=1}^{n} (S_j - y_j) \cdot X_{claro, j0} \hspace{1.8cm} \text{// Bias is unregularized} \\
14 & \quad \quad \textbf{else} \\
15 & \quad \quad \quad grad_k \leftarrow \frac{\gamma}{n} \sum_{j=1}^{n} (S_j - y_j) \cdot X_{claro, jk} + \frac{1}{n}\beta_{claro, k} \hspace{0.3cm} \text{// Apply } L_2 \text{ penalty} \\
16 & \quad \quad \Delta_k \leftarrow \alpha \cdot grad_k \\
17 & \quad \quad \textbf{for } c = 1 \textbf{ to } C \textbf{ do} \\
18 & \quad \quad \quad ⟦ \beta_k ⟧_c \leftarrow ⟦ \beta_k ⟧_c - \Delta_k \\
19 & \quad \quad \beta_{claro, k} \leftarrow \beta_{claro, k} - \Delta_k \hspace{3.4cm} \text{// Tracker Sync} \\
20 & \textbf{return } ⟦ \beta_0 ⟧_c, ⟦ \beta_1 ⟧_c, \dots, ⟦ \beta_m ⟧_c \quad \forall c \in \{1,\dots,C\}
\end{array}
$$

<hr>
##### 3.2. Código Fuente
##### 3.3. Resultados: Tablas
##### 3.4. Resultados: Figuras


#### 4. Perceptrón Multicapa (MLP)

##### 4.1. El Algoritmo

**Algoritmo: Hybrid Homomorphic Shallow MLP Training (CKKS with Quadratic Activation)**
<hr>

$$
\begin{array}{rl}
& \text{Input: Encrypted QFT feature chunks } ⟦ X ⟧ = \{⟦ X ⟧_1, \dots, ⟦ X ⟧_C\}\text{, Plaintext matrix } X_{claro}\text{,} \\
& \quad \text{Target } Y=\{y_1,\dots,y_n\}\text{, learning rate } \alpha\text{, epochs } EPOCHS\text{, hidden neurons } H \\
& \text{Output: Encrypted MLP weights } ⟦ W ⟧ \text{ (hidden), } ⟦ V ⟧ \text{ (output), } ⟦ b_{out} ⟧ \text{ (bias)} \\
1 & \text{Initialize } ⟦ W_{h,j} ⟧_c, ⟦ V_h ⟧_c, ⟦ b_{out} ⟧_c \leftarrow ⟦ 0.01 ⟧ \quad \forall h, j, c \hspace{2cm} \text{// Encrypted weights} \\
2 & \text{Initialize plaintext trackers } W_{claro}, V_{claro}, b_{claro} \leftarrow 0.01 \\
3 & \textbf{for } epoch = 1 \textbf{ to } EPOCHS \textbf{ do} \\
4 & \quad \textbf{for } h = 1 \textbf{ to } H \textbf{ do} \\
5 & \quad \quad \textbf{for } c = 1 \textbf{ to } C \textbf{ do} \\
6 & \quad \quad \quad ⟦ Z_h ⟧_c \leftarrow ⟦ W_{h,0} ⟧_c \cdot ⟦ X_0 ⟧_c + \sum_{j=1}^{m} ⟦ W_{h,j} ⟧_c \cdot ⟦ X_j ⟧_c \\
7 & \quad \quad \quad Z_{claro, h, c} \leftarrow \text{Decrypt}(⟦ Z_h ⟧_c) \\
8 & \quad \textbf{for } i = 1 \textbf{ to } n \textbf{ do} \\
9 & \quad \quad \textbf{for } h = 1 \textbf{ to } H \textbf{ do} \\
10 & \quad \quad \quad A_{i, h} \leftarrow (Z_{claro, i, h})^2 \hspace{5.5cm} \text{// Quadratic activation } f(z) = z^2 \\
11 & \quad \quad \hat{y}_i \leftarrow b_{claro} + \sum_{h=1}^H A_{i, h} \cdot V_{claro, h} \\
12 & \quad \quad e_i \leftarrow \hat{y}_i - y_i \\
13 & \quad \quad d_{salida, i} \leftarrow \frac{2}{n} e_i \\
14 & \quad grad_b \leftarrow \sum_{i=1}^n d_{salida, i} \\
15 & \quad \textbf{for } h = 1 \textbf{ to } H \textbf{ do} \\
16 & \quad \quad grad_{V_h} \leftarrow \sum_{i=1}^n A_{i, h} \cdot d_{salida, i} \\
17 & \quad \quad \textbf{for } j = 0 \textbf{ to } m \textbf{ do} \\
18 & \quad \quad \quad grad_{W_{h,j}} \leftarrow \frac{1}{n} \sum_{i=1}^n (d_{salida, i} \cdot V_{claro, h}) \cdot (2 Z_{claro, i, h}) \cdot X_{claro, ij} \hspace{0.5cm} \text{// Derivative of } z^2 \text{ is } 2z \\
19 & \quad \textbf{for } c = 1 \textbf{ to } C \textbf{ do} \\
20 & \quad \quad ⟦ b_{out} ⟧_c \leftarrow ⟦ b_{out} ⟧_c - \alpha \cdot grad_b \\
21 & \quad \quad \textbf{for } h = 1 \textbf{ to } H \textbf{ do} \\
22 & \quad \quad \quad ⟦ V_h ⟧_c \leftarrow ⟦ V_h ⟧_c - \alpha \cdot grad_{V_h} \\
23 & \quad \quad \quad \textbf{for } j = 0 \textbf{ to } m \textbf{ do} \\
24 & \quad \quad \quad \quad ⟦ W_{h,j} ⟧_c \leftarrow ⟦ W_{h,j} ⟧_c - \alpha \cdot grad_{W_{h,j}} \\
25 & \quad \text{Update } W_{claro}, V_{claro}, b_{claro} \text{ using gradients and } \alpha \\
26 & \textbf{return } ⟦ W ⟧, ⟦ V ⟧, ⟦ b_{out} ⟧
\end{array}
$$

<hr>

##### 4.2. Código Fuente
##### 4.3. Resultados: Tablas
##### 4.4. Resultados: Figuras

La arquitectura experimental del MLP es:

```plaintext
   Entrada
      │
      ▼
  3 neuronas
      │
      ▼
  4 neuronas
      │
      ▼
   1 salida
```
### 🧪 Configuración Experimental

La comparación experimental evalúa dos configuraciones principales:

**Configuración A**

```plaintext
Representación Espacial Clásica
               │
               ▼
     Aprendizaje Automático
```

**Configuración B**

```plaintext
Representación de Frecuencia QFT
               │
               ▼
       Encriptación CKKS
               │
               ▼
     Aprendizaje Automático
```

Se utilizan los mismos conjuntos de datos y condiciones experimentales controladas siempre que sea posible.

### 📈 Métricas de Evaluación

#### Métricas de Regresión

**Error Cuadrático Medio (MSE)**

$$MSE= \frac{1}{n} \sum_{i=1}^{n} (y_i-\hat{y}_i)^2$$

**Raíz del Error Cuadrático Medio (RMSE)**

$$RMSE=\sqrt{MSE}$$

**Error Absoluto Medio (MAE)**

$$MAE= \frac{1}{n} \sum_{i=1}^{n} \vert{}y_i-\hat{y}_i\vert{}$$

**Coeficiente de Determinación (R²)**

$$R^2= 1- \frac{ \sum_i(y_i-\hat{y}_i)^2 }{ \sum_i(y_i-\bar{y})^2 }$$

#### Métricas de Clasificación

Los modelos de clasificación se evalúan utilizando:

- Precisión (Accuracy)
- Exactitud (Precision)
- Sensibilidad (Recall)
- Puntuación F1 (F1-score)

### 📊 Resultados Experimentales

#### Regresión Lineal — Conjunto de Datos 1K

Resultados experimentales obtenidos utilizando CKKS:

| Métrica | CKKS |
| :--- | :--- |
| MSE | 73.911092 |
| RMSE | 8.597156 |
| MAE | 6.981118 |
| R² | 0.738903 |

#### Regresión Logística — Conjunto de Datos 1K

La aproximación polinomial de tercer grado produjo:

| Configuración | Precisión (Accuracy) |
| :--- | :--- |
| Clásica | 0.804 |
| CKKS | 0.800 |

El umbral de clasificación experimental fue: 104.2544218260

#### LS-SVM — Conjunto de Datos 1K

| Configuración | Precisión (Accuracy) |
| :--- | :--- |
| Clásica | 0.802 |
| CKKS | 0.799 |

El tiempo de ejecución clásico para el experimento de 1K fue de aproximadamente: 0.005942 segundos

---

### 🔐 Escalabilidad de CKKS

Los resultados experimentales del procesamiento CKKS fueron:

| Conjunto de Datos | Fragmentos (Chunks) | Tiempo | Memoria Aprox. |
| :--- | :--- | :--- | :--- |
| 1K | 1 | 0.031567 s | 1.28 MB |
| 10K | 3 | 0.096470 s | 3.90 MB |
| 100K | 25 | 0.753616 s | 32.64 MB |
| 500K | 123 | 3.696715 s | 160.67 MB |
| 1M | 245 | 7.371040 s | ~320 MB |

El enfoque basado en fragmentos (chunks) permite procesar conjuntos de datos más grandes utilizando múltiples vectores de texto cifrado CKKS.

### ⏱️ Escalabilidad de QFT

Los tiempos de ejecución medidos para QFT en GPU fueron:

| Conjunto de Datos | QFT GPU |
| :--- | :--- |
| 1K | 10.0168 s |
| 10K | 95.2341 s |
| 100K | 943.0058 s |
| 500K | 4625.0757 s |
| 1M | 9284.5773 s |

Estos resultados muestran el costo computacional asociado con la aplicación de la transformación QFT a medida que aumenta el tamaño del conjunto de datos.

### ⏱️ Estructura del Repositorio

```text
Aprendizaje_automatico_cifrado_homorfico/
│
├── data/
│   ├── raw/
│   │   ├── dataset_sintetico_1k.csv
│   │   ├── dataset_sintetico_10k.csv
│   │   ├── dataset_sintetico_100k.csv
│   │   ├── dataset_sintetico_500k.csv
│   │   └── dataset_sintetico_1m.csv
│   │
│   └── qft/
│       ├── dataset_qft_1k.csv
│       ├── dataset_qft_10k.csv
│       ├── dataset_qft_100k.csv
│       ├── dataset_qft_500k.csv
│       └── dataset_qft_1m.csv
│
├── src/
│   ├── generar_datos_sinteticos.py
│   ├── 01_procesamiento_qft.py
│   ├── 02_cifrado_ckks.py
│   ├── 03_01_entrenamiento_ciego.py
│   │
│   └── config/
│       └── contexto_ckks.bytes
│
├── results/
│   ├── figures/
│   ├── tables/
│   └── logs/
│
├── requirements.txt
├── README.md
└── LICENSE
```

### 🚀 Instalación

1. Clonar el Repositorio

```bash
git clone [https://github.com/USUARIO/REPOSITORIO.git](https://github.com/USUARIO/REPOSITORIO.git)
cd REPOSITORIO
```

### 2. Instalar Dependencias

```bash
pip install numpy
pip install pandas
pip install scikit-learn
pip install matplotlib
pip install qiskit==1.1.1
pip install qiskit-aer==0.15.1
pip install tenseal==0.3.16
```

### 💻 Hardware y Entorno

Los experimentos se llevaron a cabo utilizando el siguiente entorno computacional:

- **CPU:** Intel Xeon W-2123 a 3.60 GHz
- **Memoria:** 64 GB RAM
- **GPU:** NVIDIA Quadro P2000 (5 GB VRAM)
- **Sistema Operativo:** Ubuntu 24.04.4 LTS
- **CUDA:** CUDA 11.8
- **Stack de Software:** Python 3.12, Qiskit 1.1.1, Qiskit Aer 0.15.1, TenSEAL 0.3.16, Microsoft SEAL, NumPy, Pandas, Scikit-learn, Matplotlib


# 🏗️ Arquitectura del Sistema (Diagrama Generado)

```mermaid
flowchart TD
    st([Datos Clásicos]) --> op1[FASE 1: Procesamiento <br/> Datos Sintéticos, Normalización]
    op1 --> op2[FASE 2: Transformación Cuántica <br/> Codificación de Amplitud, QFT]
    op2 --> op3[FASE 3: Protección Homomórfica <br/> Frecuencia, Encriptación CKKS]
    op3 --> op4[FASE 4: Aprendizaje Automático <br/> RL, Regresión Logística, LS-SVM, MLP]
    op4 --> e([Evaluación])
```

# ⚙️ Flujo Avanzado: Arquitectura Cuántica-Clásica y Entrenamiento Cifrado

```mermaid
flowchart TD
    st([Inicio: Conjunto de Datos Sintético]) --> fase1[FASE 1: Preprocesamiento Clásico <br/> Normalización]
    fase1 --> fase2[[FASE 2: Ejecución QFT <br/> Representación de Frecuencia]]
    fase2 --> fase3[FASE 3: Generación de Contexto y Encriptación CKKS]
    fase3 --> init_ml[Inicializar Pesos del Modelo <br/> RL, LS-SVM, MLP]
    init_ml --> cond_epochs{¿Épocas < Límite <br/> Ej. 150?}
    
    cond_epochs -- Sí --> fase4_fwd[FASE 4: Operaciones Homomórficas <br/> Sumas y Multiplicaciones]
    fase4_fwd --> fase4_aprox[Aproximación Polinomial <br/> Ej. Maclaurin de 3er grado]
    fase4_aprox --> fase4_bwd[Actualización de Pesos Encriptados]
    fase4_bwd --> cond_epochs
    
    cond_epochs -- No --> desencriptar[[Desencriptación Controlada <br/> Texto Plano Aproximado]]
    desencriptar --> evaluacion[/Evaluación de Métricas <br/> MSE, R², Accuracy, F1/]
    evaluacion --> e([Fin: Resultados Experimentales])
```

# 🔄 Diagrama de Secuencia: Interacción Cuántica-Criptográfica

```mermaid
sequenceDiagram
    participant P as Preprocesamiento Clásico
    participant Q as Módulo Cuántico (QFT)
    participant C as Módulo Criptográfico (CKKS)
    participant M as Modelo Machine Learning

    P->>Q: Envía características normalizadas
    Note right of Q: Ajuste de dimensión<br/>Codificación de amplitud<br/>Ejecución QFT
    Q-->>P: Retorna representación de frecuencia (Compleja)
    
    P->>C: Envía vectores QFT
    Note right of C: Configuración BKZ<br/>Ranuras CKKS: 4096<br/>Generación de texto cifrado
    
    C->>M: Envía Vectores Encriptados
    Note right of M: Aproximación Polinomial<br/>Optimizador Basado en Gradiente
    
    M->>M: Épocas de Entrenamiento (Operaciones Homomórficas)
    M-->>P: Modelo Encriptado (Pesos Ajustados)
    
    Note left of P: Desencriptación Controlada<br/>Cálculo de MSE, R², Accuracy
```

<img width="1299" height="697" alt="Cifrado Homomórfico CKKS de Amplitudes Cuánticas" src="https://github.com/user-attachments/assets/146ec4fe-58c6-4ac4-9449-c1a5b1be69e0" />

[Solicitud a bienestar universitario.pdf](https://github.com/user-attachments/files/33113369/Solicitud.a.bienestar.universitario.pdf)


### 📝 Algoritmo 1: Preprocesamiento y Transformación QFT


<hr>

**Algoritmo 7:** Homomorphic Multiple Linear Regression Training

<hr>

$$
\begin{flalign*}
& \textsf{\textbf{Input:}} \textsf{ Encrypted QFT-transformed feature matrix } X = \{x_1, \dots, x_n\}\textsf{, target vector} & \\
& \quad Y = \{y_1, \dots, y_n\}\textsf{, learning rate } \alpha\textsf{, number of iterations } num_{iter} & \\
& \textsf{\textbf{Output:}} \textsf{ Encrypted model coefficients } \beta_0, \beta_1, \dots, \beta_m & \\
& \textsf{\textbf{1}} \quad \textsf{\textbf{Initialize }} \beta_0, \beta_1, \dots, \beta_m \leftarrow 0.01 & \\
& \textsf{\textbf{2}} \quad \textsf{\textbf{for }} i = 1 \textsf{\textbf{ to }} num_{iter} \textsf{\textbf{ do}} & \\
& \textsf{\textbf{3}} \quad \quad \textsf{\textbf{for }} j = 1 \textsf{\textbf{ to }} n \textsf{\textbf{ do}} & \\
& \textsf{\textbf{4}} \quad \quad \quad \hat{y}_j \leftarrow \beta_0 + \sum_{k=1}^{m} \beta_k \cdot x_{jk} & \\
& \textsf{\textbf{5}} \quad \quad \quad e_j \leftarrow \hat{y}_j - y_j & \\
& \textsf{\textbf{6}} \quad \quad \textsf{\textbf{for }} k = 0 \textsf{\textbf{ to }} m \textsf{\textbf{ do}} & \\
& \textsf{\textbf{7}} \quad \quad \quad grad_k \leftarrow \frac{1}{n} \sum_{j=1}^{n} e_j \cdot x_{jk} \hspace{4cm} \textsf{// } x_{j0} = 1 & \\
& \textsf{\textbf{8}} \quad \quad \quad \beta_k \leftarrow \beta_k - \alpha \cdot grad_k & \\
& \textsf{\textbf{9}} \quad \textsf{\textbf{return }} \beta_0, \beta_1, \dots, \beta_m &
\end{flalign*}
$$

<hr>

### 📊 Comparativa de Rendimiento y Parámetros del Modelo

| Métrica / Parámetro | Símbolo Matemático | Modelo Clásico (Texto Plano) | Modelo Híbrido (CKKS + QFT) |
| :--- | :---: | :---: | :---: |
| **Precisión de Predicción (MSE)** | $MSE$ | $0.0152$ | $0.0158$ |
| **Grado del Polinomio (CKKS)** | $N$ | N/A | $8192$ |
| **Niveles de Multiplicación** | $L$ | N/A | $3$ |
| **Factor de Escala** | $\Delta$ | N/A | $2^{40}$ |
| **Tiempo por Época de Entrenamiento** | $t$ | $1.2 \text{ s}$ | $45.6 \text{ s}$ |
| **Complejidad del Circuito** | $\mathcal{O}$ | $\mathcal{O}(n)$ | $\mathcal{O}(n \log n)$ |



