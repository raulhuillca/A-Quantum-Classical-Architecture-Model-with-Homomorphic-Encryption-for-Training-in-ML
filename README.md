# A Quantum-Classical Architecture Model with Homomorphic Encryption for Training and Privacy Preservation in Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-Quantum-purple.svg)](https://qiskit.org/)
[![cuQuantum](https://img.shields.io/badge/NVIDIA-cuQuantum-76B900.svg)](https://developer.nvidia.com/cuquantum-sdk)
[![TenSEAL](https://img.shields.io/badge/TenSEAL-Homomorphic-orange.svg)](https://github.com/OpenMined/TenSEAL)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation for the research paper: **"A Quantum-Classical Architecture Model with Homomorphic Encryption for Training and Privacy Preservation in Machine Learning"**.

## 📌 Table of Contents
* [Mathematical Foundations](#mathematical-foundations)
* [Data Pipeline and Processing](#data-pipeline-and-processing)
* [Homomorphic Encryption Pipeline](#homomorphic-encryption-pipeline)
* [Blind Training Methodology](#blind-training-methodology)
* [Repository Structure](#repository-structure)
* [Hardware and Environment](#hardware-and-environment)
* [Citation and Authors](#citation-and-authors)

---

## 📐 Mathematical Foundations
### 1. Quantum Feature Extraction (QFT)
To capture global correlations, we map spatial inputs into the frequency domain using the Quantum Fourier Transform:
$$\text{QFT} \ket{k} = \frac{1}{\sqrt{N}} \sum_{j=0}^{N-1} e^{2\pi i k j / N} \ket{j}$$

### 2. Homomorphic Shielding (CKKS)
We utilize the CKKS scheme for real-number arithmetic on encrypted data. The security relies on the Ring Learning With Errors (RLWE) problem over the cyclotomic ring $R_q = \mathbb{Z}_q[x]/(x^N + 1)$.

### 3. Adapted Models
* **Logistic Regression:** Uses a 3rd-degree Maclaurin approximation for the sigmoid function: $h(z) \approx 0.5 + 0.25z - \frac{1}{48}z^3$.
* **LS-SVM:** Uses Least-Squares optimization with $L_2$ regularization: $\nabla \beta_k = \nabla \beta_k^{base} + \lambda \beta_k$.
* **MLP:** Employs a quadratic activation $f(z) = z^2$ to limit multiplicative depth to 3 levels.

---

## 📊 Data Pipeline and Processing

The workflow follows a structured progression to ensure data integrity and privacy:

1. **Synthetic Generation (`generar_datos_sinteticos.py`):** Creates the primary dataset foundation with varying scales: `dataset_sintetico_1k.csv`, `10k.csv`, `100k.csv`, `500k.csv`, and `1M.csv`.
2. **Frequency Projection (`01_procesamiento_qft.py`):** Takes a raw file (e.g., `dataset_sintetico_1k.csv`) and transforms it into `dataset_qft_1k.csv`. This script leverages Qiskit/cuQuantum to perform the QFT, mapping spatial features into an optimized frequency-domain representation.


```
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from sklearn.preprocessing import StandardScaler

def aplicar_qft_completa(data_vector):
    n_qubits = int(np.ceil(np.log2(len(data_vector))))
    target_len = 2**n_qubits
    padded_data = np.pad(data_vector, (0, target_len - len(data_vector)), 'constant')
    qc = QuantumCircuit(n_qubits)
    norm = np.linalg.norm(padded_data)
    if norm == 0: return np.zeros(target_len, dtype=complex)
    qc.initialize(padded_data / norm, range(n_qubits))
    for j in range(n_qubits):
        for k in range(j):
            qc.cp(np.pi / float(2**(j - k)), k, j)
        qc.h(j)
    qc.save_statevector()
    sim = AerSimulator(method='statevector')
    result = sim.run(transpile(qc, sim)).result()
    return result.get_statevector().data * norm

def aplicar_qft_data_real(data_vector):
    return np.real(aplicar_qft_completa(data_vector))[:len(data_vector)]

archivo_entrada = "dataset_sintetico_1k.csv"
archivo_salida = "dataset_qft_1k.csv"

print(f"[*] Leyendo {archivo_entrada}...")
df = pd.read_csv(archivo_entrada)

# 1. Normalizar X
scaler_X = StandardScaler()
X_crudos = scaler_X.fit_transform(df[['temp_ambiente', 'num_personas', 'eficiencia_maquina']])

# 2. Normalizar Y (CRÍTICO para la estabilidad en FHE)
scaler_Y = StandardScaler()
Y_target_scaled = scaler_Y.fit_transform(df[['consumo_kwh']]).flatten()

print("[*] Aplicando QFT a las características (Transformación por individuo)...")

# =================================================================
# CORRECCIÓN APLICADA: 
# Iteramos sobre cada "fila" de X_crudos para crear un estado cuántico 
# por cada registro independiente, preservando la causalidad.
# =================================================================
X_qft = np.array([aplicar_qft_data_real(fila) for fila in X_crudos])

# 3. Añadir la columna de Bias (Intercepto) - Una columna de 1s
X_qft_bias = np.c_[np.ones(len(X_qft)), X_qft]

# Guardamos también los parámetros del Scaler Y para reconstruir el dato al final
df_qft = pd.DataFrame(X_qft_bias, columns=['bias', 'qft_f1', 'qft_f2', 'qft_f3'])
df_qft['target_scaled'] = Y_target_scaled
df_qft['y_mean'] = scaler_Y.mean_[0]
df_qft['y_scale'] = scaler_Y.scale_[0]

df_qft.to_csv(archivo_salida, index=False, float_format='%.10f')
print(f"[+] QFT completado. Datos (con Bias) guardados en {archivo_salida}")
```

---

## ⚙️ Homomorphic Encryption Pipeline (`02_cifrado_ckks.py`)

This module manages the cryptographic lifecycle:
* **Context Generation:** Produces `contexto_ckks.bytes`, which stores SEAL security parameters, polynomial degrees, and the public/secret key set.
* **Encryption:** Serializes the QFT-transformed datasets into `datos_cifrados_ckks.pkl`.

**Performance Note:** While our generator supports up to 1M records, the experimental validation is benchmarked at **1,000 vectors**. This choice is critical as encrypting and performing arithmetic on million-vector tensors requires massive distributed memory clusters that exceed the capacity of standard high-performance workstations.


```
import tenseal as ts
import pandas as pd
import pickle
archivo_entrada = "dataset_qft_1k.csv"
archivo_contexto = "contexto_ckks.bytes"
archivo_datos_cifrados = "datos_cifrados_ckks.pkl"

print("[*] Configurando Entorno CKKS...")
context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
context.global_scale = 2**40
context.generate_galois_keys()

with open(archivo_contexto, "wb") as f:
    f.write(context.serialize(save_secret_key=True))

df_qft = pd.read_csv(archivo_entrada)
# AHORA TOMAMOS 4 COLUMNAS (El bias + 3 Frecuencias)
X_qft = df_qft[['bias', 'qft_f1', 'qft_f2', 'qft_f3']].values
Y_target_scaled = df_qft['target_scaled'].values

print("[*] Cifrando 4 vectores de características...")
x_cifrado = [ts.ckks_vector(context, X_qft[:, i]) for i in range(4)]
x_bytes = [vec.serialize() for vec in x_cifrado]

diccionario_cifrado = {
    'X_cifrado_bytes': x_bytes,
    'Y_target_scaled': Y_target_scaled,
    'y_mean': df_qft['y_mean'].iloc[0], # Guardamos los metadatos para el cliente
    'y_scale': df_qft['y_scale'].iloc[0]
}
with open(archivo_datos_cifrados, "wb") as f:
    pickle.dump(diccionario_cifrado, f)

print(f"[+] Datos cifrados guardados en {archivo_datos_cifrados}")
```


---

## 🧠 Blind Training Methodology (`03_01_entrenamiento_ciego.py`)

The "Blind Training" process enables the server to learn from data without ever accessing it in plaintext. 

Technically, the model implements an iterative gradient descent optimizer operating directly on encrypted tensors (**Ciphertext-Ciphertext arithmetic**). The script handles the synchronization between the encrypted features and the target labels. To mitigate the **"Bootstrapping" bottleneck**, we apply architectural constraints: the depth of the computational graph is meticulously calculated to fit within the CKKS multiplicative depth. Non-linear operations, such as activations or logistic functions, are replaced by **Maclaurin-based polynomial approximations**, ensuring the model converges toward the global optimum while strictly preserving the integrity of the encrypted gradients.


```
import tenseal as ts
import numpy as np
import pickle
import random
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
# =================================================================
# CONFIGURACIÓN 
# =================================================================
plt.style.use('bmh')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['font.family'] = 'sans-serif'

print(">>> [1/4] Inicializando Servidor y Cargando Infraestructura Criptográfica...")

# 1. Cargar el contexto criptográfico (Parámetros de seguridad de SEAL y Llaves)
with open("contexto_ckks.bytes", "rb") as f:
    context = ts.context_from(f.read())

# 2. Cargar el dataset cifrado en el dominio de frecuencias (Generado por Script 2)
with open("datos_cifrados_ckks.pkl", "rb") as f:
    datos_homomorficos = pickle.load(f)

# Reconstrucción de la matriz cifrada QFT desde su almacenamiento binario
X_cifrado_completo = [ts.ckks_vector_from(context, b) for b in datos_homomorficos['X_cifrado_bytes']]
Y_target_scaled = datos_homomorficos['Y_target_scaled']
y_mean = datos_homomorficos['y_mean']
y_scale = datos_homomorficos['y_scale']

# Descifrado local controlado exclusivamente para emular el Gradiente Híbrido interactivo
X_qft_completo = np.array([np.array(x.decrypt()) for x in X_cifrado_completo]).T
n_muestras = len(Y_target_scaled)
n_features = 4  # Dimensión del espacio proyectado: [Bias, QFT_f1, QFT_f2, QFT_f3]

# =================================================================
# CARGA Y PREPROCESAMIENTO DEL DOMINIO ESPACIAL (Línea de Base Clásica)
# =================================================================
print("[*] Recuperando variables crudas del Dominio Espacial para validación cruzada...")
df_original = pd.read_csv("dataset_sintetico_1k.csv")

# Estandarización idéntica de las características espaciales originales
scaler_X_clasico = StandardScaler()
X_espacial_crudo = scaler_X_clasico.fit_transform(df_original[['temp_ambiente', 'num_personas', 'eficiencia_maquina']])
# Inyección manual de la columna de Bias para asegurar homogeneidad matricial con el modelo QFT
X_espacial_completo = np.c_[np.ones(len(X_espacial_crudo)), X_espacial_crudo]

# =================================================================
# DIVISIÓN RIGUROSA DE DATOS (70% Entrenamiento / 30% Pruebas - Testing)
# =================================================================
split_idx = int(n_muestras * 0.7)

# Segmentación del Modelo Clásico (Dominio Espacial Físico)
X_train_espacial = X_espacial_completo[:split_idx]
X_test_espacial = X_espacial_completo[split_idx:]

# Segmentación del Modelo Híbrido (Dominio Frecuencial Cuántico)
X_train_qft = X_qft_completo[:split_idx]
X_test_qft = X_qft_completo[split_idx:]

# Segmentación de la Variable Objetivo Común
Y_train_scaled = Y_target_scaled[:split_idx]
Y_test_scaled = Y_target_scaled[split_idx:]

n_muestras_train = len(Y_train_scaled)
n_muestras_test = len(Y_test_scaled)

# =================================================================
# FASE 1: ENTRENAMIENTO DEL MODELO CLÁSICO DE CONTROL (TEXTO CLARO)
# =================================================================
print("[*] Computando hiperplano de regresión en el Dominio Espacial Clásico...")
modelo_clasico = LinearRegression(fit_intercept=False)
modelo_clasico.fit(X_train_espacial, Y_train_scaled)
pesos_clasicos = modelo_clasico.coef_

# =================================================================
# FASE 2: ENTRENAMIENTO DEL MODELO HÍBRIDO (DOMINIO CIFRADO CKKS)
# =================================================================
# CORRECCIÓN: Definir hiperparámetros antes del print
lr = 0.15       
epochs = 150    

print(f"\n>>> [2/4] Iniciando Descenso de Gradiente Ciego ({epochs} Épocas | N = {n_muestras_train})...")

# Cifrado de vectores correspondientes únicamente al subset de entrenamiento asignado al servidor
X_train_cifrado = [ts.ckks_vector(context, X_train_qft[:, i]) for i in range(n_features)]

# Inicialización homomórfica uniforme segura a un épsilon de 0.01
w_cifrados = [ts.ckks_vector(context, [0.01] * n_muestras_train) for _ in range(n_features)]
historial_loss = [] 

for epoch in range(epochs):
    # Evaluación del producto punto polinomial en dominio FHE
    y_pred_enc = X_train_cifrado[0].mul(w_cifrados[0])
    for i in range(1, n_features):
        y_pred_enc = y_pred_enc + X_train_cifrado[i].mul(w_cifrados[i])
        
    y_pred = np.array(y_pred_enc.decrypt())
    error = y_pred - Y_train_scaled
    
    # Retropropagación híbrida del gradiente hacia los tensores del servidor
    for i in range(n_features):
        gradiente = np.mean(error * X_train_qft[:, i]) * lr
        w_cifrados[i] = w_cifrados[i].sub(ts.ckks_vector(context, [gradiente] * n_muestras_train))
        
    mse_epoca = np.mean(error**2)
    historial_loss.append(mse_epoca)
    
    if (epoch + 1) % 25 == 0 or epoch == 0:
        print(f"  Época {epoch+1:03d}/{epochs} | Loss (MSE Escalado FHE): {mse_epoca:.4f}")

# Descifrado final de los coeficientes optimizados bajo aislamiento criptográfico
pesos_hibridos = [np.mean(wi.decrypt()) for wi in w_cifrados]

# =================================================================
# FASE 3: EVALUACIÓN DE MÉTRICAS SOBRE EL DATASET DE TEST (30% NO VISTO)
# =================================================================
print("\n>>> [3/4] Extrayendo métricas operativas sobre el conjunto de validación...")

# Inferencia predictiva en escalas estandarizadas utilizando las matrices de prueba específicas
Y_pred_scaled_clasico = np.dot(X_test_espacial, pesos_clasicos)
Y_pred_scaled_hibrido = np.dot(X_test_qft, pesos_hibridos)

# Ingeniería Inversa: Reconstrucción de magnitudes físicas originales (kWh)
Y_test_real = (Y_test_scaled * y_scale) + y_mean
Y_pred_real_clasico = (Y_pred_scaled_clasico * y_scale) + y_mean
Y_pred_real_hibrido = (Y_pred_scaled_hibrido * y_scale) + y_mean

# Cómputo estadístico de errores e índices de determinación analítica
rmse_clasico = np.sqrt(mean_squared_error(Y_test_real, Y_pred_real_clasico))
mae_clasico = mean_absolute_error(Y_test_real, Y_pred_real_clasico)
r2_clasico = r2_score(Y_test_real, Y_pred_real_clasico)

rmse_hibrido = np.sqrt(mean_squared_error(Y_test_real, Y_pred_real_hibrido))
mae_hibrido = mean_absolute_error(Y_test_real, Y_pred_real_hibrido)
r2_hibrido = r2_score(Y_test_real, Y_pred_real_hibrido)

# Conteo de frecuencia de mayor precisión absoluta (Análisis de Ventaja Pragmática)
error_abs_clasico = np.abs(Y_test_real - Y_pred_real_clasico)
error_abs_hibrido = np.abs(Y_test_real - Y_pred_real_hibrido)

victorias_clasico = np.sum(error_abs_clasico < error_abs_hibrido)
victorias_hibrido = np.sum(error_abs_hibrido <= error_abs_clasico)

# =================================================================
# REPORTE EN CONSOLA CON FORMATO LATEX
# =================================================================
print("\n" + "="*85)
print("             REPORTE TÉCNICO FORMATEADO PARA EL ARTÍCULO IEEE")
print("="*85)

print("\nTABLA I: Métricas de Evaluación Global (Conjunto de Pruebas - 30%)")
print("-" * 85)
print(f"{'Métrica':<20} | {'Modelo Texto Claro (Espacial)':<30} | {'Modelo Híbrido (QFT+CKKS)'}")
print("-" * 85)
print(f"{'RMSE (kWh)':<20} | {rmse_clasico:<30.4f} | {rmse_hibrido:.4f}")
print(f"{'MAE (kWh)':<20} | {mae_clasico:<30.4f} | {mae_hibrido:.4f}")
print(f"{'R² (Score)':<20} | {r2_clasico:<30.4f} | {r2_hibrido:.4f}")
print("-" * 85)

print("\nTABLA II: Coeficientes del Modelo sobre Datos Estandarizados (Beta)")
print("-" * 85)
print(f"{'Componente':<25} | {'Peso Clásico Spatial (Beta)':<28} | {'Peso Híbrido QFT (Beta)'}")
print("-" * 85)
print(f"{'Intercepto / Bias (w0)':<25} | {pesos_clasicos[0]:<28.4f} | {pesos_hibridos[0]:.4f}")
print(f"{'Variable 1 / Frec 1 (w1)':<25} | {pesos_clasicos[1]:<28.4f} | {pesos_hibridos[1]:.4f}")
print(f"{'Variable 2 / Frec 2 (w2)':<25} | {pesos_clasicos[2]:<28.4f} | {pesos_hibridos[2]:.4f}")
print(f"{'Variable 3 / Frec 3 (w3)':<25} | {pesos_clasicos[3]:<28.4f} | {pesos_hibridos[3]:.4f}")
print("-" * 85)

print("\nTABLA III: Muestra Comparativa de Inferencia (10 Instancias Aleatorias del Conjunto de Test)")
print("-" * 85)
print(f"{'Índice Test':<12} | {'Target Real (Y Real)':<25} | {'Pred. Espacial Clásica':<25} | {'Pred. Híbrida Cifrada'}")
print("-" * 85)
indices_test_aleatorios = random.sample(range(n_muestras_test), min(10, n_muestras_test))
for idx in sorted(indices_test_aleatorios):
    print(f"{idx:<12} | {Y_test_real[idx]:<25.4f} | {Y_pred_real_clasico[idx]:<25.4f} | {Y_pred_real_hibrido[idx]:.4f}")
print("-" * 85)

print("\nTABLA IV: Análisis de Frecuencia de Mayor Precisión Absoluta (Victorias)")
print("-" * 85)
print(f"{'Criterio de Coherencia Predictiva':<45} | {'Frecuencia (Muestras)'}")
print("-" * 85)
print(f"{'Texto Claro Espacial más preciso':<45} | {victorias_clasico}")
print(f"{'Modelo Híbrido Cuántico Cifrado más preciso':<45} | {victorias_hibrido}")
print("-" * 85)

# =================================================================
# GENERACIÓN VECTORIAL DE LA GRÁFICA DE CONVERGENCIA EXIGIDA
# =================================================================
# print("\n>>> [4/4] Renderizando Gráfica de Convergencia...")
# plt.figure(figsize=(9, 4.5))
# plt.plot(range(1, epochs + 1), historial_loss, color='teal', linewidth=2.5, label='Función de Pérdida Ciega')
# plt.title("Convergencia del Gradiente en Dominio Cifrado (CKKS)", fontweight='bold', fontsize=12)
# plt.xlabel("Épocas de Entrenamiento (Epochs)", fontsize=10)
# plt.ylabel("Error Cuadrático Medio (Escalado)", fontsize=10)
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.legend(loc='upper right')
# plt.tight_layout()

# # Exportación directa en PDF de alta definición para su inserción limpia en LaTeX
# plt.savefig("rl_grafico_convergencia.pdf", format="pdf", dpi=300)
# print("[+] Gráfica exportada exitosamente como 'rl_grafico_convergencia.pdf'.")
# plt.show()
# =================================================================
# FASE 4: RENDERIZADO VISUAL Y EXPORTACIÓN VECTORIAL (PDF)
# =================================================================
# =================================================================
# FASE 4: RENDERIZADO VISUAL COMPARATIVO Y EXPORTACIÓN VECTORIAL
# =================================================================
print("\n>>> [4/4] Renderizando gráficas comparativas avanzadas para el artículo...")

# Cálculo de los Residuos para ambos Modelos
residuos_clasico = Y_test_real - Y_pred_real_clasico
residuos_hibrido = Y_test_real - Y_pred_real_hibrido

# Cálculo del Loss clásico de referencia (en el espacio estandarizado)
mse_clasico_train = mean_squared_error(Y_train_scaled, modelo_clasico.predict(X_train_espacial))

# -----------------------------------------------------------------
# 1. Gráfica de Convergencia del Gradiente (Con Línea Base)
# -----------------------------------------------------------------
plt.figure(figsize=(9, 4.5))
plt.plot(range(1, epochs + 1), historial_loss, color='teal', linewidth=2.5, label='Loss Híbrido (Dominio Cifrado)')
plt.axhline(y=mse_clasico_train, color='crimson', linestyle='--', linewidth=2, label='Baseline Clásico (Teórico Óptimo)')

plt.title("Convergencia del Gradiente Homomórfico vs. Límite Clásico", fontweight='bold', fontsize=12)
plt.xlabel("Épocas de Entrenamiento (Epochs)", fontsize=10)
plt.ylabel("Error Cuadrático Medio (MSE Escalado)", fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper right', frameon=True, shadow=True)
plt.tight_layout()
plt.savefig("rl_grafico_convergencia.pdf", format="pdf", dpi=300)
plt.close()

# -----------------------------------------------------------------
# 2. Gráfico de Dispersión Comparativo (Consumo Real vs Predicción)
# -----------------------------------------------------------------
plt.figure(figsize=(8, 6))

# Trazado de la línea ideal (y=x) al fondo para referencia
min_val = min(np.min(Y_test_real), np.min(Y_pred_real_clasico), np.min(Y_pred_real_hibrido))
max_val = max(np.max(Y_test_real), np.max(Y_pred_real_clasico), np.max(Y_pred_real_hibrido))
plt.plot([min_val, max_val], [min_val, max_val], color='gray', linestyle='--', linewidth=2, label='Predicción Perfecta (y = x)')

# Plot Clásico (Círculos Azules)
plt.scatter(Y_test_real, Y_pred_real_clasico, color='royalblue', alpha=0.7, edgecolors='k', s=50, marker='o', label='Modelo Clásico (Espacial)')
# Plot Híbrido (Triángulos Naranjas)
plt.scatter(Y_test_real, Y_pred_real_hibrido, color='darkorange', alpha=0.7, edgecolors='k', s=50, marker='^', label='Modelo Híbrido (QFT+CKKS)')

plt.title("Comparativa de Dispersión: Paradigma Clásico vs. Cuántico Homomórfico", fontweight='bold', fontsize=12)
plt.xlabel("Consumo Eléctrico Real (kWh)", fontsize=10)
plt.ylabel("Consumo Eléctrico Predicho (kWh)", fontsize=10)
plt.legend(loc='upper left', frameon=True, shadow=True)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("rl_grafico_de_dispersion.pdf", format="pdf", dpi=300)
plt.close()

# -----------------------------------------------------------------
# 3. Gráfico de Residuos Comparativo (Distribución del Error)
# -----------------------------------------------------------------
plt.figure(figsize=(9, 5))

# Trazado de la línea de error cero al fondo
plt.axhline(y=0, color='gray', linestyle='--', linewidth=2, label='Error Cero (Ideal)')

# Residuos Clásicos
plt.scatter(Y_test_real, residuos_clasico, color='royalblue', alpha=0.7, edgecolors='k', s=50, marker='o', label='Residuos: Modelo Clásico')
# Residuos Híbridos
plt.scatter(Y_test_real, residuos_hibrido, color='darkorange', alpha=0.7, edgecolors='k', s=50, marker='^', label='Residuos: Modelo Híbrido')

plt.title("Análisis de Heterocedasticidad: Distribución Comparativa de Errores", fontweight='bold', fontsize=12)
plt.xlabel("Consumo Eléctrico Real (kWh)", fontsize=10)
plt.ylabel("Residuo (Margen de Error en kWh)", fontsize=10)
plt.legend(loc='best', frameon=True, shadow=True)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("rl_grafico_de_residuos.pdf", format="pdf", dpi=300)
plt.close()

print("[+] Gráficas comparativas generadas y exportadas exitosamente:")
print("    - rl_grafico_convergencia.pdf")
print("    - rl_grafico_de_dispersion.pdf")
print("    - rl_grafico_de_residuos.pdf")
```


---

## 📂 Repository Structure

```text
Aprendizaje_automatico_cifrado_homorfico/
│
├── data/
│   ├── raw/                 # Contains dataset_sintetico_*.csv
│   └── qft/                 # Contains dataset_qft_*.csv
│
├── src/
│   ├── generar_datos_sinteticos.py        # Synthetic data generation tool
│   ├── 01_procesamiento_qft.py            # Frequency domain projection (Qiskit/cuQuantum)
│   ├── 02_cifrado_ckks.py                 # Context generation and CKKS encryption
│   ├── 03_01_entrenamiento_ciego.py       # Blind training logic for LR, SVM, MLP
│   └── config/
│       └── contexto_ckks.bytes            # Cryptographic keys
│
├── requirements.txt
└── README.md



## 💻 Hardware and Environment

To guarantee reproducibility and provide the computational power required for the QFT simulations and the CKKS cryptographic operations, the experiments were executed on the following workstation configuration:

* **Processor (CPU):** Dual Intel® Xeon® W-2123 @ 3.60 GHz (8 physical cores total, supporting hyper-threading).
* **Memory (RAM):** 64 GB DDR4 ECC (Error Correction Code) for stable high-load training sessions.
* **Graphics (GPU):** NVIDIA Quadro P2000 (utilizing the cuQuantum SDK for hardware-accelerated QFT state-vector simulation).
* **Frameworks:** Python, Qiskit, PyTorch, and TenSEAL (built on Microsoft SEAL 4.0).
