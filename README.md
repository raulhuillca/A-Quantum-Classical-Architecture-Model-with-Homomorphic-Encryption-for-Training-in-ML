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
```


---

## 🧠 Blind Training Methodology (`03_01_entrenamiento_ciego.py`)

The "Blind Training" process enables the server to learn from data without ever accessing it in plaintext. 

Technically, the model implements an iterative gradient descent optimizer operating directly on encrypted tensors (**Ciphertext-Ciphertext arithmetic**). The script handles the synchronization between the encrypted features and the target labels. To mitigate the **"Bootstrapping" bottleneck**, we apply architectural constraints: the depth of the computational graph is meticulously calculated to fit within the CKKS multiplicative depth. Non-linear operations, such as activations or logistic functions, are replaced by **Maclaurin-based polynomial approximations**, ensuring the model converges toward the global optimum while strictly preserving the integrity of the encrypted gradients.



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
