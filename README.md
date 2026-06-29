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

---

## 📊 Data Pipeline and Processing

The workflow follows a structured progression to ensure data integrity and privacy:

1. **Synthetic Generation (`generar_datos_sinteticos.py`):** Creates the primary dataset foundation with varying scales: `dataset_sintetico_1k.csv`, `10k.csv`, `100k.csv`, `500k.csv`, and `1M.csv`.
2. **Frequency Projection (`01_procesamiento_qft.py`):** Takes a raw file (e.g., `dataset_sintetico_1k.csv`) and transforms it into `dataset_qft_1k.csv`. This script leverages Qiskit/cuQuantum to perform the QFT, mapping spatial features into an optimized frequency-domain representation.

---

## ⚙️ Homomorphic Encryption Pipeline (`02_cifrado_ckks.py`)

This module manages the cryptographic lifecycle:
* **Context Generation:** Produces `contexto_ckks.bytes`, which stores SEAL security parameters, polynomial degrees, and the public/secret key set.
* **Encryption:** Serializes the QFT-transformed datasets into `datos_cifrados_ckks.pkl`.

**Performance Note:** While our generator supports up to 1M records, the experimental validation is benchmarked at **1,000 vectors**. This choice is critical as encrypting and performing arithmetic on million-vector tensors requires massive distributed memory clusters that exceed the capacity of standard high-performance workstations.

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
