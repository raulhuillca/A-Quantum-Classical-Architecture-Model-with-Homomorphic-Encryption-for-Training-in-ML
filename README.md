# A Quantum-Classical Architecture Model with Homomorphic Encryption for Training and Privacy Preservation in Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-Quantum-purple.svg)](https://qiskit.org/)
[![cuQuantum](https://img.shields.io/badge/NVIDIA-cuQuantum-76B900.svg)](https://developer.nvidia.com/cuquantum-sdk)
[![TenSEAL](https://img.shields.io/badge/TenSEAL-Homomorphic-orange.svg)](https://github.com/OpenMined/TenSEAL)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation and source code for the research paper: **"A Quantum-Classical Architecture Model with Homomorphic Encryption for Training and Privacy Preservation in Machine Learning"**.

The project proposes a hybrid architecture integrating the Quantum Fourier Transform (QFT) and the CKKS homomorphic encryption scheme. This allows for the secure training of machine learning models on fully encrypted data, preserving statistical utility and ensuring absolute data privacy in distributed cloud environments.

---

## 📌 Table of Contents
* [Mathematical Foundations](#mathematical-foundations)
* [Adapted Machine Learning Models](#adapted-machine-learning-models)
* [Hardware and Environment](#hardware-and-environment)
* [Repository Structure](#repository-structure)
* [Usage and Execution](#usage-and-execution)
* [Experimental Results](#experimental-results)
* [Citation and Authors](#citation-and-authors)

---

## 📐 Mathematical Foundations

### 1. Quantum Feature Extraction (QFT)
To capture global correlations hidden in the spatial domain, data is mapped into quantum states. The QFT operates over a Hilbert space of dimension $N = 2^n$ shifting the base states according to:

$$\text{QFT} \ket{k} = \frac{1}{\sqrt{N}} \sum_{j=0}^{N-1} e^{2\pi i k j / N} \ket{j}$$

### 2. Homomorphic Shielding (CKKS)
The frequency projection is encrypted using the CKKS (Cheon-Kim-Kim-Song) scheme via Microsoft SEAL. CKKS relies on the Ring Learning With Errors (RLWE) problem over the cyclotomic polynomial ring:

$$R_q = \mathbb{Z}_q[x]/(x^N + 1)$$

Values are represented as $\text{Enc}(m) = m + e$, where $e$ is the security noise. The architecture ensures that the noise budget remains within the modulus $q$ limits during all training epochs.

---

## 🧠 Adapted Machine Learning Models

Due to the strict arithmetic constraints of FHE (supporting only polynomial additions and multiplications), standard algorithms were mathematically adapted to avoid exhaustive bootstrapping:

* **Homomorphic Logistic Regression:** The non-polynomial sigmoid function is replaced by a 3rd-degree Maclaurin series approximation:
  $$h(z) \approx 0.5 + 0.25z - \frac{1}{48}z^3$$
* **LS-SVM (Polynomial Kernel):** The standard Hinge loss is replaced with a continuous least-squares residual optimization, incorporating Tikhonov ($L_2$) regularization to prevent tensor overflow:
  $$\nabla \beta_k = \frac{1}{N} \sum (\llbracket e_j \rrbracket \cdot \llbracket x_{jk} \rrbracket) + \lambda \llbracket \beta_k \rrbracket$$
* **Shallow Neural Network (MLP):** Utilizes a quadratic activation function $f(x) = x^2$ to limit the multiplicative depth strictly to 3 levels, supported by an interactive hybrid backpropagation scheme.

---

## 💻 Hardware and Environment

To guarantee reproducibility, the experiments and simulations were executed on the following workstation setup:

* **CPU:** Dual Intel® Xeon® W-2123 @ 3.60 GHz (4 cores with hyper-threading per processor)
* **RAM:** 64 GB DDR4 ECC
* **GPU:** NVIDIA Quadro P2000 (Utilized for cuQuantum QFT simulation acceleration)
* **Frameworks:** Python, Qiskit, NVIDIA cuQuantum, PyTorch, and TenSEAL.

---

## 📂 Repository Structure

The project is modularized to separate quantum feature extraction, cryptographic context generation, and homomorphic training.

```text
Aprendizaje_automatico_cifrado_homorfico/
│
├── data/
│   ├── raw/
│   │   └── dataset_sintetico_1k.csv       # Raw spatial data (Plaintext)
│   └── processed/
│       └── datos_cifrados_ckks.pkl        # Encrypted QFT tensors (Generated)
│
├── src/
│   ├── config/
│   │   └── contexto_ckks.bytes            # SEAL cryptographic parameters
│   ├── 01_qft_transform.py                # Maps spatial data to frequency domain (Qiskit)
│   ├── 02_ckks_encryption.py              # Encrypts projected tensors using TenSEAL
│   ├── 03_entrenamiento_ciego.py          # Executes blind gradient descent (LR, SVM, MLP)
│   └── utils.py                           # Helper functions (Metrics, Plotting)
│
├── notebooks/
│   └── experiment_analysis.ipynb          # Jupyter notebook for interactive visualization
│
├── imagenes/
│   ├── rl_grafico_convergencia.pdf        # Gradient convergence plots
│   ├── rl_grafico_de_dispersion.pdf       # Predictive scatter plots
│   └── rl_grafico_de_residuos.pdf         # Homoscedasticity residual analysis
│
├── requirements.txt                       # Python dependencies
└── README.md                              # Project documentation
