<p align="center">
  <h1 align="center">🧬 Project Genesis-HIV</h1>
  <p align="center"><strong>Advanced HIV Simulation Platform & In Silico Cure Research</strong></p>
</p>

<p align="center">
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python Version">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
  </a>
  <a href="https://genesis-hiv.streamlit.app/">
    <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Streamlit App">
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/field-bioinformatics-success.svg" alt="Bioinformatics">
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/stage-beta-orange.svg" alt="Stage">
  </a>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#-visual-showcase">Visuals</a> •
  <a href="#key-features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#architecture">Architecture</a>
</p>

---

## Overview

**Project Genesis-HIV** is a **high-fidelity digital twin** of the HIV virus designed to accelerate therapy testing and cure research. This platform integrates multiple scales of HIV biology—from atomic protein interactions to population-level dynamics—to enable realistic simulation of viral behavior, immune responses, and complex therapeutic interventions like CRISPR/Cas9.

> [!IMPORTANT]
> **Research Goal:** This platform is a scientific tool aimed at accelerating the discovery of a functional HIV cure through *in silico* modeling and AI-driven evolution prediction.

---

## 🚀 Visual Showcase

One of the core strengths of Project Genesis-HIV is its cinematic, scientifically accurate 3D visualization engine.

| **🔬 Molecular Animations** | **🧬 Virion Structure** |
| :--- | :--- |
| Real-time 3D animation of CRISPR/Cas9 gene editing handled natively in the browser.<br><br>![animation](https://github.com/user-attachments/assets/fb836763-136b-444e-b1b0-966cdb5c7d44) | High-fidelity surface meshes representing the viral capsid and envelope proteins.<br><br><img width="100%" alt="virion" src="https://github.com/user-attachments/assets/d3fbfadd-dd3f-4820-b83e-95427bdee7ac" /> |

---

## Key Features

### 🏢 Multi-Scale Modeling ecosystem
* **⚛️ Atomic Level:** Protein structure and molecular interactions with OpenMM integration.
* **🧬 Genetic Level:** Genomic analysis, mutation modeling, and resistance prediction.
* **🦠 Cellular Level:** Immune dynamics, viral replication, and cell-to-cell transmission.
* **👥 Population Level:** Viral load dynamics and treatment responses.
* **🏥 Clinical Level:** Treatment protocols and patient outcomes.

### 🔬 Biological Realism & "Digital Twin" Capabilities
Project Genesis-HIV acts as a computational ecosystem built on verified biological data:

* **Viral Evolution:** Realistic mutation rates based on RT (Reverse Transcriptase) fidelity data.
* **Immune Complexity:** Detailed modeling of CD4+, CD8+ T cells, B cells, and cytokine signaling.
* **Pharmacokinetics:** Accurate PK/PD models for standard-of-care antiretroviral drugs.
* **Latent Reservoir:** Advanced modeling of reservoir establishment and reactivation dynamics.

---

## Installation

### Prerequisites
* **Python 3.10+**
* **pip** package manager

### Setup Guide

1.  **Clone the repository**
    ```bash
    git clone <repository-url>
    cd project_genesis_hiv
    ```

2.  **Create Virtual Environment**
    ```bash
    # Linux/MacOS
    python -m venv hiv_env
    source hiv_env/bin/activate

    # Windows
    python -m venv hiv_env
    hiv_env\Scripts\activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

### 🖥️ Running the Dashboard

The main interface is accessed through the unified Streamlit dashboard:

```bash
streamlit run src/visualization/unified_dashboard.py
