# 📰 Multi-Model Fake News Detection System

A comprehensive machine learning pipeline designed to identify and classify misinformation across various datasets (such as **GossipCop**). This project features a multi-model architecture, a web interface for real-time inference, and model explainability components.

---

## 🚀 Key Features

* **Multi-Model Architecture:** Supports training and evaluation of various NLP models to compare performance.
* **Interpretability:** Includes an `explain.py` module to understand why certain news items are flagged as fake.
* **Full-Stack Deployment:** Integrated with a **Flask** web application (`app.py`) for real-time user interaction.
* **Hardware Optimized:** Specialized scripts for GPU diagnostics and memory management (`clear_gpu.py`, `gpu_diagnostic.py`).
* **End-to-End Pipeline:** Covers everything from raw data preparation to confusion matrix visualization.

---

## 📁 Project Structure

| File/Folder | Description |
| :--- | :--- |
| `data/` | Dataset storage and sample data (e.g., `gossipcop_fake.csv`). |
| `models/` | Architecture definitions and saved weights. |
| `app.py` | Flask application for the web-based UI. |
| `explain.py` | Logic for model interpretability and transparency. |
| `evaluate.py` | Performance metrics and evaluation logic. |
| `confusion_matrix.png` | Visual breakdown of classification accuracy. |

---

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/Multi-Model-Fake-News-Detection.git
   cd Multi-Model-Fake-News-Detection
