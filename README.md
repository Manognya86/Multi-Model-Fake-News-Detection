Multi-Model Fake News Detection System
A comprehensive machine learning pipeline designed to identify and classify misinformation across various datasets (such as GossipCop). This project features a multi-model architecture, a web interface for real-time inference, and model explainability components.

🚀 Key Features
Multi-Model Architecture: Supports training and evaluation of various NLP models to compare performance.

Interpretability: Includes an explain.py module to understand why certain news items are flagged as fake.

Full-Stack Deployment: Integrated with a Flask web application (app.py) for user interaction.

Hardware Optimized: Includes specialized scripts for GPU diagnostics and memory management (clear_gpu.py, gpu_diagnostic.py).

End-to-End Pipeline: Covers everything from raw data preparation to confusion matrix visualization.

📂 Project Structure
data/ & gossipcop_fake.csv: Dataset storage and sample data.

models/ & saved_models/: Architecture definitions and trained weights.

app.py: Flask application for the web-based UI.

explain.py: Logic for model interpretability and transparency.

evaluate.py & confusion_matrix.png: Performance metrics and visualization.

🛠️ Installation
Clone the repository:

Bash
git clone https://github.com/your-username/Multi-Model-Fake-News-Detection.git
cd Multi-Model-Fake-News-Detection
Install Dependencies:

Bash
pip install -r requirements.txt
💻 Usage
1. Data Preparation
Clean and prepare your datasets:

Bash
python prepare_dataset.py
2. Training
Train the multi-model ensemble:

Bash
python train.py
3. Running the Web App
Launch the Flask interface to test news articles manually:

Bash
python app.py
📊 Evaluation
After training, you can find the detailed performance metrics in the evaluation_results/ folder and view the confusion_matrix.png for a visual breakdown of classification accuracy.
