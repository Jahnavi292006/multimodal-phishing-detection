# Multimodal Phishing Detection with Late-Fusion Architecture

## 📌 Project Overview
Modern phishing attacks increasingly utilize "visual spoofing"—such as rendering fake login forms on compromised legitimate domains—to evade traditional, text-centric security filters. This project addresses the critical need for a multidimensional security system by developing a **Modular Late-Fusion Ensemble** that evaluates a webpage's URL, text, and visual layout simultaneously.

By decoupling the modalities, this architecture eliminates the computational bloat and "black-box" nature of current State-of-the-Art (SOTA) joint-fusion models, achieving **98.2% accuracy** while maintaining full mathematical interpretability and lightweight edge-deployability.

## 🧠 The Architecture (9-Model Wide-Stacking)
Our pipeline mimics human threat evaluation through three distinct modalities:

1. **Lexical (URL):** Evaluates 54 deterministic, offline URL heuristics using Tree-based models (XGBoost, LightGBM, Random Forest).
2. **Semantic (HTML Text):** Analyzes semantic intent across multiple languages using frozen **MPNet** embeddings passed into Distance/Neural models (SVM, MLP).
3. **Spatial (Visual Image):** Verifies webpage layout and logos using frozen **ResNet50** embeddings passed into Linear Classifiers (Logistic Regression, Linear SVM) to prevent high-dimensional overfitting.

**Late-Fusion:** The final probabilities from the top expert in each modality are fused using an XGBoost Meta-Classifier, allowing us to extract exact modality contribution weights (e.g., 86% Text, 8% URL, 5% Image).

## 📂 Repository Structure
* `Final_Phish360_Master_Pipeline_v4.ipynb`: The main, end-to-end execution pipeline (Data Loading -> Feature Extraction -> Training -> Evaluation -> Visualization).
* `Phish360 EDA PQ.pdf`: Exploratory Data Analysis documentation.
* *(Note: The Phish360 dataset and heavy model embeddings are not hosted in this repo due to size constraints. See setup instructions below).*

## ⚙️ Installation & Setup Instruction

**1. Clone the repository:**
```bash
git clone [https://github.com/your-username/multimodal-phishing-detection.git](https://github.com/your-username/multimodal-phishing-detection.git)
cd multimodal-phishing-detection
