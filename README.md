# 🔍 AI-Based Document Forgery Detection

An advanced, multi-agent AI system designed to detect digital manipulations, forged signatures, and altered text in official documents, ID cards, and invoices. 

### 🛑 Problem Statement
Fraudsters manipulate digital documents across financial institutions, government services, and digital verification systems. Detecting subtle manipulations in scanned documents across multiple formats (PDFs, JPEGs) requires high detection accuracy.

### ⚙️ How It Works (The Multi-Agent Pipeline)
This system utilizes multiple specialized AI models working in parallel:
1. **Error Level Analysis (ELA):** Uses OpenCV to detect "copy-paste" forgeries and compression anomalies at the pixel level.
2. **Metadata Extraction:** Deep scanning of EXIF data to reveal if software like Photoshop, GIMP, or Canva was used to save the file.
3. **CNN Classifier:** A PyTorch-based ResNet18 neural network fine-tuned specifically to classify documents as `Real` or `Forged`.
4. **Interactive Dashboard:** A complete robust frontend built on Streamlit for instant results and visual heatmaps.

### 🚀 Tech Stack
* **Frontend:** Streamlit
* **Deep Learning:** PyTorch, Torchvision
* **Computer Vision:** OpenCV, PIL, Scikit-learn
* **Document Parsing:** PyMuPDF, ExifRead
