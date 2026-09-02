# 👕 Multimodal T-Shirt Retrieval System

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?logo=pytorch)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-F9AB00?logo=huggingface)
![ChromaDB](https://img.shields.io/badge/Chroma-Vector%20DB-4ECCA3)

An end-to-end multimodal Information Retrieval (IR) system built to retrieve T-shirt images based on natural language queries. This project leverages state-of-the-art **pretrained Vision-Language Models (VLMs)** and **Dense Vector Search** from Hugging Face, requiring zero fine-tuning.

## 🚀 Pipeline Overview

The system is divided into three core tasks, showcasing both direct visual matching and text-based semantic search:

1. **Image Retrieval (Zero-Shot CLIP):** 
   Projects both the text query and all 1,993 images into a shared latent space. Retrieves images by computing the cosine similarity between the text vector and image vectors.
2. **Image Captioning (BLIP):** 
   Automatically generates descriptive textual captions for every image in the dataset, identifying colors, patterns, and typography.
3. **Text Retrieval & Reranking:** 
   Embeds the generated captions into a **ChromaDB** vector database. Initial retrieval is performed using a fast Bi-Encoder, followed by a powerful **Cross-Encoder Reranker** to evaluate deep semantic interactions between the query and captions.

## 🧠 Models Utilized

This project strictly utilizes pretrained models (No training/fine-tuning):

*   **Task 1 (CLIP):** `openai/clip-vit-base-patch32`
    *   *Why?* Industry standard for zero-shot text-to-image retrieval.
*   **Task 2 (Captioning):** `Salesforce/blip-image-captioning-base`
    *   *Why?* Lightweight, fast, and VRAM-efficient conditional language generation. Avoids Out-Of-Memory (OOM) errors on standard GPUs while providing highly accurate apparel descriptions.
*   **Task 3 (Embeddings):** `BAAI/bge-small-en-v1.5`
    *   *Why?* Top-tier performance on the MTEB (Massive Text Embedding Benchmark) for dense retrieval.
*   **Task 3 (Reranker):** `BAAI/bge-reranker-base`
    *   *Why?* Cross-encoders process the query and document together, capturing complex linguistic nuances that standard bi-encoders miss.

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/MK2091379/multimodal-tshirt-retrieval.git](https://github.com/MK2091379/multimodal-tshirt-retrieval.git)
   cd multimodal-tshirt-retrieval

2. **Install dependencies:**
    Ensure you have a GPU-enabled environment (CUDA).
   ```bash
   pip install torch torchvision transformers==4.43.4 pillow chromadb sentence-transformers accelerate bitsandbytes

3. **Download the Dataset:**
   Download the dataset from [Kaggle](https://www.kaggle.com/datasets/danizo/applied-data-mining-final-project-t-shirt-data).
   Extract the images into a directory named ./dataset/images/ at the root of the project.

## 💻 Usage
The entire pipeline is structured sequentially inside a Jupyter Notebook.
Open Multimodal_IR_System.ipynb and run the cells sequentially. The pipeline includes automatic memory management (gc.collect() and torch.cuda.empty_cache()) to run smoothly on standard consumer GPUs (like Kaggle's T4 or P100).

## 📊 Results & Visualization
At the end of the pipeline, the system generates a side-by-side comparative plot demonstrating the Top-N results from both the direct CLIP Visual Search and the Text Reranking Pipeline.
(Insert your output image here: upload tshirt_retrieval_comparison.png to your repo and link it below)

## 🤝 Acknowledgments
Dataset provided by danizo on Kaggle.
Models hosted by Hugging Face.