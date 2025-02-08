# Neuro Sudoku

A full-stack web application for real-time Sudoku solving. Users upload an image of a Sudoku puzzle, the application extracts and recognizes the digits using a Vision Transformer (ViT) model trained on the extended EMNIST dataset, and then solves the puzzle with a backtracking algorithm. The project leverages OpenCV for image preprocessing, FastAPI for the backend, and Next.js for the frontend.

## Table of Contents

- [About](#about)
- [Model Training and Architecture](#model-training-and-architecture)
- [Image Processing Workflow](#image-processing-workflow)
- [Sudoku Solving Algorithm](#sudoku-solving-algorithm)
- [Installation and Setup](#installation-and-setup)
  - [Conda Environment](#conda-environment)
  - [Frontend Setup](#frontend-setup)
  - [Backend Setup](#backend-setup)
- [Usage](#usage)

## About

This application offers a seamless, AI-driven solution for solving Sudoku puzzles:
- **Real-time Image Processing:** Detects and extracts the Sudoku grid from an uploaded image.
- **Digit Recognition:** Uses a state-of-the-art Vision Transformer model (ViT) to recognize individual digits.
- **Automated Puzzle Solving:** Employs a backtracking algorithm to solve the recognized Sudoku puzzle.
- **Full-Stack Implementation:** FastAPI for the backend API and Next.js for the frontend interface.

# Model Training and Architecture: TrOCR-based Replica on EMNIST

This document details the design, pre-training, fine-tuning, and evaluation of our transformer-based OCR model—an adaptation of [TrOCR: Transformer-based Optical Character Recognition](https://ar5iv.org/abs/2109.10282)—using the EMNIST dataset for handwritten digit recognition.

> **Note:** While the original TrOCR paper employs a two-stage pre-training on hundreds of millions of printed and handwritten textline images, our replication leverages the EMNIST dataset. We adapt the same principles (encoder-decoder initialization, task formulation, data augmentation, etc.) to recognize handwritten digits (0–9).


## 1. Overview


Our OCR system uses an **encoder-decoder architecture** where:
- The **encoder** is based on a Vision Transformer (ViT) pre-trained using methods inspired by **DeiT/BEiT**.
- The **decoder** is adapted from pre-trained language models (e.g., RoBERTa or MiniLM) and simplified to output a single digit token.
- The model is trained end-to-end on the EMNIST dataset, which contains isolated handwritten digits.

---

## 2. Dataset

### EMNIST Details
- **Dataset:** Extended MNIST (EMNIST) Digits
- **Content:** Handwritten digits (0–9)
- **Splits:**  
  - Training set: ~280,000 images  
  - Test set: ~40,000 images

#### Preprocessing Steps:
- **Resizing:** Images are resized to **32×32 pixels** (or adjusted to match the ViT patch size).
- **Normalization:** Pixel intensities are scaled to the range **[0, 1]**.
- **Binarization (optional):** Enhance contrast for improved digit delineation.
- **Augmentation:**
  - Random rotations (e.g., ±10°)
  - Gaussian blurring
  - Affine transformations (dilation/erosion)
  - Slight scaling adjustments

---

### 3. Model Architecture

#### 3.1 Encoder Initialization
Inspired by TrOCR, the encoder leverages ViT models pre-trained with techniques from **DeiT** and **BEiT**:
- **DeiT-based Initialization:**
  - Pre-trained on ImageNet for robust image feature extraction.
  - Incorporates a **distilled token** to capture class-specific representations.
- **BEiT-based Initialization:**
  - Uses **Masked Image Modeling** by masking random patches and recovering them.
  - Converts input images into sequences of patch embeddings.

**Our Setup:**
- **Patch Tokenization:** The 32×32 image is divided into fixed-size patches (e.g., 4×4 patches).
- **Transformer Layers:** 12 layers, 8 attention heads, with a hidden dimension of 768.
- **Positional Embeddings:** Added to each patch embedding to preserve spatial relationships.

#### 3.2 Decoder Initialization
The decoder is adapted from pre-trained language models:
- **RoBERTa / MiniLM:** Originally designed for language tasks; repurposed here for digit recognition.
- **Modifications for Single-Digit Recognition:**
  - Manually insert encoder-decoder attention layers to align visual and token embeddings.
  - Map pre-trained weights to the decoder’s self-attention layers.
  - Randomly initialize additional parameters (e.g., cross-attention layers) not present in the original model.
  - The decoder is streamlined to output a single digit token, along with special tokens `[BOS]` and `[EOS]`.

**Decoder Details:**
- **Tokenization:** Uses a Byte Pair Encoding (BPE) or SentencePiece tokenizer with a vocabulary limited to digits (0–9) and special tokens.
- **Special Tokens:**
  - `[BOS]`: Indicates the beginning of the sequence.
  - `[EOS]`: Indicates the end of the sequence.
- **Architecture:** A simplified stack of Transformer layers tailored for a single-token output.

---

## 4. Task Pipeline

Following the TrOCR approach, our pipeline for single-digit recognition consists of:

1. **Input Processing:**
   - The EMNIST image is fed into the encoder, which converts it into a sequence of visual embeddings.

2. **Sequence Generation:**
   - During training, the decoder receives the ground-truth digit token prepended with `[BOS]`.
   - The model predicts a single digit token, and the prediction is compared against the ground truth (with `[EOS]` appended) using **cross-entropy loss**.

3. **Inference:**
   - The decoder starts with `[BOS]` and outputs a single digit token.
   - The process stops immediately after the first token is generated, ensuring that each cell is recognized as a single digit.

---

## 5. Pre-training and Fine-tuning

### 5.1 Pre-training (Adapted for EMNIST)
While TrOCR uses a two-stage pre-training strategy on massive synthetic datasets, our approach focuses on the domain of single-digit recognition:
- **Stage 1: Basic Visual-Language Pre-training (Optional)**
  - Uses synthetically augmented digit images to pre-train the model on a similar recognition task.
  - Helps the model learn joint visual and token representations.
- **Stage 2: EMNIST Fine-tuning**
  - The full model is fine-tuned on the EMNIST training set.
  - Hyperparameters (learning rate, batch size, number of epochs) are adjusted for optimal performance on this focused task.

### 5.2 Fine-tuning Details:
- **Loss Function:** Cross-entropy loss computed over the predicted digit token.
- **Hyperparameters:**
  - **Epochs:** ~50 (adjustable based on convergence)
  - **Batch Size:** 128 (depending on available GPU memory)
  - **Learning Rate:** 5e-4, with cosine decay scheduling
  - **Optimizer:** AdamW with a weight decay of 0.01
  - **Dropout:** 0.1 to mitigate overfitting

---

## 6. Data Augmentation

Aligned with the TrOCR methodology, various augmentation strategies are employed to improve model robustness:
- **Handwritten-Specific Augmentations:**
  - **Random Rotation:** ±10° to simulate natural handwriting variations.
  - **Gaussian Blur:** Imitates scanning or imaging artifacts.
  - **Affine Transformations:** Dilation and erosion mimic variations in ink flow.
  - **Scaling:** Minor adjustments to represent different writing sizes.
- **Selection:** Each image is randomly subjected to one or more augmentations during training.

---

## 7. Experiments and Evaluation

### 7.1 Metrics
- **Accuracy:** Top-1 accuracy for single-digit classification.
- **Precision, Recall, F1-Score:** Computed across all 10 digit classes (0–9).

### 7.2 Results (Example Values)
- **Validation Accuracy:** ~98%
- **Validation Loss:** ~0.05

---

## 8. Conclusion

This document details the design of a TrOCR-inspired model tailored for single-digit recognition using the EMNIST dataset. By integrating a ViT-based encoder with a streamlined transformer decoder and employing advanced data augmentation strategies, the model achieves robust recognition performance. This forms the backbone of the Neuro Sudoku application, enabling efficient, real-time Sudoku puzzle solving.



## Image Processing Workflow

When a user uploads an image of a Sudoku puzzle, the following steps occur:

1. **Grayscale Conversion:**  
   The image is converted to grayscale.

2. **Preprocessing:**  
   - **Gaussian Blurring:** Reduces noise.
   - **Adaptive Thresholding:** Converts the image to a binary format.
   - **Image Inversion & Dilation:** Enhances the grid lines.

3. **Grid Detection:**  
   The largest contour in the processed image is assumed to be the Sudoku grid.

4. **Perspective Transform:**  
   The detected grid is warped to produce a top-down view.

5. **Cell Extraction:**  
   The transformed grid is divided into 81 cells. Each cell is:
   - Cropped by a 15% margin to remove border noise.
   - Evaluated for blankness (using a standard deviation threshold).
   - Processed by the OCR pipeline (if non-blank) to extract a digit.

## Sudoku Solving Algorithm

Once the grid is extracted, the puzzle is solved using a backtracking algorithm:
- **Find Empty Cell:** Searches for the next empty cell (denoted by 0).
- **Validation:** Checks if placing a digit (1–9) violates any Sudoku rules (row, column, or 3x3 subgrid).
- **Recursive Backtracking:** Recursively attempts to fill the grid. If no valid number is found for a cell, the algorithm backtracks to try alternative digits.
- **Solution Output:** The final solved grid is returned and displayed.

## Installation and Setup

### Conda Environment

1. **Create the Conda Environment:**  
   Use the provided `environment.yml` file to set up the environment:
   ```bash
   conda env create -f environment.yml

2. **Activate the Environment**:

   ```
   conda activate your_env_name
   ```

### Frontend Setup

1. **Navigate to the frontend directory:**  
   ```bash
   cd frontend
2. **Install dependencies and start the development server**:

   ```
   npm install
   npm run dev
   ```

### Backend Setup

1. **Navigate to the backend directory**  
   ```bash
   cd backend
2. **Start the FastAPI backend**:

   ```
   uvicorn api.main:app --reload
   ```

## Usage

- **Upload a Sudoku Image:**  
  Use the frontend interface to upload an image of a Sudoku puzzle.

- **Digit Recognition and Solving:**  
  The backend processes the image, extracts the grid, recognizes digits using the AI model, and solves the puzzle using the backtracking algorithm.

- **View Results:**  
  The recognized grid and solved puzzle are displayed on the frontend.


