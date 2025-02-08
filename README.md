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

## Model Training and Architecture

The digit recognition model is built using a Vision Transformer (ViT) architecture with the following key components:

- **Dataset:**  
  We use the extended EMNIST dataset—a variant of the MNIST dataset with additional handwritten digits—to train our model. The dataset is preprocessed (resized, normalized) to match the input requirements of the transformer.

- **Model Architecture:**  
  - **Image Processor:** The `ViTImageProcessor` (formerly `ViTFeatureExtractor`) processes input images.
  - **Transformer Encoder-Decoder:** A transformer-based encoder-decoder architecture is used to interpret the image features and convert them into a textual representation of digits.
  - **Tokenizer:** The tokenizer converts model outputs into human-readable digits.

- **Training Workflow:**  
  1. **Preprocessing:** Extended EMNIST images are normalized and resized.
  2. **Training:** The model is trained on the training split of the dataset.  
     - **Validation Accuracy:** ~98%  
     - **Validation Loss:** ~0.05  
  3. **Validation:** The model is validated on a separate validation set.
  4. **Deployment:** The trained model is saved to the `./model` directory for use in the OCR pipeline.

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


