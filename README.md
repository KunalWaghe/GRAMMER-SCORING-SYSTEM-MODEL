# SHL Grammar Scoring Engine (Intern Assessment 2025)

## 📌 Project Overview

This project builds an automated **Grammar Scoring Engine** for spoken audio samples. It takes audio input (speech), transcribes it into text, analyzes the linguistic structure, and predicts a grammar score on a scale of 1-5 (Likert Scale).

**Key Metric:** Root Mean Squared Error (RMSE) & Pearson Correlation.

## 🛠️ Tech Stack & Methodology

The solution uses a **"Transcribe-then-Analyze"** pipeline approach:

1. **Audio Preprocessing:**
* **Challenge:** The provided dataset contained MP4/AAC audio files incorrectly renamed with `.wav` extensions (corrupt headers).
* **Solution:** Implemented a robust conversion layer using `moviepy` and `ffmpeg` to repair headers and convert raw audio to standard 16kHz PCM WAV format suitable for AI models.


2. **ASR (Automatic Speech Recognition):**
* Used **OpenAI Whisper (Base Model)** for high-accuracy transcription of spoken audio to text.


3. **Feature Engineering:**
* **Semantic Embeddings:** Used **BERT (Sentence-Transformers)** to capture the semantic meaning and sentence structure context.
* **Grammar Analysis:** (Optional/Planned) Integration of `language_tool_python` to explicitly count syntax errors.


4. **Machine Learning Model:**
* **XGBoost Regressor:** Chosen for its efficiency with tabular embeddings and ability to prevent overfitting on small datasets (409 samples).



## 📂 Project Structure

```bash
SHL_Assessment/
│
├── data/
│   ├── train/               # Training audio files
│   ├── test/                # Test audio files
│   ├── train.csv            # Training labels
│   └── test.csv             # Test filenames
│
├── notebooks/
│   └── main_pipeline.ipynb  # The core analysis notebook
│
├── models/                  # Saved models (if applicable)
│
├── submission.csv           # Final predictions for Kaggle/SHL
├── README.md                # Project Documentation
└── requirements.txt         # Python dependencies

```

## ⚙️ Installation & Prerequisites

This project requires **Python 3.8+** and **FFmpeg** installed on the system.

### 1. Install Libraries

```bash
pip install pandas numpy scikit-learn xgboost openai-whisper
pip install sentence-transformers moviepy imageio-ffmpeg

```

### 2. FFmpeg Configuration (Windows)

*Note: This project relies on `moviepy` to handle audio conversion. If you encounter "File Not Found" errors during audio loading, ensure `imageio-ffmpeg` is correctly linked in your environment variables.*

## 🚀 How to Run the Pipeline

### Step 1: Data Setup

Ensure your `train.csv` and `test.csv` are in the root directory, and update the `TRAIN_AUDIO_FOLDER` path in the notebook to point to your unzipped audio files.

### Step 2: Audio Transcription (Batch Processing)

Run the **Transcription Batch** in the notebook.

* **Input:** Raw Audio (Fake WAVs).
* **Process:** Converts MP4 -> WAV -> Text.
* **Output:** Generates `train_transcribed.csv` and `test_transcribed.csv`.
* *Note: This step takes approx. 15-20 minutes on CPU.*

### Step 3: Training & Prediction

Run the **Modelling Batch**.

* Loads the transcribed CSVs.
* Generates BERT embeddings.
* Trains the XGBoost Regressor.
* Calculates RMSE on the validation set.

### Step 4: Submission

The notebook generates a `submission.csv` file formatted according to the SHL guidelines (columns: `filename`, `label`).

## 📊 Performance Notes

* **Dataset:** 409 Training samples, 197 Test samples.
* **Model Validation:** Stratified K-Fold validation is recommended due to the small dataset size.
* **Handling Corrupt Files:** The custom `transcribe_mp4_masked_as_wav` function ensures 100% of the dataset is processed, handling files that standard libraries (`scipy`, `librosa`) fail to open.

## 👤 Author

**[Kunal Waghe]**
*Candidate for SHL Intern Hiring Assessment 2025*
