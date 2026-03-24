# Speech Emotion Recognition using Hybrid CNN–BiLSTM

A deep learning system that detects human emotions from speech signals using a hybrid **CNN + Bidirectional LSTM** architecture with an attention mechanism — capable of classifying emotions such as happiness, sadness, anger, fear, disgust, and surprise.

---

## Overview

Human speech carries emotional cues beyond just words — through tone, pitch, and rhythm. This project presents an automated Speech Emotion Recognition (SER) system that identifies emotional states from audio signals, making it useful for applications like virtual assistants, mental health monitoring, customer service analytics, and affective computing.

---

## How It Works

1. **Data Collection** — Audio samples are loaded from three benchmark datasets: RAVDESS, TESS, and SAVEE
2. **Preprocessing** — Audio is converted to mono, resampled to 16kHz, normalized, and trimmed/padded to a fixed 3-second length
3. **Feature Extraction** — MFCC, Delta, and Delta-Delta coefficients are extracted using Librosa to capture spectral and temporal speech characteristics
4. **Data Balancing** — SMOTE (Synthetic Minority Over-sampling Technique) is applied to handle class imbalance
5. **Model Training** — A hybrid CNN + Bi-LSTM model with an attention mechanism is trained using Adam optimizer and categorical cross-entropy loss
6. **Prediction** — The system outputs the detected emotion along with a confidence score

---

## Model Architecture

```
Input (MFCC Features)
        ↓
1D Convolutional Layers    ← Spatial pattern extraction
        ↓
Batch Normalization + MaxPooling + Dropout
        ↓
Bidirectional LSTM Layer   ← Temporal dependency modeling
        ↓
Attention Mechanism        ← Focus on emotionally relevant frames
        ↓
Fully Connected Dense Layers
        ↓
Softmax Output             ← Emotion class probabilities
```

---

## Emotions Detected

| Emotion | | Emotion |
|---------|---|---------|
| 😠 Angry | | 😨 Fearful |
| 🤢 Disgust | | 😊 Happy |
| 😢 Sad | | 😲 Surprised |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core programming language |
| TensorFlow / Keras | Deep learning model |
| Librosa | Audio feature extraction |
| scikit-learn | Preprocessing and evaluation |
| imbalanced-learn | SMOTE for class balancing |
| NumPy / Pandas | Data handling |
| Matplotlib / Seaborn | Visualization |

---

## Project Structure

```
speech-emotion-recognition/
│
├── SER.ipynb             # Main notebook (Google Colab)
├── README.md             # Project documentation
```

---

## How to Run

1. Open the notebook in [Google Colab](https://colab.research.google.com)
2. Mount your Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Place your datasets in the following structure:
   ```
   datasets/
   ├── RAVDESS/    ← Ryerson Audio-Visual Database
   ├── TESS/       ← Toronto Emotional Speech Set
   └── SAVEE/      ← Surrey Audio-Visual Expressed Emotion
   ```
4. Run all cells from top to bottom
5. Use the prediction function at the end to test with your own `.wav` audio files

---

## Datasets

This project uses three publicly available benchmark datasets:

| Dataset | Description |
|---------|-------------|
| [RAVDESS](https://zenodo.org/record/1188976) | 24 professional actors, 8 emotions |
| [TESS](https://tspace.library.utoronto.ca/handle/1807/24487) | Toronto Emotional Speech Set |
| [SAVEE](http://kahlan.eps.surrey.ac.uk/savee/) | Surrey Audio-Visual Expressed Emotion |

---

## Future Scope

- Integration of **Transformer-based architectures** for capturing long-range emotional patterns
- **Real-time microphone input** for live emotion detection
- **Multi-modal emotion recognition** combining voice with facial expressions
- **Multilingual support** for diverse speakers and accents
- Integration into **virtual assistants and chatbots** for emotionally aware responses
- **Emotion intensity estimation** using regression-based models

---

## Publication

This project is based on the research paper:

> *"A Hybrid CNN–BiLSTM Approach for Speech Emotion Recognition"*  
> Madhusuthanan K M, Chandra Eswaran  
> Department of Computer Science, Bharathiar University, Coimbatore, Tamil Nadu, India

---

## Author

**madhusuthanankm**  
[GitHub Profile](https://github.com/madhusuthanankm)

---

## License

This project is open source and available under the [MIT License](LICENSE).
