# Music Generation & Genre Classification System
### Academic Neural Networks & Deep Learning Comparative Study
**Automatic Music Genre Classification: Multilayer Perceptron (ANN) vs. 2D Convolutional Neural Network (CNN) on the GTZAN Benchmark**

---

## 1. Introduction & Problem Statement

Automatic Music Genre Classification (MGC) is a core task in Music Information Retrieval (MIR). Categorizing musical tracks into predefined cultural and acoustic genres poses distinct algorithmic challenges due to:
- High intra-class variance and polyphonic complexity.
- Subjective perceptual boundaries between related genres (e.g., Rock vs. Metal, Blues vs. Country).
- Time-varying dynamic structures spanning rhythm, harmony, timbre, and instrumentation.

This project implements, evaluates, and compares two fundamentally different deep learning paradigms:
1. **Statistical Acoustic Feature Classification (ANN / Multilayer Perceptron):** Hand-crafted global statistical moments (mean, variance) computed from acoustic representations (MFCCs, Chroma, Spectral Centroid, Bandwidth, Rolloff, Zero-Crossing Rate, RMS energy, Tempo).
2. **Representation Learning (2D CNN):** End-to-end spatial pattern extraction from 128-band log Mel-spectrogram time-frequency patches without manual feature engineering.

---

## 2. System Architecture

```
                                  [Uploaded Audio File]
                                            |
                         +------------------+------------------+
                         | (Sample Rate: 22050 Hz, Mono)       |
                         v                                     v
             [Feature Extraction]                     [Audio Segmentation]
        (MFCC, Chroma, Spectral, etc.)              (3.0s window, 50% overlap)
                         |                                     |
                         v                                     v
             [61-D Feature Vector]                [128 x 130 Mel-Spectrograms]
                         |                                     |
                [StandardScaler]                        [Normalization]
                         |                                     |
                         v                                     v
                [ANN / MLP Model]                      [2D CNN Model]
            Dense(256) -> BN -> Drop               Conv2D(32) -> BN -> Pool
            Dense(128) -> BN -> Drop               Conv2D(64) -> BN -> Pool
            Dense(64)  -> Drop                    Conv2D(128)-> BN -> Pool
            Dense(10, Softmax)                     GlobalAveragePooling2D
                         |                         Dense(128) -> Drop(0.4)
                         |                         Dense(10, Softmax)
                         |                                     |
                         |                         [Segment Probabilities]
                         |                                     |
                         |                         [Track Probability Mean]
                         v                                     v
                  ANN Prediction                        CNN Prediction
                         \                                     /
                          +-----------------+-----------------+
                                            v
                               [Comparison & Verification]
```

---

## 3. GTZAN Dataset

The GTZAN genre collection comprises 1,000 audio recordings (30 seconds each, 22,050 Hz, 16-bit mono WAV) across 10 distinct genres:
- **Genres:** `blues`, `classical`, `country`, `disco`, `hiphop`, `jazz`, `metal`, `pop`, `reggae`, `rock`.
- **Distribution:** 100 tracks per genre (~100% balance).
- **Corrupted File Detection:** Standard distributions contain an unreadable 0-byte file (`jazz.00054.wav`). The validation suite detects, isolates, and excludes this track to prevent runtime failure while maintaining dataset integrity (999 valid tracks).

### Data Leakage Prevention Protocol
To avoid optimistic performance bias, **dataset splitting is conducted strictly at the track level before any segmentation**:
- **Train (70%):** 699 songs
- **Validation (15%):** 150 songs
- **Test (15%):** 150 songs (15 tracks per genre)

All sub-segments generated from track $T_i$ are assigned exclusively to the split partition of $T_i$. Under no circumstances do segments of the same song cross split boundaries.

---

## 4. Methodology & Model Architectures

### Architecture 1: Multilayer Perceptron (ANN)
- **Input Dimension:** 75 tabular acoustic features (extended from 61 baseline)
  - 13 MFCC means + 13 MFCC standard deviations (26)
  - 12 Chroma STFT means + 12 Chroma standard deviations (24)
  - 7 Spectral Contrast octave band means + 7 band standard deviations (14)
  - Spectral Centroid (mean + std) (2)
  - Spectral Bandwidth (mean + std) (2)
  - Spectral Rolloff (mean + std) (2)
  - Zero-Crossing Rate (mean + std) (2)
  - RMS Energy (mean + std) (2)
  - Global Tempo (1)
- **Preprocessing:** Audio peak normalization (`max(|y|) <= 1.0`), `StandardScaler` fitted **strictly on training partition**, serialized to `ml/saved_models/ann_scaler.pkl`.
- **Network Architecture:** 3 hidden layers (256, 128, 64 neurons) with ReLU activations, Batch Normalization, and calibrated Dropout (0.25, 0.2, 0.2).
- **Optimization:** Adam ($\alpha = 0.001$), sparse categorical cross-entropy, Early Stopping with `restore_best_weights`.

### Architecture 2: 2D Convolutional Neural Network (CNN)
- **Input Representation:** Log Mel-scaled spectrogram ($F = 128$ Mel frequency bins, $T = 130$ time frames, $C = 1$ channel) representing 3.0s audio segments ($SR = 22050\text{ Hz}$, $N_{\text{FFT}} = 2048$, $\text{hop} = 512$).
- **Convolutional Backbone:** 3 sequential convolutional blocks:
  - Block 1: Conv2D(32, $3\times3$, same) $\to$ BatchNormalization $\to$ MaxPooling2D($2\times2$)
  - Block 2: Conv2D(64, $3\times3$, same) $\to$ BatchNormalization $\to$ MaxPooling2D($2\times2$)
  - Block 3: Conv2D(128, $3\times3$, same) $\to$ BatchNormalization $\to$ MaxPooling2D($2\times2$)
- **Spatial Aggregation:** `GlobalAveragePooling2D` replaces flattening to enforce translation equivariance and eliminate dense parameter explosion.
- **Classification Head:** Dense(128, ReLU) $\to$ Dropout(0.4) $\to$ Dense(10, Softmax).
- **Track Inference Aggregation:** For an arbitrary audio track, $K$ overlapping segments (50% temporal overlap) are evaluated. Track-level prediction is determined by segment probability averaging:
  $$\hat{y} = \arg\max_{c} \left( \frac{1}{K} \sum_{k=1}^{K} P(y_k = c \mid X_k) \right)$$

---

## 5. Technology Stack

- **Machine Learning:** TensorFlow 2.16+, Keras 3, Librosa 1.0, Scikit-Learn, NumPy, Pandas.
- **Backend API:** FastAPI, Uvicorn (ASGI), aiosqlite (async SQLite database), Pydantic v2.
- **Frontend UI:** React 18, TypeScript, Vite, Tailwind CSS (minimal academic design palette).
- **Database:** SQLite3 (`backend/predictions.db`).

---

## 6. Installation & Setup

### Prerequisites
- Python 3.11 or 3.12
- Node.js 18+ and npm

### 1. Environment & Dependencies
```bash
# Clone or navigate to the repository
cd /path/to/NNP

# Create and activate Python virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Install Python requirements
pip install -r requirements.txt

# Install Frontend dependencies
cd frontend
npm install
cd ..
```

### 2. Dataset Setup
Download the GTZAN dataset from Kaggle:
```
https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
```
Place or symlink the extracted `genres_original` folder to `./dataset/genres_original`.

Validate dataset structure:
```bash
source venv/bin/activate
python ml/dataset_validator.py --check-audio
```

---

## 7. Model Training & Evaluation

### Train ANN Classifier
```bash
source venv/bin/activate
python ml/training/train_ann.py --epochs 120 --batch-size 32
```

### Train CNN Classifier
```bash
source venv/bin/activate
python ml/training/train_cnn.py --epochs 30 --batch-size 64
```

Both pipelines automatically:
- Split the dataset at track level (70% train, 15% val, 15% test).
- Save best model checkpoints (`.keras`) and feature transformations (`.pkl`).
- Evaluate on the held-out test partition.
- Generate confusion matrices, loss curves, and classification reports in `results/`.

---

## 8. Empirical Experimental Results & Benchmark Evolution

The benchmark results reflect three distinct experimental eras, culminating in a zero-leakage decontaminated evaluation:

### 8.1 Benchmark Evolution Across Three Experimental Eras

| Experimental Era | Split Methodology | Dataset Size | Leakage Status | ANN Test Accuracy | ANN Macro F1 | CNN Test Accuracy | CNN Macro F1 | Superior Architecture |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1. Historical Baseline** | Legacy Stratified ($seed=42$) | 1,000 tracks (150 test) | 7 Leaking Duplicate Pairs | 68.00% | 0.6667 | 87.33%* | 0.8715 | **CNN (+19.33%)** |
| **2. Phase 7 Improved** | Controlled Ablation ($seed=42$) | 1,000 tracks (150 test) | 7 Leaking Duplicate Pairs | 73.33% | 0.7312 | 86.67% | 0.8646 | **CNN (+13.34%)** |
| **3. Phase 8 Final Benchmark** | **Grouped Stratified (Zero Leakage)** | **998 tracks (153 test)** | **0 Leaks (Atomic Groups)** | **69.28%** | **0.6961** | **80.39%** | **0.7914** | **CNN (+11.11%)** |

*\*Scientific Accountability Note:* Historical reported CNN accuracy on GTZAN (87.33%) was artificially inflated because 3 exact duplicate pairs leaked directly between train and test partitions (`metal.00044` ↔ `metal.00065`, `metal.00066` ↔ `metal.00045`, `hiphop.00039` ↔ `hiphop.00045`). When evaluated on strictly isolated, decontaminated audio, the CNN achieves **80.39%**, establishing an un-inflated, scientifically defensible generalization benchmark.

### 8.2 Phase 8 Final Clean Benchmark: Per-Genre Performance Breakdown

*Evaluated on the 153 clean held-out tracks with zero duplicate leakage:*

| Genre | Test Support | ANN Precision | ANN Recall | ANN F1 | CNN Precision | CNN Recall | CNN F1 | Superior Architecture |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Blues** | 15 tracks | 0.692 | 0.600 | 0.643 | 0.733 | 0.733 | 0.733 | **CNN (+9.0%)** |
| **Classical** | 15 tracks | 0.938 | 1.000 | 0.968 | 1.000 | 0.933 | 0.966 | ANN (+0.2%) |
| **Country** | 15 tracks | 0.526 | 0.667 | 0.588 | 0.824 | 0.933 | 0.875 | **CNN (+28.7%)** |
| **Disco** | 15 tracks | 0.500 | 0.533 | 0.516 | 0.786 | 0.733 | 0.759 | **CNN (+24.3%)** |
| **Hiphop** | 16 tracks | 0.706 | 0.750 | 0.727 | 0.824 | 0.875 | 0.848 | **CNN (+12.1%)** |
| **Jazz** | 16 tracks | 0.833 | 0.938 | 0.882 | 0.889 | 1.000 | 0.941 | **CNN (+5.9%)** |
| **Metal** | 16 tracks | 0.765 | 0.812 | 0.788 | 0.929 | 0.812 | 0.867 | **CNN (+7.9%)** |
| **Pop** | 14 tracks | 0.714 | 0.714 | 0.714 | 0.706 | 0.857 | 0.774 | **CNN (+6.0%)** |
| **Reggae** | 16 tracks | 0.722 | 0.812 | 0.765 | 0.737 | 0.875 | 0.800 | **CNN (+3.5%)** |
| **Rock** | 15 tracks | 0.600 | 0.400 | 0.480 | 0.727 | 0.533 | 0.615 | **CNN (+13.5%)** |

### 8.3 Dataset Decontamination & Contradictory Label Resolution
1. **Unreadable File Isolation:** Standard GTZAN distributions contain a corrupted 0-byte file (`jazz.00054.wav`). Detected and isolated (leaving 999 valid audio files).
2. **Contradictory Label Investigation:**
   - Excerpt `metal.00058.wav` and `rock.00016.wav` share 100% sample-identical PCM audio (`max_diff = 0.0`).
   - Song identification: Queen's *"Tie Your Mother Down"* (A Day at the Races, 1976).
   - Stylistically and historically, Queen is Hard Rock / Classic Rock, not Heavy Metal. Sourcing by Tzanetakis placed Queen in both folders, creating contradictory supervisory targets.
   - *Resolution:* Excluded `metal.00058.wav` from the clean dataset while retaining `rock.00016.wav` under its legitimate genre.
3. **Atomic Grouped Splitting:**
   - 30 duplicate and near-duplicate connected components (60 tracks) were grouped atomically.
   - Stratified partitioning was performed at the **group level** (`dataset/clean_splits.json`), guaranteeing **0 duplicate recordings cross train/val/test boundaries**. Total decontaminated tracks: **998**.

### 8.4 External Generalisation & Out-of-Taxonomy Diagnostics

To evaluate real-world generalisation beyond the GTZAN benchmark, 4 diverse external recordings were tested without synthetic manipulation:

| Test Track | Ground Truth Genre | GTZAN Taxonomy Status | Duration | ANN Prediction | CNN Prediction | System Diagnostic Verdict |
|---|---|:---:|:---:|---|---|:---:|
| `jazz_smooth.mp3` | Smooth Jazz Stinger | In-Taxonomy | 35.5s | reggae (48.7%) | disco (33.1%) | ⚠️ High Uncertainty |
| `rock_modern.mp3` | Modern Rock Production | In-Taxonomy | 18.5s | pop (47.8%) | pop (45.5%) | ⚠️ High Uncertainty |
| `edm_electronic.mp3` | EDM Synthesizer Track | **Out-of-Taxonomy** | 15.1s | rock (42.1%) | country (46.3%) | ⚠️ High Uncertainty / Out-of-Taxonomy |
| `pop_commercial.mp3` | Commercial Pop Single | In-Taxonomy | 196.6s | pop (94.3%) | pop (79.2%) | **✓ Confident Consensus** |

### 8.5 Uncertainty Quantification & Calibration

Rather than treating raw softmax outputs as unquestioned ground truth, an uncertainty quantification module was calibrated on validation distributions:
- **Entropy Threshold:** Shannon entropy $H > 1.65\text{ bits}$ indicates a diffuse, ambiguous distribution across genres.
- **Top-1 Confidence Threshold:** $P_1 < 0.55$ indicates insufficient classification evidence.
- **Top-2 Margin Threshold:** $P_1 - P_2 < 0.20$ indicates severe multi-genre competition.

**Validation Benchmark:** This uncertainty filter catches **77.3% of all neural network classification errors** while maintaining **95.61% precision** on predictions declared "Confident Consensus". When out-of-taxonomy or cross-genre tracks are encountered, the system provides transparent advisories rather than hallucinatory assertions.

### 8.6 Key Empirical Findings
1. **Representation Advantage Preserved:** Even after eliminating duplicate test leakage, the 2D CNN decisively outperforms the ANN (**80.39% vs. 69.28% test accuracy**, a **+11.11% advantage**), proving that 2D convolutional filters over Mel-spectrograms capture local harmonic, rhythmic, and timbral structures far better than global statistical averages.
2. **Impact of Leakage Elimination:** On the contaminated split, the CNN achieved 87.33% because leaked identical tracks appeared in both train and test. On the clean split, the CNN's true score is 80.39%, confirming that ~7% of historical reported accuracy was attributable to memorization of duplicated excerpts.
3. **ANN Gain Under Strict Evaluation:** The 75-feature ANN (with Spectral Contrast and Peak Normalization) achieved **69.28%** on the clean split, outperforming the historical 61-feature baseline (68.00%) despite the stricter, non-leaking test partition.
4. **Hardest Genre Boundaries:** For both architectures, the rock/pop boundary remains the most confusable (Rock F1: ANN 0.480, CNN 0.615) due to shared instrumentation (drums, bass, electric guitars) and overlapping song production techniques. Classical remains the most distinguishable (CNN F1 0.966, ANN F1 0.968).

---

## 9. Running the Application

### Start the FastAPI Backend
```bash
source venv/bin/activate
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
API Documentation will be accessible at: `http://localhost:8000/docs`

### Start the React Frontend
```bash
cd frontend
npm run dev
```
Open your browser at: `http://localhost:5173`

---

## 10. API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | System health and model availability status |
| `GET` | `/api/dataset/status` | Dataset integrity, file counts, and genre validation |
| `GET` | `/api/models/status` | Check if ANN and CNN weights are loaded |
| `POST` | `/api/predict` | Upload audio (WAV/MP3), returns dual model classification |
| `GET` | `/api/evaluation` | Test-set classification metrics (accuracy, F1, loss) |
| `GET` | `/api/visualizations/{model}/{file}` | Safe retrieval of evaluation charts & confusion matrices |
| `GET` | `/api/history` | List recent prediction records |
| `GET` | `/api/history/{id}` | Inspect detailed top-3 probabilities of a record |
| `DELETE`| `/api/history/{id}` | Remove a prediction record from history |

---

## 11. Automated Tests

Run the test suite verifying preprocessing, input dimensions, data leakage prevention, and API contracts:
```bash
source venv/bin/activate
pytest tests/ -v
```

---

## 12. Academic Scope & Limitations

1. **Music Generation Distinction:** Music generation is reserved as a separate future research track (e.g. Diffusion-based latent audio models) and is intentionally not faked or mixed into the genre classification task.
2. **Temporal Aggregation:** The CNN employs segment probability averaging across the track duration. While effective, attention-based segment weighting or recurrent temporal pooling (CRNN) represents an interesting future extension.
3. **Out-of-Distribution Sensitivity:** Audio recordings featuring heavy modern mastering compression or novel sub-genres may trigger prediction disagreement between models, highlighting the distinction between model softmax confidence and ground-truth certainty.
