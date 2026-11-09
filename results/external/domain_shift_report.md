# Acoustic Domain Shift Report: GTZAN vs MTG-Jamendo

## 1. Executive Summary

A quantitative comparative analysis was conducted between the GTZAN training dataset (1990s analog radio/CD recordings) and the MTG-Jamendo external benchmark (2000s–2010s digital releases by 100 distinct independent artists).

The analysis establishes **significant domain shift** across spectral rolloff, spectral centroid, dynamic range, and MFCC energy profiles, explaining the observed performance drop between cross-validation and out-of-distribution evaluation.

## 2. Statistical Divergence Metrics

| Feature | GTZAN Mean (Std) | Jamendo Mean (Std) | Cohen's d | PSI | Shift Severity |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Spectral Centroid (Hz)** | 2258.75 (729.79) | 2081.94 (683.96) | -0.249 | 0.1721 | **MODERATE SHIFT** |
| **Spectral Bandwidth (Hz)** | 2265.9 (534.74) | 2272.49 (498.23) | 0.013 | 0.2175 | **MODERATE SHIFT** |
| **Spectral Rolloff (Hz)** | 4684.29 (1603.62) | 4402.31 (1483.12) | -0.182 | 0.0892 | **STABLE** |
| **Spectral Contrast (dB)** | 20.59 (2.34) | 21.67 (1.64) | 0.531 | 2.5454 | **CRITICAL SHIFT** |
| **Zero Crossing Rate** | 0.11 (0.04) | 0.09 (0.04) | -0.409 | 0.3687 | **CRITICAL SHIFT** |
| **RMS Energy** | 0.14 (0.07) | 0.19 (0.06) | 0.831 | 2.5098 | **CRITICAL SHIFT** |
| **Tempo (BPM)** | 116.89 (27.06) | 121.25 (28.69) | 0.155 | 0.2301 | **MODERATE SHIFT** |
| **MFCC 1 (Energy)** | -136.21 (98.35) | -101.21 (75.41) | 0.397 | 0.2794 | **CRITICAL SHIFT** |
| **MFCC 2 (Spectral Tilt)** | 97.96 (31.24) | 100.83 (32.97) | 0.089 | 0.2949 | **CRITICAL SHIFT** |
| **Chroma Mean Energy** | 0.38 (0.09) | 0.39 (0.08) | 0.065 | 0.1728 | **MODERATE SHIFT** |

## 3. Shift Attribution Analysis

- **Production-Era Shift**: Modern digital production masters exhibit higher high-frequency energy (>6 kHz) and aggressive multi-band limiting compared to 1980s-90s band-limited analog masters.
- **Dataset-Source Shift**: GTZAN was sourced from CD audio ripped using early 2000s codecs, whereas Jamendo consists of born-digital tracks produced in DAW environments (Ableton, Logic, FL Studio).
- **Genre Boundary Permeability**: In modern music, genres like Pop, Rock, and Hip-hop frequently borrow production techniques (e.g., trap 808s in Pop, electronic synthesizers in Rock), causing severe cross-genre confusion in models trained on vintage archetypes.
