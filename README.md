# Interpretable ECG Classification Using Criss-Cross Transformers and Soft Local Completeness

This project investigates deep learning architectures for multi-class ECG diagnosis on the PTB-XL dataset, with a focus on model interpretability via Soft Local Explanations (SLOC). Three architectures are compared: A Criss-Cross Transformer (CCT), a CNN, and a CNN+BiLSTM. Their learned representations are analysed through SLOC generated attribution maps.

## Key Contributions

- **Criss-Cross Transformer (CCT)** - a dual-attention transformer that applies separate spatial (cross-lead) and temporal attention, to capture both intra-lead morphology and inter-lead dependencies in 12-lead ECGs.
- **SLOC interpretability pipeline** - a local completeness based interpretability method that generates per-sample attribution maps by utilising gradient descent on a novel completeness gap metrics. Adapted from the Soft Local Completeness paper by Haddad et al. (citation below)
- **Systematic comparison** of CCT, CNN, and CNN+BiLSTM on both clean (neurokit2-processed) and raw ECG data, with analysis of per-class and cross-lead saliency maps.

## Results Summary

All models were trained on the PTB-XL dataset (21,837 recordings, 12 leads, 100 Hz) with 5-class superdiagnostic labels: NORM, HYP, MI, CD, STTC.

### Classification Performance (Clean Data, Test Set)

| Model | Weighted F1 | Accuracy | Macro AUROC |
|------------|-------------|----------|-------------|
| **CCT** | **0.705** | **0.720** | **0.906** |
| CNN | 0.678 | 0.710 | 0.868 |
| CNN+BiLSTM | 0.663 | 0.699 | 0.878 |

### SLoC Interpretability (Clean Models)

| Model | Insertion AUC ↑ | Deletion AUC ↓ |
|------------|-----------------|----------------|
| **CCT** | **0.897** | **0.015** |
| CNN+BiLSTM | 0.889 | 0.088 |
| CNN | 0.827 | 0.111 |

The CCT achieves the best classification and interpretability scores indicating that the learned saliency maps accurately capture diagnostically relevant regions.

## Project Structure

```
src/
├── dataset.py          # PTB-XL data loading, cleaning (neurokit2), fold splitting
├── training.py         # Training loop, SLoC mask optimisation
├── metrics.py          # Classification metrics, insertion/deletion AUC, saliency analysis
├── utils.py            # Plotting, predictions, normalisation utilities
├── models/
│   ├── cct.py          # Criss-Cross Transformer (spatial + temporal attention)
│   ├── cnn.py          # ConvNet with skip connections
│   ├── cnnbilstm.py    # CNN feature extractor + BiLSTM
│   ├── sloc.py         # SLoC attribution map, mask generation, TV loss
│   └── transformer.py  # Standard transformer (baseline)
├── final_notebook.ipynb # End-to-end pipeline notebook
data/
├── raw/                # PTB-XL raw data (download separately)
├── weights/            # Saved model checkpoints
├── processed/          # Results, figures, saliency outputs
├── notebooks/          # Individual model experiment notebooks
```

## How to Run

### Requirements

* Python 3.10+
* Required packages:
  ```
  torch (with CUDA recommended for training)
  numpy
  pandas
  matplotlib
  scikit-learn
  wfdb
  neurokit2
  ast
  ```

### Setup

1. **Clone the repository** and navigate to the `src/` directory:
   ```bash
   git clone <repo-url>
   cd repo/src
   ```

2. **Install dependencies**:
   ```bash
   pip install torch numpy pandas matplotlib scikit-learn wfdb neurokit2 ast
   ```

3. **Download the PTB-XL dataset** from [PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/) and place it in `data/raw/`. Or direct zip to the 100hz samples can be found [here](https://drive.google.com/file/d/1QnkKU0TWBvdly4uU3aq_imaIKPRROynC/view?usp=sharing).

### Running the Notebook

Open `src/final_notebook.ipynb` in Jupyter or VS Code. The notebook provides an end-to-end pipeline:

1. **Select a model** - set the `model_choice` variable to `"cct"`, `"cnn"`, or `"lstm"`.
2. **Data loading** - loads PTB-XL, applies neurokit2 cleaning, and creates train/validation/test splits.
3. **Training** - trains the selected model for 50 epochs with class-weighted cross-entropy loss (or loads pre-trained weights from `data/weights/`).
4. **Evaluation** - computes classification metrics (F1, accuracy, precision, recall, AUROC).
5. **SLoC analysis** - generates attribution maps for individual samples and aggregates saliency statistics across the test set.

To skip training and use pre-trained weights, set `LOAD = True` in the notebook. Available weights:
- `data/weights/cct_clean_final.pth`
- `data/weights/lstm_clean.pth`

Soft Local Completeness Paper Citation
@InProceedings{Haddad_2025_ICCV,
    author    = {Haddad, Ziv Weiss and Barkan, Oren and Elisha, Yehonatan and Koenigstein, Noam},
    title     = {Soft Local Completeness: Rethinking Completeness in XAI},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {19794-19804}
}

