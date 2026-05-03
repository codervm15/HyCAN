# HyCAN: A Hybrid Framework for ML-Based CAN Bus Intrusion Detection

> Comparative performance analysis of Machine Learning, Deep Learning, and Transformer-based models for detecting cyberattacks on in-vehicle Controller Area Network (CAN) bus.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)
![Status](https://img.shields.io/badge/status-research-green.svg)

📚 **Master's Thesis** — *HyCAN: A Hybrid Approach for CAN Bus Intrusion Detection Using Performance Analysis of Machine Learning Techniques*  
🎓 **University of Windsor** · School of Computer Science  
👨‍🔬 **Author:** Vikrant Mehla · **Supervisor:** Dr. Ikjot Saini 

---

## 🧠 Overview

Modern vehicles rely on hundreds of Electronic Control Units (ECUs) communicating over the **CAN bus** — a protocol with no built-in encryption or authentication, leaving it exposed to spoofing, injection, DoS, and replay attacks.

**HyCAN** is a unified, extensible framework that lets you train, tune, and benchmark **multiple ML, DL, and Transformer-based IDS models in one pipeline**

A key novelty: HyCAN **converts numerical CAN bus data into feature strings** so it can be processed by BERT, treating CAN traffic like a language modeling problem.

---

## ✨ Key Contributions

- 🔬 **First hybrid framework** to compare 10+ ML/DL/Transformer models on the same CAN dataset under identical conditions
- 📊 **Real-world dataset** — uses the `can-train-and-test` dataset collected from 4 real vehicles on the road
- 🤖 **BERT for CAN** — novel approach that tokenizes CAN frames as feature strings for transformer fine-tuning
- 🧱 **Stacked Ensemble** — combines KNN, DT, RF, XGBoost, and CatBoost with Logistic Regression as meta-learner
- ⏱️ **Timestamp ablation study** — quantifies how much temporal features contribute to detection accuracy (RS1 vs RS2)
- 🚗 **5 attack types** — DoS · Fuzzy · Gear Spoofing · Speed Spoofing · Standstill

---

## 🛠️ Models Implemented

| Category | Models |
|---|---|
| **Traditional ML** | Logistic Regression · KNN · Decision Tree · Random Forest · XGBoost · CatBoost |
| **Ensemble** | Stacked Generalization (5 base learners + LR meta-learner) |
| **Deep Learning** | MLP · LSTM |
| **Transformer** | BERT (`bert-base-uncased`, fine-tuned) |

---

## 🚗 Dataset

This work uses the **`can-train-and-test`** dataset — CAN bus data captured via Korlan USB2CAN cable from four vehicles driven on real roads:

- Chevrolet Impala
- Subaru Forester
- Chevrolet Silverado
- Chevrolet Traverse

Each vehicle has both attack-free and attacked traces across all 5 attack types. 100k samples per class were used for balanced training.

| Feature | Description |
|---|---|
| `timestamp` | Recorded time (seconds) |
| `arbitration_id` | CAN message identifier (HEX) |
| `data_field` | Payload value (byte) |
| `attack` | 1 = injected, 0 = normal |

---

## 🔬 Research Scenarios

To assess the importance of temporal information:

- **RS1 — With Timestamp:** full feature set, including time
- **RS2 — Without Timestamp:** only `arbitration_id` + `data_field`

This reveals which models actually *depend* on temporal patterns vs. which rely on message content.

---

## 📁 Repository Structure

```
CAN_RESEARCH/
├── CANData/                         # Root folder for all data and experiments
│   ├── CAND/                        # Dataset — CAN data per vehicle and attack type
│   ├── Examples/                    # Sample code for reference experiments
│   │
│   ├── Forester/                    # Per-vehicle experiments
│   │   ├── With_Timestamp/          # RS1 — includes timestamp feature
│   │   │   ├── DOS/
│   │   │   │   ├── Exp1-Bert-base_uncased.ipynb
│   │   │   │   ├── Exp1-Bert-base_uncased.py
│   │   │   │   ├── Exp2-Roberta_base_dos_binary.ipynb
│   │   │   │   ├── Exp2-Roberta_base_dos_binary.py
│   │   │   │   ├── Exp3-ML_Models_dos_binary.ipynb
│   │   │   │   ├── Exp3-ML_Models_dos_binary.py
│   │   │   │   ├── Exp4-MLP_Dos_binary.ipynb
│   │   │   │   ├── Exp4-MLP_Dos_binary.py
│   │   │   │   ├── Exp5-LSTM_Dos_binary.ipynb
│   │   │   │   └── Exp5-LSTM_Dos_binary.py
│   │   │   ├── Fuzzy/
│   │   │   ├── Gear/
│   │   │   ├── Speed/
│   │   │   ├── Standstill/
│   │   │   └── MultiClass.ipynb     # Multi-class classification across all attacks
│   │   │
│   │   └── Without_Timestamp/       # RS2 — excludes timestamp feature
│   │       └── ... (same structure)
│   │
│   ├── Impala/                      # Same structure as Forester
│   ├── Silverado/                   # Same structure as Forester
│   └── Traverse/                    # Same structure as Forester
│
└── README.md
```

> 💡 Each experiment is provided in **two formats**:
> - `.ipynb` — Jupyter notebook (originally run on Google Colab) with inline outputs and visualizations
> - `.py` — equivalent Python script for command-line execution and easier code review
### How experiments are organized

Each attack folder contains **5 experiments** (each available as both `.ipynb` and `.py`):

| Experiment | Models Covered |
|---|---|
| `Exp1-Bert-base_uncased` | BERT (fine-tuned for CAN classification) |
| `Exp2-Roberta_base_*_binary` | RoBERTa (transformer baseline) |
| `Exp3-ML_Models_*_binary` | Logistic Regression, KNN, Decision Tree, Random Forest, XGBoost, CatBoost, Stacked Ensemble |
| `Exp4-MLP_*_binary` | Multi-Layer Perceptron |
| `Exp5-LSTM_*_binary` | Long Short-Term Memory |

Each `With_Timestamp/` and `Without_Timestamp/` folder also contains a `MultiClass.ipynb` (and `.py`) for multi-class classification across all five attack types.

### Navigating the repo

CANData/{Vehicle}/{With_Timestamp | Without_Timestamp}/{AttackType}/{Experiment}.{ipynb|py}

### Run experiments

You can run experiments in **two ways**:

**Option 1: Jupyter / Colab (recommended for exploration)**
```bash
jupyter notebook CANData/Forester/With_Timestamp/DOS/Exp1-Bert-base_uncased.ipynb
```

**Option 2: As a Python script (recommended for reproduction or CI)**
```bash
python CANData/Forester/With_Timestamp/DOS/Exp1-Bert-base_uncased.py
```

> ⚠️ The notebooks were originally executed on **Google Colab** with GPU acceleration. For BERT and RoBERTa experiments, a GPU is strongly recommended. The `.py` versions can run anywhere with the dependencies installed.
---

## 📈 Key Findings

- **BERT** achieved **near-perfect accuracy (up to 99.9%)** in RS1 across most attacks — but at high computational cost
- **Stacked Ensemble** consistently delivered **>90% accuracy** with far better efficiency, making it the most deployable choice for real systems
- Removing the timestamp (RS2) caused **significant performance drops** for BERT and LSTM, confirming their reliance on temporal context
- **KNN** showed surprising adaptability — improving substantially in RS2 across multiple vehicles
- **Logistic Regression** consistently performed worst (~50%), highlighting the need for non-linear models

> 🔑 **Takeaway:** For real-world deployment, the **Stacked Model offers the best accuracy-to-efficiency tradeoff**. BERT is the gold standard for accuracy when compute isn't a constraint.

---

## 📊 Evaluation Metrics

All models are evaluated on:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **ROC-AUC**
- **Training & inference time** (computational efficiency)

---

## 🔭 Future Directions

- Real-time deployment and latency benchmarking on embedded ECUs
- Pre-training and quantizing open-source LLMs (e.g., LLaMA) for CAN data
- Adding **xAI** (SHAP / LIME) for interpretable IDS decisions
- Extending to **V2X** communication and **Automotive Ethernet**
- Incorporating **GANs** and unsupervised methods for zero-day detection

---

## 📜 Citation

If you use HyCAN or build on this work, please cite:

> Vikrant. (2024). *HyCAN: A Hybrid Approach for CAN Bus Intrusion Detection Using Performance Analysis of Machine Learning Techniques.* ProQuest Dissertations & Theses.

```bibtex
@mastersthesis{vikrant2024hycan,
  title     = {HyCAN: A Hybrid Approach for CAN Bus Intrusion Detection Using Performance Analysis of Machine Learning Techniques},
  author    = {Vikrant},
  year      = {2024},
  publisher = {ProQuest Dissertations \& Theses}
}
```
---
## 🙏 Acknowledgements

- Dr. **Ikjot Saini** — thesis supervisor, University of Windsor
- **Lavanya Nagaraju** — research collaborator
- **B. Lampe & W. Meng** — for the `can-train-and-test` dataset
- The University of Windsor's School of Computer Science for compute resources

---

## 📬 Contact

**Vikrant Mehla**  
Security Engineer @ Arctic Wolf · MSc, University of Windsor  
[LinkedIn](https://www.linkedin.com/in/vmehla15/) · [GitHub](https://github.com/codervm15) 
