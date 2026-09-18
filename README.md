# Joint Appliance-Level Load Disaggregation and Fault Diagnosis Using Graph Neural Networks and Multi-Task Learning

This repository implements a unified deep learning framework for **smart building energy management**, combining **Non-Intrusive Load Monitoring (NILM)** and **Fault Detection and Diagnosis (FDD)**. 

By using, **Graph Neural Networks**, **Multi-Task Learning (MTL)**, and **Self-Supervised Representation Learning**, this project estimates appliance-level energy consumption from aggregate smart meter data while simultaneously detecting and classifying equipment operational faults.

---

## Project explanations

Buildings account for 30–40% of global energy consumption. Traditional Fault Detection and Diagnosis (FDD) requires expensive sub-metering and dedicated physical sensors on every appliance. Conversely, conventional NILM focuses purely on energy disaggregation, ignoring abnormal behavior and equipment degradation.

This project bridges this gap with a single, end-to-end multi-task framework capable of:
1. **Appliance-level load disaggregation** 
2. **Equipment fault detection** 
3. **Fault diagnosis & classification** (Multi-class: Refrigerant leak, bearing wear, compressor degradation, etc.)
4. **Appliance interaction modeling** via learned dynamic dependency graphs.

You can check the results of this methods of the Report section.
---

## Repository Structure

```text
stage-nilm/
├── .gitignore
├── README.md
├── requirements.txt
├── dataset/
│   ├── README.md                   # Dataset preprocessing & synthetic fault injection guidelines
│   └── raw/                        # Public datasets (UK-DALE, REFIT, REDD, AMPds)
├── src/
│   ├── code_nilm_unified.ipynb     # Interactive pipeline (SSL pretraining, GNN, MTL training)
│   ├── code_nilm_unified.py        # Main execution script
│   ├── models/                     # Architectures (Transformer, GAT, TGN, Heads)
│   └── utils/                      # Data loaders, fault injection, and evaluation metrics
└── report/
    └── article_internship_Elisa.pdf # Internship research paper / project report
