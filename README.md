# Graph Learning for Molecular Data: Classification & Generation


Collaborative project by **Achref Soua** and **[Khalil Braham](https://github.com/khalilbraham)**, focusing on both **graph classification** and **molecular graph generation**.

---
## Project Description

This project explores **graph machine learning** for two fundamental challenges: **graph classification** and **molecular graph generation**.  

- **Graph Classification** was studied on both **graph-level (PROTEINS dataset)** and **node-level (CORA dataset)** tasks.  
  We benchmarked **Graph Convolutional Networks (GCN)**, **Graph Attention Networks (GAT)**, and their **ensembles**, while systematically testing a range of **graph augmentation strategies** (feature noise, dropout, edge perturbations, node dropping, random walks, k-hop subgraphs).  
  Results show that augmentations and ensembles enhance robustness, with **GCN+GAT ensembles achieving the best overall performance** across both binary and multiclass tasks.

- **Molecular Graph Generation** was performed on the **ZINC dataset** using multiple generative paradigms: **GraphGAN, GraphVAE, DiGress-lite (diffusion-based), and GraphRNN**.  
  We evaluated generated molecules with **validity, novelty, and diversity metrics**, highlighting the trade-offs between models:  
  - **GraphVAE** and **GraphRNN** excel in **validity and novelty (100%)**.  
  - **DiGress-lite** achieves the **highest diversity (~78%)**, best matching real molecular properties.  
  - **GraphGAN** generates realistic fragments but struggles with consistency.  

  Together, these findings suggest **diffusion or VAE for exploration** and **GraphRNN for guaranteed validity**, making them complementary approaches.

This project delivers a **comprehensive comparative study** that bridges **classification** and **generation** in graph learning, with practical insights for bioinformatics and cheminformatics applications.  
A detailed **15-page report** is included, covering methodology, experiments, ablations, and discussions.

---
##  Project Highlights

###  Part 1: Graph Classification
- **Datasets**:
  - **PROTEINS** → Binary classification (protein vs. non-protein)
  - **CORA** → Multiclass classification (scientific paper categories)
- **Models**:
  - Graph Convolutional Network (GCN)
  - Graph Attention Network (GAT)
  - GCN + GAT Ensembles
- **Augmentation Strategies**:
  - Feature noise, dropout, edge perturbations, node mixing, random walk subgraphs, k-hop extraction
- **Metrics**:
  - Test Accuracy, Macro F1, Parameter count

#### Classification Performance — CORA (Multiclass)
| Model               | Augmentation        | Test Accuracy | Macro F1 | Parameters |
|---------------------|---------------------|---------------|----------|------------|
| Baseline (LogReg)   | N/A                 | 0.473         | 0.471    | N/A        |
| GCN                 | None                | 0.818         | 0.805    | 92k        |
| GAT                 | None                | 0.826         | 0.816    | 92k        |
| GCN                 | Feature Noise       | 0.738         | 0.722    | 92k        |
| GAT                 | Feature Noise       | 0.519         | 0.519    | 92k        |
| GCN                 | Feature Dropout     | 0.821         | 0.809    | 92k        |
| GAT                 | Feature Dropout     | 0.799         | 0.787    | 92k        |
| GCN                 | Edge Dropout        | 0.821         | 0.810    | 92k        |
| GAT                 | Edge Dropout        | 0.819         | 0.811    | 92k        |
| GCN                 | Add Random Edges    | 0.822         | 0.810    | 92k        |
| GAT                 | Add Random Edges    | 0.806         | 0.806    | 92k        |
| GCN                 | Node Mixing         | 0.819         | 0.808    | 92k        |
| GAT                 | Node Mixing         | 0.823         | 0.815    | 92k        |
| **GCN+GAT Ensemble**| None                | **0.829**     | **0.817**| N/A        |
| GCN+GAT Ensemble    | Feature Dropout     | 0.823         | 0.812    | N/A        |
| **GCN+GAT Ensemble**| Edge Dropout        | **0.831**     | **0.817**| N/A        |

#### Classification Performance — PROTEINS (Binary)
| Model               | Augmentation              | Test Accuracy | Macro F1 | Parameters |
|---------------------|---------------------------|---------------|----------|------------|
| Baseline (LogReg)   | N/A                       | 0.714         | 0.547    | N/A        |
| GCN                 | None                      | **0.804**     | **0.764**| 4.5k       |
| GAT                 | None                      | 0.741         | 0.713    | 36k        |
| GCN                 | Node Dropping             | **0.804**     | **0.764**| 4.5k       |
| GAT                 | Node Dropping             | 0.795         | 0.755    | 36k        |
| GCN                 | Random Walk Subgraph      | 0.786         | 0.742    | 4.5k       |
| GAT                 | Random Walk Subgraph      | 0.777         | 0.738    | 36k        |
| GCN                 | Edge Perturbation         | 0.777         | 0.729    | 4.5k       |
| GAT                 | Edge Perturbation         | **0.804**     | **0.768**| 36k        |
| GCN                 | Feature Masking           | 0.777         | 0.724    | 4.5k       |
| GAT                 | Feature Masking           | 0.777         | 0.738    | 36k        |
| GCN+GAT Ensemble    | None                      | 0.777         | 0.742    | 40k        |
| GCN+GAT Ensemble    | Random Walk Subgraph      | **0.804**     | **0.764**| 40k        |
| GCN+GAT Ensemble    | Node Dropping             | 0.786         | 0.747    | 40k        |

---

###  Part 2: Molecular Graph Generation
- **Dataset**: ZINC subset
- **Models**:
  - GraphGAN
  - GraphVAE
  - DiGress-Lite
  - GraphRNN

#### Generation Performance Comparison
| Model         | Validity | Novelty | Diversity | Typical Failure Mode                          |
|---------------|----------|---------|-----------|-----------------------------------------------|
| GraphGAN      | 84–91%   | 92–97%  | 70–74%    | Hydrophobic bias, ring inconsistencies         |
| GraphVAE      | 100.0%   | 100.0%  | 76.7%     | Under-models fused/aromatic rings              |
| DiGress-Lite  | >90%     | ~80%    | ~78%      | Sparse or broken macrocycles (rare)            |
| GraphRNN      | 100.0%   | 100.0%  | 64.5%     | Conservative motif reuse                       |

---

##  Project Structure

```
.
├── Classification/
│   ├── binary_classification.ipynb
│   ├── multiclass_Classification.ipynb
│
├── Generation/
│   ├── graph_gan_vae_digress.ipynb/
│   ├── graphrnn.ipynb/
│
├── report.pdf
└── README.md 

```

---

##  Setup & Usage

```bash
git clone https://github.com/achref-soua/Graph-Learning-for-Molecular-Data--Classification---Generation.git
cd project-repo
conda create -n graphlearn python=3.11
conda activate graphlearn
pip install torch torch-geometric rdkit-pypi scikit-learn matplotlib numpy
```

### Classification
- Run `binary_classification.ipynb` or `multiclass_Classification.ipynb` (with/without augmentation).

### Generation
- Run `graph_gan_vae_digress.ipynb` or `graphrnn.ipynb`.


### Report
A detailed 15-page report with full methodology, experiments, and comparative study is included in this project.

---

##  Contact
Developed by **Achref Soua** and **[Khalil Braham](https://github.com/khalilbraham)**.

Contributions are welcome—feel free to open issues or pull requests!
