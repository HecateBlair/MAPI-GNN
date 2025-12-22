# MAPI-GNN: Multi-Activation Plane Interaction Graph Neural Network for Multimodal Medical Diagnosis

This repository contains the official PyTorch implementation of the paper:
**"MAPI-GNN: Multi-Activation Plane Interaction Graph Neural Network for Multimodal Medical Diagnosis"**, accepted at **AAAI 2026**.

## 📝 Abstract
Graph neural networks are increasingly applied to multimodal medical diagnosis for their inherent relational modeling capabilities. However, their efficacy is often compromised by the prevailing reliance on a single, static graph built from indiscriminate features, hindering the ability to model patient-specific pathological relationships. To this end, the proposed Multi-Activation Plane Interaction Graph Neural Network (MAPI-GNN) reconstructs this single-graph paradigm by learning a multifaceted graph profile from semantically disentangled feature subspaces. The framework first uncovers latent graph-aware patterns via a multi-dimensional discriminator; these patterns then guide the dynamic construction of a stack of activation graphs; and this multifaceted profile is finally aggregated and contextualized by a relational fusion engine for a robust diagnosis. Extensive experiments on two diverse tasks, comprising over 1300 patient samples, demonstrate that MAPI-GNN significantly outperforms state-of-the-art methods.

## 🏗️ Methodology & Framework

The proposed **MAPI-GNN** framework moves beyond the prevailing static single-graph paradigm by learning a dynamic, multifaceted graph profile for each patient. The architecture follows a rigorous **two-stage process**:

### Stage I: Multi-Activation Graph Construction (Unsupervised Manifold Learning)
This stage focuses on learning the intrinsic feature manifold and constructing patient-specific topologies without accessing label information.
* **Multi-Dimensional Feature Discriminator (MDFD)**: Utilizes a **Concept AutoEncoder** to project high-dimensional features into a disentangled semantic space. It identifies salient features via perturbation analysis to uncover latent graph-aware patterns.
* **Multi-Activation Graph Construction Strategy (MAGCS)**: Instead of a single static graph, we dynamically construct a stack of **Activation Graphs**. Each graph corresponds to a specific semantic dimension, where edges are weighted by the semantic importance learned by the MDFD.

### Stage II: Hierarchical Feature Dynamic Association (Inductive Classification)
This stage performs the final diagnosis using a hierarchical graph neural network trained in a strictly inductive manner.
* **Hierarchical Feature Dynamic Association Network (HFDAN)**:
    1.  **Intra-Sample Fusion**: A bank of **Planar Graph Encoders** (based on GAT) aggregates information within each patient's activation graphs to capture local pathological dependencies.
    2.  **Inter-Sample Fusion**: A global **Fusion-Relation Graph** connects patients based on feature similarity. A GCN then propagates information across the patient population to model global distributions and perform robust classification.

## 🛠️ Requirements
- Python >= 3.8
- PyTorch >= 1.12.0
- DGL (Deep Graph Library) >= 1.0.0
- CUDA (for GPU support)

Install dependencies via:
```bash
pip install -r requirements.txt
