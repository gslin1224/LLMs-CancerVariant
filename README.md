# 🧬 LLMs-CancerVariant: Benchmarking Large Language Models for Cancer Variant Classification

## 📄 Paper

**Title**: Benchmarking large language models GPT-4o, Llama 3.1, and Qwen 2.5 for cancer genetic variant classification  
**Journal**: *npj Precision Oncology*  
**DOI**: [10.1038/s41698-025-00935-4](https://doi.org/10.1038/s41698-025-00935-4)  
**Authors**: Kuan-Hsun Lin, Tzu-Hang Kao, Lei-Chi Wang, Chen-Tsung Kuo, Paul Chih-Hsueh Chen, Yuan-Chia Chu*, Yi-Chen Yeh*  
(*Corresponding authors)

---

## 🎯 Objective

This project benchmarks GPT-4o, Llama 3.1, and Qwen 2.5 for cancer variant classification using:

- Public databases: **OncoKB**, **CIViC**
- Real-world data: **FoundationOne CDx reports**

Evaluation tasks include:

- Clinical relevance classification (Clinically Relevant vs VUS)
- Evidence tier classification (e.g., OncoKB levels 1–4, R1/R2; CIViC A–E)
- Model consistency over 100 iterations
- Prompt design effects
- Temperature sensitivity
- Retrieval-Augmented Generation (RAG)

All prompt templates used in the experiments are available in the paper (see Supplementary Table 3).

---

## 🔬 Key Results

| Dataset | GPT-4o | Llama 3.1 | Qwen 2.5 |
|--------|--------|-----------|----------|
| **FoundationOne** (Relevant vs VUS) | 0.7318 | 0.4976 | 0.5731 |
| **OncoKB** (Top-1 tier) | 0.3393 | 0.3066 | 0.3328 |
| **CIViC** (Top-1 tier) | 0.1865 | 0.1212 | 0.2485 |

- GPT-4o was most aligned with expert annotations  
- RAG boosted Qwen 2.5's accuracy from 0.5731 → 0.6616  
- Refined prompts significantly improved classification accuracy  
- Lower temperature improved both stability and accuracy  
- All models tended to overclassify weak-evidence variants

---

## 🧪 Dataset Summary

- **FoundationOne CDx**: 10,506 variants (5,240 relevant, 5,266 VUS)  
- **OncoKB**: 625 variant-evidence associations  
- **CIViC**: 4,426 variant-evidence associations  

---

## 💻 Script Overview

- `XXX_all_models.py` - Tests the performance of three models on the respective database using basic prompts
- `XXX_details.py` - Tests the performance of three models on the respective database using refined prompts
- `foundationone_rag.py` - Conducts testing using RAG (Retrieval-Augmented Generation)
- `foundationone_binary.py` - Tests performance using a binary classification system prompt
- `XXX_explain.py` - Enables the model to explain its reasoning behind each answer
- `onco_temperature.py` - Tests model performance with temperature set to 0

---

## 📦 Environment

- Python 3.10.12  
- GPT-4o (via Azure OpenAI, 2024-05-13 version)  
- Qwen 2.5 & Llama 3.1 (via Ollama server, 4× NVIDIA A100)  
- Vector embedding: [`nomic-embed-text`](https://arxiv.org/abs/2402.01613)

---

## 📜 Citation

If you use this codebase or findings in your work, please cite:

```bibtex
@article{lin2025llms,
  title={Benchmarking large language models GPT-4o, Llama 3.1, and Qwen 2.5 for cancer genetic variant classification},
  author={Lin, Kuan-Hsun and Kao, Tzu-Hang and Wang, Lei-Chi and Kuo, Chen-Tsung and Chen, Paul Chih-Hsueh and Chu, Yuan-Chia and Yeh, Yi-Chen},
  journal={npj Precision Oncology},
  volume={9},
  number={141},
  year={2025},
  doi={10.1038/s41698-025-00935-4},
  publisher={Nature Publishing Group}
}
```
