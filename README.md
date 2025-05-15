# 🧬 LLMs-CancerVariant: Benchmarking Large Language Models for Cancer Variant Classification

## 📄 Paper
**Title**: Benchmarking large language models GPT-4o, Llama 3.1, and Qwen 2.5 for cancer genetic variant classification  
**Journal**: *npj Precision Oncology*  
**DOI**: [10.1038/s41698-025-00935-4](https://doi.org/10.1038/s41698-025-00935-4)  
**Authors**: Kuan-Hsun Lin, Tzu-Hang Kao, Lei-Chi Wang, Chen-Tsung Kuo, Paul Chih-Hsueh Chen, Yuan-Chia Chu*, Yi-Chen Yeh*  
(*Corresponding authors)

---

## 🎯 Objective
This project benchmarks GPT-4o, Llama 3.1, and Qwen 2.5 on cancer variant classification tasks using:

- Public databases: **OncoKB**, **CIViC**
- Real-world dataset: **FoundationOne CDx reports**

The models were evaluated on:
- Clinical relevance classification (Clinically Relevant vs VUS)
- Evidence tier classification (e.g., OncoKB levels 1–4, R1/R2; CIViC A–E)
- Response consistency across 100 iterations
- Effects of **prompt design**, **model temperature**, and **RAG (Retrieval-Augmented Generation)**

---

## 🗃️ Module Overview

| File | Function |
|------|----------|
| `XXX_all_models.py` | Benchmarking 3 LLMs with basic prompts |
| `XXX_details.py` | Evaluation using refined prompts |
| `foundationone_rag.py` | Classification with RAG (retrieval-augmented generation) |
| `foundationone_binary.py` | Binary classification: Clinically Relevant vs VUS |
| `XXX_explain.py` | LLM reasoning trace for classification decisions |
| `onco_temperature.py` | Model stability under varying temperature settings |

---

## 🔬 Key Results

| Dataset | GPT-4o | Llama 3.1 | Qwen 2.5 |
|--------|--------|-----------|----------|
| **FoundationOne** (VUS vs Relevant) | 0.7318 | 0.4976 | 0.5731 |
| **OncoKB** (Top-1 tier) | 0.3393 | 0.3066 | 0.3328 |
| **CIViC** (Top-1 tier) | 0.1865 | 0.1212 | 0.2485 |

- **Prompt engineering** greatly improved performance
- **RAG** increased accuracy for Qwen 2.5 from 0.5731 ➜ 0.6616 (FoundationOne)
- Lower **temperature (0)** improved consistency and accuracy
- GPT-4o was most aligned with **pathologist annotations**
- All models tended to **over-classify** weaker evidence variants

---

## 🧪 Dataset Info

- **FoundationOne CDx**: 10,506 variants (5,240 relevant, 5,266 VUS)
- **OncoKB**: 625 annotated variant associations
- **CIViC**: 4,426 variant-evidence entries

---

## 📦 Dependencies & Setup

- Python 3.10.12
- Qwen 2.5 & Llama 3.1 via Ollama server + 4× NVIDIA A100
- GPT-4o via Azure OpenAI API
- Retrieval vectorization: [`nomic-embed-text`](https://arxiv.org/abs/2402.01613)

---

## 🔍 Repository Structure

```
LLMs-CancerVariant/
├── foundationone_rag.py
├── foundationone_binary.py
├── XXX_all_models.py
├── XXX_details.py
├── XXX_explain.py
├── onco_temperature.py
├── prompts/
│   ├── basic/
│   ├── refined/
│   └── binary/
├── data/
│   ├── oncokb.csv
│   ├── civic.tsv
│   └── foundationone.json
└── results/
```

---

## 📜 Citation

If you use this codebase or data, please cite:

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
