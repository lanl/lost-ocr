# Lost in OCR Translation: Vision-Based Approaches to Robust Document Retrieval

[![Paper](https://img.shields.io/badge/arXiv-2505.05666-b31b1b.svg)](https://arxiv.org/abs/2505.05666)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📄 Overview

This repository contains implementation code for our paper **“Lost in OCR Translation? Vision-Based Approaches to Robust Document Retrieval”**. We study how **document degradation** impacts retrieval across **OCR-based**, **vision-only**, and **hybrid** systems, and we provide utilities to evaluate retrieval and QA robustness.

### Key Findings (paper highlights)
- Vision-based models show **stronger robustness** to degradation than pure OCR pipelines.
- OCR quality has a **direct, measurable effect** on downstream retrieval and QA.
- **Hybrid** approaches can exploit both textual and visual cues and often perform best.

> Paper: arXiv:2505.05666 (May 2025).

---

## 🏗️ Architecture

We compare three settings:

1. **OCR-Based Retrieval** — OCR (Nougat / LLaMA Vision) → chunk & embed text → retrieve → generate.
2. **Vision-Based Retrieval** — image encoders (e.g., ColPali/ColQwen via ViDoRe) → retrieve directly from visuals.
3. **Hybrid** — combine visual & textual features.

---

## 📦 Installation

### Prerequisites
- Python **3.10+**
- CUDA-capable GPU recommended
- ≥16GB RAM suggested for larger models

### Setup
```bash
# Clone your repo
git clone https://github.com/yourusername/lost-in-ocr-translation.git
cd lost-in-ocr-translation

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
This project ships a minimal `requirements.txt` generated from the codebase. Core libs include:
- `torch`, `transformers`, `sentence-transformers`
- `vidore-benchmark` (vision-only retrieval baselines)
- `pandas`, `numpy`, `scikit-learn`, `tqdm`
- `python-Levenshtein`, `nltk`, `Pillow`, `rouge_score`, `openai`

> If you prefer pinned versions, a typical tested set is:
> ```txt
> torch>=1.9.0
> transformers>=4.30.0
> sentence-transformers>=2.2.0
> datasets>=2.0.0
> pandas>=1.3.0
> numpy>=1.21.0
> scikit-learn>=0.24.0
> rouge-score>=0.1.2
> nltk>=3.6.0
> Levenshtein>=0.20.0
> Pillow>=8.0.0
> tqdm>=4.62.0
> openai>=0.27.0
> vidore-benchmark>=0.1.0
> ```

---

## 🗂️ Dataset Structure

Experiments expect documents organized by distortion difficulty:

```
data/
├── easy/              # Low distortion
├── medium/            # Medium distortion
├── hard/              # High distortion
│   └── {doc_folder}/
│       ├── images/      # Page images (inputs)
│       ├── llama_txt/   # LLaMA Vision OCR outputs (this repo)
│       └── mmd/         # Nougat OCR outputs (markup)
├── pdf_labels.csv       # Distortion scores per document
└── Expanded_Pages.csv   # Page metadata
```

---

## 🚀 Usage

> **Important:** The current scripts are **config-file style** (no CLI flags). Open each script and edit the paths/parameters near the top (e.g., `root_dir`, output folders). Then run with `python <script>.py`.

### 1) OCR Extraction (LLaMA Vision via SambaNova)
Set your API key and root directory:
```bash
export OPENAI_API_KEY="YOUR_SAMBANOVA_API_KEY"
# In samba_llama.py, set: root_dir = "/path/to/data" (or pass via env inside main)
python samba_llama.py
```
This will traverse `data/{easy,medium,hard}/.../images/*.png`, extract text with **Llama‑3.2‑90B‑Vision‑Instruct**, and write `llama_txt/*.txt` files (with automatic resume & retry).

### 2) Nougat OCR + RAG
`nougat_rag.py` orchestrates OCR (Nougat) and a simple RAG pipeline. Edit the input and output directories inside the script (e.g., PDF root, index/output directories), then:
```bash
python nougat_rag.py
```

### 3) Vision-only Retrieval (ViDoRe retrievers)
`vidore_vlm.py` runs OCR-free retrieval using **vidore-benchmark** retrievers (e.g., ColPali/ColQwen). Set the dataset name and `k` in the script:
```bash
python vidore_vlm.py
```

### 4) QA Pair Generation
Create synthetic/assisted Q&A from your corpus with `new_qa_datagen.py` (configure `root_dir` and `output_dir` inside):
```bash
python new_qa_datagen.py
```

### 5) Evaluation & Analysis
- Retrieval/QA metrics for LLaMA-based RAG: `rag_llama_measurement.py`  
- Reference RAG pipeline (index→retrieve→generate): `rag.py`  
- OCR quality vs. distortion: `levenshtein_per_page_by_degradation.py`

Each script prints concise summaries (e.g., **nDCG@5**, **Recall@k**, **EM/F1**, and per-page **Levenshtein** by distortion).

> **Tip:** If you prefer a full CLI, you can wrap these scripts with `argparse`. We kept code minimal to make the pipeline easy to read/modify.

---

## 📊 Experiments

### Distortion Levels
We bin documents using four levels (example thresholds):
- **Level 0**: pristine (score < 1)
- **Level 1**: low (1 ≤ score < 2)
- **Level 2**: medium (2 ≤ score < 3)
- **Level 3**: high (score ≥ 3)

### Metrics
- **Retrieval**: MRR, Recall@k, NDCG@5  
- **OCR Quality**: Levenshtein distance (page-level)  
- **Answer Quality**: ROUGE, Exact Match, token-level F1  
- **Performance**: throughput / latency (optional logs)

> Some numbers below are illustrative; reproduce exact results by aligning your dataset and seeds with the paper’s setup.

| Approach | Clean Docs | High Distortion | Relative Drop |
|----------|------------|-----------------|---------------|
| OCR‑RAG  | 0.82       | 0.45            | −45%          |
| Vision   | 0.78       | 0.71            | −9%           |
| Hybrid   | 0.85       | 0.74            | −13%          |

---

## 🔬 Key Scripts

| Script | Description |
|--------|-------------|
| `samba_llama.py` | LLaMA Vision OCR extraction via SambaNova (resumable, backoff, organized outputs) |
| `nougat_rag.py` | Nougat OCR + RAG orchestration; evaluate retrieval under degradation |
| `vidore_vlm.py` | Vision-only retrieval using **vidore-benchmark** retrievers |
| `llama_vidore.py` | LLaMA-based evaluation on ViDoRe datasets |
| `rag_llama_measurement.py` | Metrics for RAG pipelines (nDCG@k, Recall@k, EM/F1) |
| `rag.py` | Reference RAG pipeline (index / retrieve / generate) |
| `new_qa_datagen.py` | Generate Q&A pairs from documents |
| `levenshtein_per_page_by_degradation.py` | OCR quality analysis by degradation level |
| `nougat_vidore.py` | Utilities to bridge Nougat outputs and ViDoRe format |

---

## 🛠️ Configuration Snippets

```python
# Example: samba_llama.py
api_key = os.getenv("OPENAI_API_KEY")  # recommended
root_dir = "/path/to/data"
k = 5  # retrieval depth where relevant

# Example: new_qa_datagen.py
root_dir = "/path/to/data"
output_dir = "./qa_pairs_output_llama"
```

---

## 📝 Citation

If you use this code, please cite:
```bibtex
@inproceedings{10.1145/3704268.3742698, author = {Most, Alexander and Winjum, Joseph and Jones, Shawn and Ranasinghe, Nishath Rajiv and Biswas, Ayan,  O'Malley, Dan and Bhattarai, Manish}, title = {Lost in OCR Translation? Vision-Based Approaches to Robust Document Retrieval}, year = {2025}, isbn = {9798400713514}, publisher = {Association for Computing Machinery}, address = {New York, NY, USA}, url = {https://doi.org/10.1145/3704268.3742698}, doi = {10.1145/3704268.3742698}, booktitle = {Proceedings of the 2025 ACM Symposium on Document Engineering}, articleno = {13}, numpages = {10}, location = {Nottingham, United Kingdom}, series = {DocEng '25} }
```

---

## 🤝 Contributing

PRs are welcome! Please open an issue to discuss substantial changes.


## 🙏 Acknowledgments

- **ViDoRe benchmark** for vision-only retrieval evaluation
- **SambaNova** for LLaMA Vision API access
- **Nougat OCR** for PDF-to-markup extraction

---



## Copyright notice
© 2025. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
 
(End of Notice)



 
## License

 
This program is Open-Source under the BSD-3 License.
 
Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 
Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 
Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 
Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 
(End of Notice)