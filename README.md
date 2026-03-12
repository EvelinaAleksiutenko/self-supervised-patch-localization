# Self-Supervised Patch Localization

This project addresses the problem of **localizing a small patch within a larger image** using a **self-supervised learning approach**.

## Overview

The methodology employs a **Siamese Network** with **Cross-Correlation** and **Soft-Argmax** for self-supervised patch localization. We use CIFAR-100 (upscaled to 64×64) to simulate medical imaging data.

> 📄 For detailed methodology, see the [Technical Report](Technical_report.md).

![Streamlit Demo](asssets/output.gif)

---

## Features

- **FastAPI Inference Service** – RESTful API for patch localization
- **Streamlit Demo UI** – Interactive web interface for visualization
- **Training & Evaluation Scripts** – Complete pipeline for model development
- **Hyperparameter Tuning** – Optuna-based optimization

---

## Model Performance

| Metric             | Value              |
|--------------------|--------------------|
| Number of Params   | 9,552              |
| Mean Euclidean Distance| 0.230 px       |
| Avg Inference(GPU NVIDIA GeForce GTX 1650) | 0.49 ms/sample|
|Avg inference(CPU) | 8.26 ms/sample      |
---

## Setup

<details>
<summary><strong>1. Clone Repository</strong></summary>

```bash
git clone https://github.com/EvelinaAleksiutenko/ujp-test-assignment.git
cd ujp-test-assignment
```

</details>

<details>
<summary><strong>2. Environment Setup</strong></summary>

**Requirements:**
- Python 3.10.11
- Pipenv (recommended)

**Option A: Using Pipenv (Recommended)**

```bash
pip install pipenv
pipenv install
pipenv shell
pipenv run pip install -r requirements.txt
```

**Option B: Using venv**

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

</details>

<details>
<summary><strong>3. Configure Environment Variables</strong></summary>

Create a `.env` file in the project root:

```env
PYTHONPATH=.;src
```

</details>

<details>
<summary><strong>4. Weights & Biases (Optional)</strong></summary>

For training monitoring, create an account on [W&B](https://wandb.ai/home) and log in:

```bash
pipenv run wandb login
```

</details>

---

## Usage
> 📄 As the default the device is set to ['cpu'](), we recommend to change it to ['cuda']().
<details>
<summary><strong>Run the API</strong></summary>

Start the FastAPI inference service:

```bash
pipenv run uvicorn src.app.api:app --host 127.0.0.1 --port 8000
```

</details>

<details>
<summary><strong>Run Streamlit Demo</strong></summary>

Launch the interactive demo UI:

```bash
pipenv run streamlit run src/app/streamlit_app.py
```

</details>

<details>
<summary><strong>Run Evaluation</strong></summary>

```bash
pipenv run python src/utils/evaluate.py model.pt test_data
```

**Optional arguments:**

```bash
pipenv run python src/utils/evaluate.py checkpoints/model.pt test_data --batch-size 64 --device cpu
```

</details>

<details>
<summary><strong>Run Training</strong></summary>

First, update `checkpoint_path` in `src/config/config.py`:

```python
checkpoint_path = 'checkpoints/model.pt'
```

Then run:
```bash
pipenv run python src/utils/train.py
```

</details>

---

## Troubleshooting

<details>
<summary><strong>Import Errors: "No module named src.config"</strong></summary>

1. Ensure commands are run with `pipenv run ...` or inside `pipenv shell`
2. Confirm `.env` exists and contains `PYTHONPATH=.;src`
3. On Windows CMD, set manually:

```bat
set PYTHONPATH=.;src
```

</details>

<details>
<summary><strong>CUDA Not Available</strong></summary>

Install PyTorch with CUDA 12.6 support:

```bash
pipenv run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

</details>
