# GC2 Glaze Retrieval — Local Streamlit App + Jupyter Notebook

This project runs **locally** on macOS or Windows.

- **Streamlit (recommended):** `glaze_retrieval_app.py` — upload an image or choose a sample, segment pottery, and search for recipes.
- **Jupyter Notebook (alternative):** `glaze_retrieval.ipynb` — specify an image file path and run the same segmentation/retrieval code in cells.

Both interfaces share `glazeprep/`, `segmentation_frontend.py`, `retrieval_service.py`, `retrieval_assets/` and `sample_images/`. Python **3.11** with a virtual environment is recommended. You only need **one** virtual environment for both interfaces.

## macOS (Apple Silicon) — first-time installation

Creates the python virtual environment and installs requirements.
In Terminal, `cd` to the extracted project folder, then:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision
python -m pip install -r requirements-macos.txt
python -m pip check
```

If using the  **Streamlit app**, to start the app:

```bash
source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1
python -m streamlit run glaze_retrieval_app.py
```

If using the **notebook**:

```bash
source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1
python -m notebook glaze_retrieval.ipynb
```

## Windows (PowerShell) — first-time installation

Creates the python virtual environment and installs requirements.
Install Python 3.11, open PowerShell in the extracted project folder, then:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and retry.

For **NVIDIA CUDA**, install the matching PyTorch wheels from https://pytorch.org/get-started/locally/ . Otherwise use:

```powershell
python -m pip install torch torchvision
```

Then:

```powershell
python -m pip install -r requirements-windows.txt
python -m pip check
python -m streamlit run glaze_retrieval_app.py
```

Or launch the notebook:

```powershell
python -m notebook glaze_retrieval.ipynb
```

## Notebook-only installation

If you **only** want the notebook, create and activate a Python 3.11 virtual environment as above, install PyTorch for your hardware, then:

```bash
python -m pip install -r requirements-notebook.txt
python -m notebook glaze_retrieval.ipynb
```

The notebook reads a local image path via `IMAGE_PATH`. Run cells in order and configure object, colour, surface, cone and result count. The regular and diversified recipe sets are both retrieved for comparison.

## Customization and assets

`demo_content.json` is used for customizing the Streamlit app: title, introduction, font, button colour, sample descriptions, displayed recipe fields and diversification text. Demo images available in `sample_images/`.

The bundled `retrieval_assets/shared_space_heads.pt` is loaded with `weights_only=True`; keep `retrieval_assets/` intact. Pretrained segmentation/encoder models are normally cached by Hugging Face in `~/.cache/huggingface/hub` (or `HF_HOME`) and may download on first use.

## Troubleshooting

- **Notebook uses the wrong interpreter:** run `import sys; print(sys.executable)` in a cell and select your `.venv` kernel.
- **GPU check:** `torch.backends.mps.is_available()` for Apple Silicon; `torch.cuda.is_available()` for Windows NVIDIA. CPU is supported but may be slow for SAM 2.1 Large.
- **First run:** allow time and disk space for model downloads.

### Model loading and reuse (Streamlit)

The segmentation models (Grounding DINO, SAM 2.1, Ceramic CLIP) are **lazy-loaded**
only when **Detect pottery** is first clicked. Streamlit's `st.cache_resource`
keeps the initialized segmenter in server memory across subsequent uploaded
images, sample-image changes and widget reruns. A shared lock serializes
segmentation requests from multiple sessions. The retrieval encoder is also
cached separately and loads on the first recipe search. Selecting another
photo does **not** reload either model group.

The models will initialize again if you restart the Streamlit server, clear
Streamlit's resource cache, or change the cached function's source code. The
Hugging Face model download cache is separate: it normally persists on disk
across app restarts, even though in-memory models must be initialized again.
