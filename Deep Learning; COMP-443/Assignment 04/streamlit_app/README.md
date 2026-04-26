# Streamlit Deployment (Assignment 04)

This app performs inference on CIFAR-10 classes using the exported Keras models from the training notebook.

## Requirements
- Exported models placed in `models/`:
  - `mobilenetv2_cifar10_transfer.keras`
  - `densenet121_cifar10_transfer.keras`
  - `efficientnetb0_cifar10_transfer.keras`

## Run locally
From `Deep Learning; COMP-443/Assignment 04/`:

```bash
pip install -r streamlit_app/requirements.txt
streamlit run streamlit_app/app.py
```
