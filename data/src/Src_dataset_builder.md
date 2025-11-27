### Prepare the Pytorch-Lighting repo

`git clone https://github.com/Lightning-AI/pytorch-lightning.git`

### Run the Builder

`python pytorch_lightning_dataset_builder.py .\pytorch-lightning\ --output-format rag --output-dir ..\final_data\  --generate-docstrings`

if further filtering needed:

`python pytorch_lightning_dataset_builder.py /path/to/pytorch-lightning \
    --output-format rag \
    --path-filter "src/lightning/pytorch/trainer" \
    --path-filter "src/lightning/pytorch/callbacks"`