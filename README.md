# LILT
This is the unofficial PyTorch implementation of the ACL 2022 paper: "LiLT: A Simple yet Effective Language-Independent Layout Transformer for Structured Document Understanding".

## Installation
```
git clone https://github.com/hrishikeshps94/LILT.git
cd LILT
pip install -r requirements.txt
``` 

## Data

### Folder Structure
```
data
├── train
│   ├── annotations
│   │   ├── 1.json
│   │   ├── 2.json
│   │   └── ...
│   └── images
│       ├── 1.jpg
│       ├── 2.jpg
│       └── ...
└── val
    ├── annotations
    │   ├── 1.json
    │   ├── 2.json
    │   └── ...
    └── images
        ├── 1.jpg
        ├── 2.jpg
        └── ...


```
The Annotations should be in the following format:
```
{
    "file_name": "path/to/image",
    "annotations": [
        {
            "category": "class name",
            "text": "text",
            "box_line": [x1, y1, w, h] # x1, y1 are the top left coordinates of the bounding box, w is the width and h is the height
        },
       
    ]
}
```

## Usage
### Inference
```
python inference.py --file_path path-to-the-json-file --model_path path-to-the-model(ONNX/PT) --processing_mode (batch/single) --visualize True/False
```

### Training
```
Update the config file with the required parameters
python train.py
```

### ONNX Conversion
```
python onnx_converter.py
```

## Acknowledgements
- The official implementation of the paper can be found [here](https://github.com/jpwang/lilt).

## TODO
- [x] Add torch implementation of the model and load the pretrained weights (from original repo).
- [x] Add the data processing and data loading scripts.
- [x] Add the Inference script.
- [x] Add the training script.
- [x] Add the evaluation script.
- [x] Add the visualization script.
- [x] Add the ONNX conversion script.
- [x] Perfomance test between quantized and non-quantized models.
- [x] Add PDF processing script.
- [x] Add MLFlow/Weights and Biases for tracking the experiments.
- [x] Add the requirements.txt file.
- [ ] Add the Dockerfile.
- [x] Add AWS Sagemaker inference script.
- [ ] Add github actions for CI/CD.



