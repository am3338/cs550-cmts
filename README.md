# cs550-cmts

NOTE: Repository built on ARMD (https://github.com/daxin007/ARMD)

The repository for the Rutgers CS550: Massive Data Mining Course -  Group 16:
- Alen Mrdovic

## Requirements

1. Python >= 3.10
2. Install requirements using:
 `pip install -r requirements.txt`

## Dataset Preparation
The datasets (ETT, Solar Energy and Exchange) can be obtain from (https://github.com/thuml/iTransformer ), and the dataset Stock can be obtained from (https://github.com/Y-debug-sys/Diffusion-TS ). Please put them to the folder ./Data/datasets in our repository.


### Training and sampling

Run 
`python main.py --config_path ./Config/[dataset].yaml`

Where [dataset] = "etth1", "etth2", "ettm1", "ettm2", "exchange" or "stock"
