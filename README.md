# TMI-CLNet: Triple-Modal Interaction Network for Chronic Liver Disease Prognosis From Imaging, Clinical, and Radiomic Data Fusion

This code is a **pytorch** implementation of our paper [**"TMI-CLNet: Triple-Modal Interaction Network for Chronic Liver Disease Prognosis From Imaging, Clinical, and Radiomic Data Fusion"**](https://arxiv.org/pdf/2502.00695).

## Pre-requisties
* Linux
* Python>=3.7
* NVIDIA GPU (memory>=32G) + CUDA cuDNN

## Dataset
Your dataset needs to include CT scans, radiomics, and clinical information. You may need to modify dataset.py to adapt to the dataset or refer to our data organization structure
```
└─data
    ├─ img
    │   ├─ 0.nii.gz
    │   ├─ 1.nii.gz
    │   └─ 3.nii.gz
    └─ summery.txt
```
## Usage
Run the following command for five-fold cross-validation:
```
python launch_new.py
```

## Citation
```
@article{wu2025tmi,
  title={TMI-CLNet: Triple-Modal Interaction Network for Chronic Liver Disease Prognosis From Imaging, Clinical, and Radiomic Data Fusion},
  author={Wu, Linglong and Shan, Xuhao and Ge, Ruiquan and Liang, Ruoyu and Zhang, Chi and Li, Yonghong and Elazab, Ahmed and Luo, Huoling and Liu, Yunbi and Wang, Changmiao},
  journal={arXiv preprint arXiv:2502.00695},
  year={2025}
}
```
