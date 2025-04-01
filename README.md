# UNeLF: Unconstrained Neural Light Field for Self-Supervised Angular Super-Resolution
Compared to supervised learning methods, selfsupervised learning methods address the domain gap problem between light field (LF) datasets collected under varying acquisition conditions, which typically leads to decreased performance when differences exist in the distribution between the training and test sets. However, current self-supervised light field angular superresolution (LFASR) techniques primarily focus on exploiting discrete spatial-angular features while neglecting continuous LF information. In contrast to previous work, we propose a selfsupervised unconstrained neural light field (UNeLF) to continuously represent LF for LFASR. Specifically, any LF can be described as the camera pose for each sub-aperture image (SAI) and the two-plane that captures these SAIs. To describe the former, we introduce a SAIs-dependent pose optimization method to solve the issue that arises from the narrow baseline of most LF data, which hinders robust camera pose estimation. This mechanism reduces the number of trainable camera parameters from a quadratic to a constant scale, thereby alleviating the complexity of joint optimization. For the latter, we propose a novel adaptive two-plane parameterization strategy to determine the two-plane that captures these SAIs, facilitating refocusing. Finally, we jointly optimize the camera parameters, near-far planes and neural light field, efficiently mapping each adaptive two-plane parameterized ray to its correspondence color in a continuous manner. Comprehensive experiments demonstrate that UNeLF achieves faster training and inference with fewer computational resources while exhibiting superior performance on both synthetic and real-world datasets.

## Getting Started

We provide a environment.yml file to set up a conda environment:

### Environment

```
conda env create -f environment.yml
```

### Datasets
HCInew:
```
https://lightfield-analysis.uni-konstanz.de/
```
Stanrford: 
```
http://lightfield.stanford.edu/lfs.html
```
Please refer to nerfmm for data format

### Training and Evaluation

```
python tasks/UNeLF.py --base_dir=data/HCInew --scene_name=xxx --gpu_id 0
```

## Testing

```
python tasks/testY.py
```

## Acknowledgments

During our UNeLF implementation, we referenced several open sourced NeRF implementations, and we thank their contributions. Specifically, we referenced functions from nerf and nerfmm, and borrowed/modified code from them.

## Citation
```
@ARTICLE{zhao2025unelf,
  author={Zhao, Mingyuan and Sheng, Hao and Chen, Rongshan and Cong, Ruixuan and Cui, Zhenglong and Yang, Da},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={UNeLF: Unconstrained Neural Light Field for Self-Supervised Angular Super-Resolution}, 
  year={2025},
  pages={1-1},
  doi={10.1109/TCSVT.2025.3548705}}5.
```

