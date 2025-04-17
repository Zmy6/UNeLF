# UNeLF: Unconstrained Neural Light Field for Self-Supervised Angular Super-Resolution
An official code implementation of "UNeLF: Unconstrained Neural Light Field for Self-Supervised Angular Super-Resolution" from Mingyuan Zhao and Hao Sheng and Rongshan Chen and Ruixuan Cong and Zhenglong Cui and Da
Yang.

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
Stanford: 
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
@article{zhao2025unelf,
  title={UNeLF: Unconstrained Neural Light Field for Self-Supervised Angular Super-Resolution}, 
  author={Zhao, Mingyuan and Sheng, Hao and Chen, Rongshan and Cong, Ruixuan and Cui, Zhenglong and Yang, Da},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  year={2025},
  doi={10.1109/TCSVT.2025.3548705}}.
```

