<div align="center">
<h1>PlanarSplatting: Accurate Planar Surface Reconstruction in 3 Minutes</h1>

<a href="https://arxiv.org/abs/2412.03451"><img src="https://img.shields.io/badge/arXiv-2412.03451-b31b1b" alt="arXiv"></a> <a href="https://icetttb.github.io/PlanarSplatting/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a> <a href=""><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue"></a>

[Bin Tan<sup>1</sup>](https://icetttb.github.io/), [Rui Yu<sup>2</sup>](https://ruiyu0.github.io/), [Yujun Shen<sup>1</sup>](https://shenyujun.github.io/), [Nan Xue<sup>1</sup>](https://xuenan.net/)

<sup>1</sup>Ant Group  <sup>2</sup>University of Louisville

</div>


## 📝 Citations
I didn't make this.  
This is the modified PlanarSplatting for general indoor/outdoor 3D planar reconstruction work.  
If you find PlanarSplatting useful in your research or projects, please cite their work:  
```
@misc{tan2024planarsplattingaccurateplanarsurface,
      title={PlanarSplatting: Accurate Planar Surface Reconstruction in 3 Minutes}, 
      author={Bin Tan and Rui Yu and Yujun Shen and Nan Xue},
      year={2024},
      eprint={2412.03451},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.03451}, 
}
```


## ⚙️ Installation
### 1. Clone PlanarSplatting
```
git clone https://github.com/MCHU-1999/PlanarSplatting.git --recursive 
```
### 2. Create the enviroment
```
conda create -n planarSplatting python=3.10
conda activate planarSplatting

pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt 
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

pip install submodules/diff-rect-rasterization
pip install submodules/quaternion-utils

# for running on self-captured images
pip install -e submodules/vggt/
```

## 🎯 Quick Start
You can run PlanarSpaltting with an interactive demo for your own data without a GUI as follows:
```shell
python run_demo.py --data_path path/to/images
```

## 🧪 Run on COLMAP data
```shell
python run_demo_colmap.py -d path/to/colmap/data
```