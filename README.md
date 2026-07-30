<div align="center">
<h1>AdaptivePS: Adaptive Planar Surface Reconstruction</h1>

</div>

> **Built on [PlanarSplatting](https://arxiv.org/abs/2412.03451)** — a great piece of work by Bin Tan, Rui Yu, Yujun Shen, and Nan Xue (Ant Group / University of Louisville).  
> AdaptivePS extends PlanarSplatting as a thesis project, adapting it for general indoor/outdoor 3D planar reconstruction under sparse and unconstrained capture conditions.

## 📝 Acknowledgements & Citations

AdaptivePS is built on top of [PlanarSplatting](https://icetttb.github.io/PlanarSplatting/). If you use this work, please also cite the original paper:

```bibtex
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

> **Requirements:** CUDA 12.8, Python 3.12

### 1. Clone PlanarSplatting
```bash
git clone https://github.com/MCHU-1999/PlanarSplatting.git --recursive
```

### 2. Create the environment
```bash
conda create -n planarSplatting python=3.12
conda activate planarSplatting
```

### 3. Install PyTorch (CUDA 12.8)
```bash
pip install torch==2.10.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 4. Install dependencies
```bash
pip install -r requirements.txt -c constraints.txt
```

### 5. Install SAM3
```bash
git clone https://github.com/MCHU-1999/sam3.git
cd sam3
pip install ninja opencv-python-headless
pip install --no-build-isolation -e ".[notebooks]"
cd ..
```

<details>
<summary>Optional: faster SAM3 inference</summary>

```bash
pip install flash-attn-3 --no-deps --index-url https://download.pytorch.org/whl/cu128
pip install --no-build-isolation git+https://github.com/ronghanghu/cc_torch.git
```
</details>

### 6. Install submodules & additional packages
```bash
pip install --no-build-isolation submodules/diff-rect-rasterization
pip install --no-build-isolation submodules/quaternion-utils
pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation
pip install git+https://github.com/MCHU-1999/DA3.git
```