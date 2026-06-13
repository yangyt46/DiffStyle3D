<div align="center">
<h1>
DiffStyle3D: Consistent 3D Gaussian Stylization via Attention Optimization
</h1>

<div>
    Yitong Yang<sup>1</sup>, &ensp;
    Xuexin Liu<sup>1</sup>, &ensp;
    Yinglin Wang<sup>1†</sup>, &ensp;
    Jing Wang<sup>1</sup>, &ensp;
    Hao Dou<sup>1</sup>, &ensp;
    Changshuo Wang<sup>2</sup>, &ensp;
    Shuting He<sup>1†</sup>, &ensp;
</div>

<div>
    <sup>1</sup>School of Computing and Artificial Intelligence, Shanghai University of Finance and Economics, Shanghai, China.
    <br>
    <sup>2</sup>Department of Computer Science University College London, London, United Kingdom.<br>
    <sup>†</sup>Corresponding Author.
</div>

<sub></sub>

<p align="center">
    <span>
        <a href="https://arxiv.org/pdf/2601.19717" target="_blank"> 
        <img src='https://img.shields.io/badge/arXiv%202601.19717-DiffStyle3D-red' alt='Paper PDF'></a> &emsp;  &emsp; 
    </span>
</p>
</div>

## 💡 Overview
we propose **DiffStyle3D**, a novel diffusion-based paradigm for 3DGS style transfer that directly optimizes in the latent space. Specifically, we introduce an Attention-Aware Loss that performs style transfer by aligning style features in the self-attention space, while preserving original content through content feature alignment. Inspired by the geometric invariance of 3D stylization, we propose a Geometry-Guided Multi-View Consistency method that integrates geometric information into self-attention to enable cross-view correspondence modeling. Based on geometric information, we additionally construct a geometry-aware mask to prevent redundant optimization in overlapping regions across views, which further improves multi-view consistency.
![Overall Framework](assets/main.jpg)

## 📢 News

* **[2026-06]** We have open-sourced the code.
* **[2026-05]** Our paper is accepted by **ICML 2026**! 🎉

## 🔧 Prepare

The repository contains the 3DGS project. Please follow the commands below to install it.

```shell
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
```
You must install the environment required for 3D Gaussian Splatting. Then, follow the commands below to install our environment.

```shell
git clone https://github.com/yangyt46/DiffStyle3D.git
cd DiffStyle3D
pip install -r requirements.txt
```

## 📂 Datasets


[Tandt DB](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip)

[Mip-NeRF 360](http://storage.googleapis.com/gresearch/refraw360/360_v2.zip)

## 🚀 Run

### Reconstruction scene
Reconstruct a scene based on 3DGS, with an example as follows:

```shell
cd gaussian-splatting
python train.py -s <path to COLMAP or NeRF Synthetic dataset>
```

### Style transfer
Perform style transfer based on the reconstructed scene, with an example as follows:
```shell
bash train.sh
```
## 🎓 Citing DiffStyle3D

If you use DiffStyle3D in your research, please use the following BibTeX entry.

```
@article{yang2026diffstyle3d,
  title={DiffStyle3D: Consistent 3D Gaussian Stylization via Attention Optimization},
  author={Yang, Yitong and Liu, Xuexin and Wang, Yinglin and Wang, Jing and Dou, Hao and Wang, Changshuo and He, Shuting},
  journal={arXiv preprint arXiv:2601.19717},
  year={2026}
}
```