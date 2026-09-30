# Craniofacial Reconstruction using Generative Models

This repository contains implementations and experimental code for **craniofacial reconstruction using generative deep learning models**, with a particular focus on reconstructing facial appearance from skull/X-ray images.

The repository currently includes implementations based on the following image-to-image translation frameworks:

* **CycleGAN**
* **Pix2Pix**
* **CUT (Contrastive Unpaired Translation)**

The code is intended for research and reproducibility purposes.

---

## Repository Structure

```text
Craniofacial-reconstruction/
│
├── models/                 # Model architectures and components
├── Models/                 # Additional model implementations
│
├── train_cyclegan.py       # CycleGAN training
├── test_cyclegan.py        # CycleGAN testing
│
├── train_pix2pix.py        # Pix2Pix training
├── test_pix2pix.py         # Pix2Pix testing
│
├── CUT_train.py            # CUT training
├── CUT_test.py             # CUT testing
│
├── environment.yml         # Conda environment
└── README.md
```

---

## Methods and Original Implementations

The implementations in this repository are based on the following established generative image-to-image translation methods.

### 1. CycleGAN

**Paper:**

Zhu, J.-Y., Park, T., Isola, P., & Efros, A. A. (2017).
**Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks.**
*Proceedings of the IEEE International Conference on Computer Vision (ICCV)*.

---

### 2. Pix2Pix

**Paper:**

Isola, P., Zhu, J.-Y., Zhou, T., & Efros, A. A. (2017).
**Image-to-Image Translation with Conditional Adversarial Networks.**
*Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.

---

### 3. CUT

**Paper:**

Park, T., Efros, A. A., Zhang, R., & Zhu, J.-Y. (2020).
**Contrastive Learning for Unpaired Image-to-Image Translation.**
*European Conference on Computer Vision (ECCV)*.

---

## Related Research

The code in this repository has been used in our research on **forensic craniofacial reconstruction from skull/X-ray images**.

### Research Article

**Prasad, R. S., et al.**
*Investigating Generative AI Models for Forensic Craniofacial Reconstruction.*

---

## Citation

If you use this repository, its code, or adaptations of the implementations in your research, please cite the corresponding research article:

```bibtex
@article{prasad2025fcr,
  title={Fcr: Investigating generative ai models for forensic craniofacial reconstruction},
  author={Prasad, Ravi Shankar and Singh, Dinesh},
  journal={arXiv preprint arXiv:2508.18031},
  year={2025}
}

---
---

## Dataset

The datasets used in our experiments can be accessed from: https://github.com/singh-ml/IIT_Mandi_S2F. This repo contains only skull images with corresponding face embeddings. For access to the corresponding face image, a data sharing agreement form has to be signed between the two parties.

---

## Environment

The required software environment is provided in:

```text
environment.yml
```

Create the Conda environment using:

```bash
conda env create -f environment.yml
conda activate pytorch-img2img
```

---

---

## Acknowledgements

This repository builds upon publicly available implementations and research from the computer vision community, particularly the authors of CycleGAN, Pix2Pix, and CUT.

We thank the original authors for making their implementations available to the research community.
