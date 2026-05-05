# DMGD: Train-Free Dataset Distillation with Semantic-Distribution Matching in Diffusion Models
**CVPR 2026**

[![arXiv](https://img.shields.io/badge/arXiv-260X.XXXXXX-b31b1b.svg)](https://arxiv.org/abs/260X.XXXXXX)
[![CVPR](https://img.shields.io/badge/CVPR-2026-blue.svg)]([https://cvpr.thecvf.com/](https://cvpr.thecvf.com/virtual/2026/poster/36526))
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/your-username/DMGD)](https://github.com/your-username/DMGD)

> **Official Implementation** of Dual Matching Guided Diffusion (DMGD), a training-free diffusion-based dataset distillation framework, outperforming state-of-the-art fine-tuning methods on ImageNet-1K and its subsets.

### Code Release Note
**Full code will be released before the CVPR 2026 conference**
## ✨ Key Highlights
- **🚀 Fully Training-Free**: Introduces guidance only during the diffusion sampling process, requiring no additional fine-tuning or training stages, delivering orders of magnitude higher computational efficiency
- **🎯 Dual Matching Mechanism**: Decouples dataset distillation into two independent objectives: **Semantic Matching** and **Distribution Matching**, with theoretical guarantees on the risk upper bound of the distilled dataset
- **🌈 Diversity Enhancement**: Proposes a dynamic soft label guidance mechanism that significantly improves the diversity of synthetic data while maintaining semantic alignment
- **📊 Optimal Transport Distribution Alignment**: Designs a distribution matching loss based on optimal transport theory, and proposes two acceleration strategies (Distribution Approximate Matching, Greedy Progressive Matching) for efficient global distribution structure alignment
Our complete framework is illustrated in fig2.

## 📊 Experimental Results
### Performance Comparison on ImageNet Subsets (Hard-label Protocol, ResNet10-AP Top-1 Accuracy)
table1 shows the comparison results between our method and state-of-the-art methods on ImageNet-Woof and ImageNet-Nette. Our method achieves the best performance under all IPC settings and can be used as a plug-and-play module to improve the performance of existing methods.

| IPC | ImageNet-Woof | | | ImageNet-Nette | | |
| --- | --- | --- | --- | --- | --- | --- |
| 10 | 20 | 50 | 10 | 20 | 50 |
| Random | 29.4±0.8 | 32.7±0.4 | 47.2±1.3 | 54.2±1.6 | 63.5±0.5 | 76.1±1.1 |
| DM [82] | 30.3±1.2 | 35.2±0.6 | 47.1±1.1 | 60.8±0.6 | 66.5±1.1 | 76.2±0.4 |
| DiT [48] | 34.7±0.5 | 41.1±0.8 | 49.3±0.2 | 59.1±0.7 | 64.8±1.2 | 73.3±0.9 |
| MiniMax [22] | 39.2±1.3 | 45.8±0.5 | 56.3±1.0 | 62.0±0.2 | 66.8±0.4 | 76.6±0.2 |
| MGD3 [8] | 40.4±1.9 | 43.6±1.6 | 56.5±0.8 | 66.4±2.4 | 71.2±0.5 | 79.5±1.3 |
| **Ours** | **40.8±1.1** | **46.7±1.4** | **60.1±0.8** | **68.4±0.2** | **72.6±0.6** | **80.6±0.5** |
| Full dataset | | 87.5±0.5 | | | 94.6±0.5 | |



### Code Release Note
**Full code will be released before the CVPR 2026 conference**

## 📝 Citation
If you find our work useful in your research, please cite our paper:
```bibtex
@inproceedings{wang2026dmgd,
  title={DMGD: Train-Free Dataset Distillation with Semantic-Distribution Matching in Diffusion Models},
  author={Wang, Qichao and Lu, Yunhong and Cao, Hengyuan and Zhang, Junyi and Zhang, Min},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={XXXX--XXXX},
  year={2026}
}
```

## 🙏 Acknowledgments
This work was supported by the National Major Science and Technology Projects (Grant No. 2022ZD0117000), the National Natural Science Foundation of China (Grant No. 62202426), and the Shanghai Institute for Mathematics and Interdisciplinary Sciences (SIMIS) Fund (Grant No. SIMIS-ID-2025-AD).

We thank the contributors of the following open-source projects:
- [DiT](https://github.com/facebookresearch/DiT): Scalable diffusion models with transformers
- [diffusers](https://github.com/huggingface/diffusers): Diffusion models toolbox
- [Minimax-DD](https://github.com/gujianyang/Minimax-DD): Minimax-based dataset distillation method

## 📄 License
This project is open-sourced under the MIT License. See the [LICENSE](LICENSE) file for details.

---

⭐ If you like this project, please give us a star! Your support is our motivation for continuous improvement. If you have any questions or suggestions, please contact us via GitHub Issues.

---
**File Name**: `README.md`  
**Encoding**: UTF-8  
**Compatible with**: GitHub, GitLab, Gitee and all Markdown renderers
