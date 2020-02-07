# 1. Deep Transfer Learning
## 1.1 Domain Adaptation
### 1.1.1 Single Domain 
* NIPS-19 [Towards Improving Transferability of Deep Neural Networks](https://papers.nips.cc/paper/8470-transferable-normalization-towards-improving-transferability-of-deep-neural-networks) [[official code]](http://github.com/thuml/TransNorm)
     - Assume that the loss of transferability mainly stems from the intrinsic limitation of the architecture design of DNNs
     - Propose a Transferable Normalization Techniques for Transfer Learning
### 1.1.2 Multi Domain
### 1.1.3 Feature Alignment
* ICCV-19 [Learning Discriminative Features for Unsupervised Domain Adaptation](https://arxiv.org/abs/1910.05562) [[official code]](https://github.com/postBG/DTA.pytorch)
     - Aim at considering the tasks while matching the distribution cross domain
     - By exploiting adversarial dropout to learn strongly discriminative features by enforcing the cluster assumption
* CVPR-18 [Aligning Domains using Generative Adversarial Networks](http://openaccess.thecvf.com/content_cvpr_2018/papers/Sankaranarayanan_Generate_to_Adapt_CVPR_2018_paper.pdf) 
     - Aim at aligning feature distributions cross domain in a learned joint feature space
     - By inducing a symbiotic relationship between the learned embedding and a generative adversarial
network.
#### 1.1.3.1 Distribution Distance-based Methods
#### 1.1.3.2 Adversarial Methods
* ICML-18 [CYCLE-CONSISTENT ADVERSARIAL DOMAIN ADAPTATION](https://arxiv.org/abs/1711.03213)
     - Aim at capturing pixel-level and low-level domain shifts
     - By exploiting a semantic consistency loss and a cycle loss
* CVPR-17 [Unsupervised Pixel–Level Domain Adaptation with Generative Adversarial Networks](http://openaccess.thecvf.com/content_cvpr_2017/papers/Bousmalis_Unsupervised_Pixel-Level_Domain_CVPR_2017_paper.pdf)
     - Aim at capturing a transformation in the pixel level
     - By exploiting a content–similarity loss
### 1.1.4 Partial Domain Adaptation
* CVPR-18 [ImportanceWeighted Adversarial Nets for Partial Domain Adaptation](https://arxiv.org/abs/1803.09210)
     - Aim at reducing the distribution with unidentical label spaces
     - Propose a novel adversarial nets-based partial domain adaptation method to identify the source samples that are potentially from the outlier classes
### 1.1.4 Applications
#### 1.1.4.1 Semantic Segmentation
* CVPR-19 [Pixel-level Domain Transfer with Cross-Domain Consistency](https://zpascal.net/cvpr2019/Chen_CrDoCo_Pixel-Level_Domain_Transfer_With_Cross-Domain_Consistency_CVPR_2019_paper.pdf)
     - Aim at capturing pixel-level domain shifts
     - Assume that translated images cross domains differ in styles
     - By exploiting a cross-domain consistency loss motivated by image-to-image translation methods



