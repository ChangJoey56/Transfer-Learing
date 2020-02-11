# 1. Deep Transfer Learning
## 1.1 Domain Adaptation
### 1.1.1 Single Domain 
* NIPS-19 [Towards Improving Transferability of Deep Neural Networks](https://papers.nips.cc/paper/8470-transferable-normalization-towards-improving-transferability-of-deep-neural-networks) [[official code]](http://github.com/thuml/TransNorm)
     - Assume that the loss of transferability mainly stems from the intrinsic limitation of the architecture design of DNNs
     - Propose a Transferable Normalization Techniques for Transfer Learning
### 1.1.2 Multi Domain

### 1.1.3 Feature Alignment
* AAAI-20 [Discriminative Adversarial Domain Adaptation](https://arxiv.org/abs/1911.12036)
     - Solve the limitation of aligning the joint distributions in domain-adversarial training
     - Propose an integrated category and domain classifier balancing between category and domain adaptation for any instance
* ICCV-19 [Learning Discriminative Features for Unsupervised Domain Adaptation](https://arxiv.org/abs/1910.05562) [[official code]](https://github.com/postBG/DTA.pytorch)
     - Aim at considering the tasks while matching the distribution cross domain
     - By exploiting adversarial dropout to learn strongly discriminative features by enforcing the cluster assumption
* CVPR-19 [Contrastive Adaptation Network for Unsupervised Domain Adaptation](http://openaccess.thecvf.com/content_CVPR_2019/papers/Kang_Contrastive_Adaptation_Network_for_Unsupervised_Domain_Adaptation_CVPR_2019_paper.pdf) 
     - Solve the misalignment caused by neglecting the class information
     - Propose a new metric for intra-class domain discrepancy and inter-class domain discrepancy
* CVPR-18 [Aligning Domains using Generative Adversarial Networks](http://openaccess.thecvf.com/content_cvpr_2018/papers/Sankaranarayanan_Generate_to_Adapt_CVPR_2018_paper.pdf) 
     - Aim at aligning feature distributions cross domain in a learned joint feature space
     - By inducing a symbiotic relationship between the learned embedding and a generative adversarial
network.

#### 1.1.3.1 Distribution Distance-based Methods
* CVPR-19 [A Deep Max-Margin Gaussian Process Approach](http://openaccess.thecvf.com/content_CVPR_2019/papers/Kim_Unsupervised_Visual_Domain_Adaptation_A_Deep_Max-Margin_Gaussian_Process_Approach_CVPR_2019_paper.pdf)
     - Introduce GP to minimize the maximum discrepancy of predictors
     - Easier to solve than previous methods based on adversarial minimax optimization
     
#### 1.1.3.2 Adversarial Methods
* AAAI-20 [Adversarial-Learned Loss for Domain Adaptation](https://arxiv.org/pdf/2001.01046) [[official code]](https://github.com/ZJULearning/ALDA)
     - Aim at considering the target discriminative features by combining the self-training methods and domain-adversarial methods
     - Enhance the accuracy of pseudo-labeling by introducing the confusion matrix learned through an adversarial manner
* AAAI-20 [Adversarial Domain Adaptation with Domain Mixup](https://arxiv.org/abs/1912.01805)
     - Assume that samples from two domains alone are not sufficient to ensure domain-invariance at most part of latent space
     - Assume that it's more reasonable for domain discriminator to use soft scores to evalue the generative images or features
     - Introduce mixup (DM-ADA), which guarantees domain-invariance in a more continuous latent space and guides the domain discriminator in judging samples’ difference relative to source and target domains
* CVPR-18 [Symmetric Bi-Directional Adaptive GAN](http://openaccess.thecvf.com/content_cvpr_2018/papers/Russo_From_Source_to_CVPR_2018_paper.pdf)
     - Introduce a new class consistency loss to preserve the class identity of an image passing through both domain mappings
     - By introducing a symmetric mapping among domains for bi-directional image transformations
* ICML-18 [CYCLE-CONSISTENT ADVERSARIAL DOMAIN ADAPTATION](https://arxiv.org/abs/1711.03213)
     - Aim at capturing pixel-level and low-level domain shifts
     - By exploiting a semantic consistency loss and a cycle loss
* CVPR-17 [Unsupervised Pixel–Level Domain Adaptation with Generative Adversarial Networks](http://openaccess.thecvf.com/content_cvpr_2017/papers/Bousmalis_Unsupervised_Pixel-Level_Domain_CVPR_2017_paper.pdf)
     - Aim at capturing a transformation in the pixel level
     - By exploiting a content–similarity loss
     
#### 1.1.3.3 Attention Alignment
* ECCV-18 [Deep Adversarial Attention Alignment for Unsupervised Domain Adaptation: the Benefit of Target Expectation Maximization](https://arxiv.org/abs/1801.10068)
     - Aim at transfering knowledge in all the convolutional layers through attention alignment
     - Assume that the discriminative regions in an image are relatively invariant to image style changes
     
#### 1.1.3.4 Class-awared for Target Data
* AAAI-20 [Unsupervised Domain Adaptation via Structured Prediction Based Selective Pseudo-Labeling](https://arxiv.org/abs/1911.07982)
     - Propose a novel selective pseudo-labeling strategy based on structured prediction
     - Assume that samples in the target domain are well clustered within the deep feature space
* CVPR-19 [Transferrable Prototypical Networks for Unsupervised Domain Adaptation](http://openaccess.thecvf.com/content_CVPR_2019/papers/Pan_Transferrable_Prototypical_Networks_for_Unsupervised_Domain_Adaptation_CVPR_2019_paper.pdf)
     - Propose a new DA by learning an embedding space and perform classification via a remold of the distances to the prototype of each class.
     - Assume that the score distributions predicted by prototypes separately on source and target data are similar.  


### 1.1.4 Partial Domain Adaptation
* CVPR-18 [Importance Weighted Adversarial Nets for Partial Domain Adaptation](https://arxiv.org/abs/1803.09210)
     - Aim at reducing the distribution with unidentical label spaces
     - Propose a novel adversarial nets-based partial domain adaptation method to identify the source samples that are potentially from the outlier classes
     
### 1.1.5 Multiple Distribution
* CVPR-18 [Boosting Domain Adaptation by Discovering Latent Domains](https://arxiv.org/abs/1805.01386)
     - Assume that most datasets can be regarded as mixtures of multiple domains
     - Propose a novel CNN archtecture to discover latent domains automatically
     
 ### 1.1.6 Layer-wsie DA
 * CVPR-19 [Unsupervised Domain Adaptation using Feature-Whitening and Consensus Loss](https://arxiv.org/pdf/1903.03215) [[official code]](https://github.com/roysubhankar/dwt-domain-adaptation)
     - Propose the feature whitening for domain alignment and the Min-Entropy Consensus loss for unlabeled target domain adaptation
 * CVPR-19 [Domain-Specific Batch Normalization for Unsupervised Domain Adaptation]( https://arxiv.org/pdf/1906.03950) 
     - Aim to adapt both domains by specializing batch normalization layers in CNN
     - Propose separate batch normalization layers for both domains
* ICCV-19 [An Adaptive Feature Norm Approach for Unsupervised Domain Adaptation](http://openaccess.thecvf.com/content_ICCV_2019/papers/Xu_Larger_Norm_More_Transferable_An_Adaptive_Feature_Norm_Approach_for_ICCV_2019_paper.pdf) [[official code]](https://github.com/jihanyang/AFN)
     - Assume that the erratic discrimination of the target domain mainly stems from its much smaller feature
norms
     - Solve the drastic model degradation on the target task
     - Propose a novel parameter-free Adaptive Feature Norm approach
     
### 1.1.4 Applications
#### 1.1.4.1 Semantic Segmentation
* CVPR-19 [Pixel-level Domain Transfer with Cross-Domain Consistency](https://zpascal.net/cvpr2019/Chen_CrDoCo_Pixel-Level_Domain_Transfer_With_Cross-Domain_Consistency_CVPR_2019_paper.pdf)
     - Aim at capturing pixel-level domain shifts
     - Assume that translated images cross domains differ in styles
     - By exploiting a cross-domain consistency loss motivated by image-to-image translation methods



