# 1.Conference/Journal

## 1.1 Single Domain 
* NIPS-19 [Towards Improving Transferability of Deep Neural Networks](https://papers.nips.cc/paper/8470-transferable-normalization-towards-improving-transferability-of-deep-neural-networks) [[official code]](http://github.com/thuml/TransNorm)
     - Assume that the loss of transferability mainly stems from the intrinsic limitation of the architecture design of DNNs
     - Propose a Transferable Normalization Techniques for Transfer Learning

### 1.1.3 Feature Alignment

* AAAI-20 [Bi-Directional Generation for Unsupervised Domain Adaptation
](https://arxiv.org/abs/2002.04869) 
     - To preserve intrinsic data structure
     - Introduce consistent classifiers interpolating two intermediate domains to bridge source and target domains

* AAAI-20 [Correlation-aware Adversarial Domain Adaptation and Generalization](https://www.sciencedirect.com/science/article/pii/S003132031930425X) 
     - Consider correlation alignment along with adversarial learning
     - By incorporating the correlation alignment module along with adversarial learning

* TIP-19 [Locality Preserving Joint Transfer for Domain Adaptation](https://ieeexplore.ieee.org/abstract/document/8746823/) 
     - The manifold structures of samples can be preserved by taking the local consistency between samples into consideration
     - Introduce two Laplacian graph terms, one for each domain, by deploying the Fisher criterion (samples from same class stay close, while samples from different class stay far from each other)

* TPMI-18 [Aggregating Randomized Clustering-Promoting Invariant Projections for Domain Adaptation](https://ieeexplore.ieee.org/document/8353356) 
     - Considering intra-domain structure 
     - Develop a ‘sampling-andfusion’ framework , where various randomized coupled domain subsets are sampled for multiple projections.




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

 * ICDM-19 [Transfer Learning with Dynamic Adversarial Adaptation Network](https://arxiv.org/abs/1909.08184) 
     - Aim at aligning both the marginal (global) and conditional (local) distributions cross domains
     - By inducing global and local A-distance to construct the weight between these two distributions
     
* CVPR-18 [Aligning Domains using Generative Adversarial Networks](http://openaccess.thecvf.com/content_cvpr_2018/papers/Sankaranarayanan_Generate_to_Adapt_CVPR_2018_paper.pdf) 
     - Aim at aligning feature distributions cross domain in a learned joint feature space
     - By inducing a symbiotic relationship between the learned embedding and a generative adversarial
network.

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
     
#### 1.1.3.4 Class-wised for Target Data
* CVPR-19 [Contrastive Adaptation Network for Unsupervised Domain Adaptation](http://openaccess.thecvf.com/content_CVPR_2019/papers/Kang_Contrastive_Adaptation_Network_for_Unsupervised_Domain_Adaptation_CVPR_2019_paper.pdf) 
     - Solve the misalignment caused by neglecting the class information
     - Propose a new metric for intra-class domain discrepancy and inter-class domain discrepancy
     
* AAAI-20 [Unsupervised Domain Adaptation via Structured Prediction Based Selective Pseudo-Labeling](https://arxiv.org/abs/1911.07982)
     - Propose a novel selective pseudo-labeling strategy based on structured prediction
     - Assume that samples in the target domain are well clustered within the deep feature space

* AAAI-20 [Discriminative Adversarial Domain Adaptation](https://arxiv.org/abs/1911.12036)
     - Solve the limitation of aligning the joint distributions in domain-adversarial training
     - Propose an integrated category and domain classifier balancing between category and domain adaptation for any instance

* TPR-20 [Deep Conditional Adaptation Networks and Label Correlation](https://www.sciencedirect.com/science/article/pii/S0031320319303735)
     - Assume that the posterior distribution of target samples is similar to that of corresponding samples with same categories.
     - Propose a label correlation transfer algorithm by inducing KL distance to measure the proximity between source posterior distribution and target posterior distribution   

* CVPR-19 [Transferrable Prototypical Networks for Unsupervised Domain Adaptation](http://openaccess.thecvf.com/content_CVPR_2019/papers/Pan_Transferrable_Prototypical_Networks_for_Unsupervised_Domain_Adaptation_CVPR_2019_paper.pdf)
     - Propose a new DA by learning an embedding space and perform classification via a remold of the distances to the prototype of each class.
     - Assume that the score distributions predicted by prototypes separately on source and target data are similar. 
     
* ICCV-19 [Learning Discriminative Features for Unsupervised Domain Adaptation](https://arxiv.org/abs/1910.05562) [[official code]](https://github.com/postBG/DTA.pytorch)
     - Aim at considering the tasks while matching the distribution cross domain
     - By exploiting adversarial dropout to learn strongly discriminative features by enforcing the cluster assumption

* KNOSY-19 [Generating target data with class labels for unsupervised domain adaptation](https://www.sciencedirect.com/science/article/pii/S0950705119300772)
     - To implement discriminative representations of target domain, regardless of lacking labeled target data
     - Disentangle the class and the style codes of the target generator and generate Target samples with given class labels
     - By enforcing high mutual information between the class code and the generated images, which is contained the high level layers of the target generator and the source generator

* TMM-19 [Deep Multi-Modality Adversarial Networks for Unsupervised Domain Adaptation](https://ieeexplore.ieee.org/abstract/document/8656504/)
     - To learn semantic multimodality representations
     - Apply stacked attention and adversarial trainning 
     - Employ multi-channel constraint to capture fine-grained categories knowledge and enhance the discrimination of target samples 

* TPR-19 [Exploring uncertainty in pseudo-label guided unsupervised domain adaptation](https://www.sciencedirect.com/science/article/pii/S0031320319302997)
     - To solve that the posterior probabilities (uncertainties) from  target data are ignored
     - Progressively increase the number of target training samples  
     - A triplet-wise instance-to-center margin is further maximized to push apart target instances and source class centers of different classes and bring closer them of the same class
  
  * MM-19 [ Joint Adversarial Domain Adaptation](https://dl.acm.org/doi/10.1145/3343031.3351070)
     - To solve class-wise mismatch across domains
     - By minimizing the disagreement between two distinct task-specific classifiers’ predictions to synthesize target features near the support of source class-wisely.
     
 
#### 1.1.3.5 Multi-representation DA
    - Considering that single structure or representations may only contain partial info like the saturation, brightness, and hue information. 
    - Align the distributions of multiple representations extracted by a hybrid structure named Inception Adaptation Module (IAM). 
    
 

## 1.2 Partial Domain Adaptation
* CVPR-18 [Importance Weighted Adversarial Nets for Partial Domain Adaptation](https://arxiv.org/abs/1803.09210)
     - Aim at reducing the distribution with unidentical label spaces
     - Propose a novel adversarial nets-based partial domain adaptation method to identify the source samples that are potentially from the outlier classes
     
## 1.3 Multiple Distribution
* CVPR-18 [Boosting Domain Adaptation by Discovering Latent Domains](https://arxiv.org/abs/1805.01386)
     - Assume that most datasets can be regarded as mixtures of multiple domains
     - Propose a novel CNN archtecture to discover latent domains automatically

* TKDE-18 [Structure-Preserved Unsupervised Domain Adaptation](https://par.nsf.gov/servlets/purl/10065320)
     - The whole structure of source domains is preserved to guide the target structure learning
    
     
 ## 1.4 Layer-wsie DA
 
  * NIPS-19 [Transferable Normalization: Towards Improving Transferability of Deep Neural Networks](https://papers.nips.cc/paper/8470-transferable-normalization-towards-improving-transferability-of-deep-neural-networks) [[official code]](https://github.com/thuml/TransNorm)
     - Propose TransNorm in place of existing normalization techniques
 
 * CVPR-19 [Unsupervised Domain Adaptation using Feature-Whitening and Consensus Loss](https://arxiv.org/pdf/1903.03215) [[official code]](https://github.com/roysubhankar/dwt-domain-adaptation)
     - Propose the feature whitening for domain alignment and the Min-Entropy Consensus loss for unlabeled target domain adaptation
     - Introduce MSE-LOSS which merges both the entropy and the consistency loss, and assume that the predictions for the same image should be similar 
     
 * CVPR-19 [Domain-Specific Batch Normalization for Unsupervised Domain Adaptation]( https://arxiv.org/pdf/1906.03950) 
     - Aim to adapt both domains by specializing batch normalization layers in CNN
     - Propose separate batch normalization layers for both domains
     
* ICCV-19 [An Adaptive Feature Norm Approach for Unsupervised Domain Adaptation](http://openaccess.thecvf.com/content_ICCV_2019/papers/Xu_Larger_Norm_More_Transferable_An_Adaptive_Feature_Norm_Approach_for_ICCV_2019_paper.pdf) [[official code]](https://github.com/jihanyang/AFN)
     - Assume that the erratic discrimination of the target domain mainly stems from its much smaller feature
norms
     - Solve the drastic model degradation on the target task
     - Propose a novel parameter-free Adaptive Feature Norm approach

 ## 1.5 Open-set DA
 * arXiv-19 [Known-class Aware Self-ensemble for Open Set Domain Adaptation](https://arxiv.org/abs/1905.01068v1) 
     - To identify the known and unknown classes from target data by encouraging a low cross-entropy for known classes and a high entropy 
 from unknown classes.

 ## 1.6 Semi-supervised DA
  * ICCV-19 [Semi-supervised Domain Adaptation via Minimax Entropy](https://arxiv.org/abs/1905.01068v1) 
     - Propose a classification layer that computes the features’ similarity to estimated prototypes (representatives of each class).
     - Adaptation is done by maximizing the conditional entropy of unlabeled target data with respect to the classifier and minizing it with respect to the feature encoder.
 
 ## 1.7 Compressing DA
  * JMLC-19 [Transfer channel pruning for compressing deep domain adaptation models](https://link.springer.com/chapter/10.1007/978-3-030-26142-9_23) 
     - First approach to accelerate the UDA by prunning less important channels.
     - Introduce Taylor expansion to judge the importance of channels


## 1.* Applications
### 1.*.1 Semantic Segmentation
* CVPR-19 [Pixel-level Domain Transfer with Cross-Domain Consistency](https://zpascal.net/cvpr2019/Chen_CrDoCo_Pixel-Level_Domain_Transfer_With_Cross-Domain_Consistency_CVPR_2019_paper.pdf)
     - Aim at capturing pixel-level domain shifts
     - Assume that translated images cross domains differ in styles
     - By exploiting a cross-domain consistency loss motivated by image-to-image translation methods


# 2.Tutorial & Survey
[Transfer Adaptation Learning: A Decade Survey](https://arxiv.org/pdf/1903.04687.pdf), Lei Z, et al. (arXiv 19)

* TCYB-18 [Robust Graph-Based Semisupervised Learning for Noisy Labeled Data via Maximum Correntropy Criterion](https://ieeexplore.ieee.org/document/8303753)
     - Propose to employ maximum correntropy criterion to suppress labeling noise, which can effectively handle outliers and non-
Gaussian noise.



