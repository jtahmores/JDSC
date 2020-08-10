# JDSC
Joint distinct subspace learning and unsupervised transfer classification for visual domain adaptation

This package contains the data and code used in Joint distinct subspace learning and unsupervised transfer classification for visual domain adaptation (JDSC), which is published at Signal, Image and Video Processing (2020).
You can download the paper from: "https://doi.org/10.1007/s11760-020-01745-w". 

# Motivation
In many real-world knowledge transfer and transfer learning scenarios, the known common problem is distribution discrepancy (i.e., the difference in type, distribution and dimensionality of features) between source and target domains. In this paper, we introduce joint distinct subspace learning and unsupervised transfer classification for visual domain adaptation (JDSC) method, which is an iterative two-step framework. JDSC is based on hybrid of feature-based and classifier-based approaches that uses the feature-based techniques to tackle the challenge of domain shift and classifier-based techniques to learn a reliable model. In addition, for subspace alignment, weighted joint geometrical and statistical alignment is proposed to learn two coupled projections for mapping the source and target data into respective subspaces by accounting the importance of marginal and conditional distributions, differently. The proposed method has been evaluated on various real-world image datasets. JDSC gets 86.2% average classification accuracy on four standard domain adaptation benchmarks. The experiments demonstrate that our proposed method achieves a significant improvement compared to other state of the arts in average classification accuracy. Our source code is available at https://github.com/jtahmores/JDSC.

# RUN

The original code is implemented using Matlab R2018a. For running the code, run the "office.m", "piedemo.m", "demomnist_usps" and "coildemo.m " files.

# Datasets

*_SURF_L10.mat:    features and labels related to Office-Caltech-10

PIE*.mat:    features and labels related to PIE dataset

MNIST_vs_USPS.mat:    features and labels related to Digit dataset

COIL1_vs_COIL2.mat:    features and labels related to Coil dataset

# Reference

Noori Saray, S., Tahmoresnezhad, J. Joint distinct subspace learning and unsupervised transfer classification for visual domain adaptation. SIViP (2020). https://doi.org/10.1007/s11760-020-01745-w		

# Contact

Jafar Tahmoresnezhad (tahmores@gmail.com)

Shiva noori saray (shivanoorisaray@gmail.com)
