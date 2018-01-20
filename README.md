## Project Image classifications
Include: BoW combine histogram, DNN, Naive Bayes
### BoW + histogram
Extract 15 highest score keypoint descroptions per image <br />
Clutering using Kmean with n_cluster = 500<br />
Compute feature_decriptors with histogram by evaluate the histogram vector of each image. Histogram vector dimentions = n_clusters<br />
Accurency ~ 88%<br />
Dataset: <br />
- Caltech-4: https://www.dropbox.com/sh/f2v3omkeozzoooe/AAAsmpZFEg6Za18bQyE_rDuJa?dl=0<br />
- Extracted BoW_histogram_feature_descriptions: https://www.dropbox.com/sh/67xvwp18te69wlf/AACF8kzu2aG2kX9C3RcEiyTfa?dl=0<br />
(or you could extract this kind of feature by running BoWCodeWordBuilder.py and histogram_feature_extraction.py)<br />
