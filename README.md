## Project Image classifications
Include: BoW combine histogram, DNN, Naive Bayes
### BoW + histogram
#### Decriptions:
- Extract 15 highest score keypoint descroptions per image <br />
- Clutering using Kmean with n_cluster = 500<br />
- Compute feature_decriptors with histogram by evaluate the histogram vector of each image. Histogram vector dimentions = n_clusters<br />
- Accurency ~ 88%<br />
- <b>Reusult</b> is shown at <a href="https://github.com/tranquoclongt1/doan_Advanced_cv/blob/master/BoW/Report_Results_SVM.ipynb">Report_Results_SVM</a>, you are able to run it yourself with iPython and Jupyter Notebook 
#### Using:
- Dowload dataset and extract to folder <b>images</b>
- Be able to modify the number of clusters by edit in BoWCodeWOrdBuileber.py
- Be able to modify the number of dimentions of description vectors by edit in histogram_feature_extraction.py
#### Dataset:
- Caltech-4:  <a href="https://www.dropbox.com/sh/f2v3omkeozzoooe/AAAsmpZFEg6Za18bQyE_rDuJa?dl=0">Donwload Here</a><br />
- Extracted BoW_histogram_feature_descriptions: <a href="https://www.dropbox.com/sh/67xvwp18te69wlf/AACF8kzu2aG2kX9C3RcEiyTfa?dl=0">Here</a><br />
(or you could extract this kind of feature by running BoWCodeWordBuilder.py and histogram_feature_extraction.py)<br />
#### References: <a href="https://www.facebook.com/tranquoclong.t1">Tran Quoc Long</a><br />
