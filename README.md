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
- Full folderstruct:<br />
![capture](https://user-images.githubusercontent.com/16191939/35181224-513c3ac4-fdf0-11e7-8b1c-7e1e0b7bb346.PNG) <br />
<i>Folder backup and dataset just for me, it's not necessary</i><br />
#### Dataset:
- Caltech-4:  <a href="https://www.dropbox.com/sh/f2v3omkeozzoooe/AAAsmpZFEg6Za18bQyE_rDuJa?dl=0">Donwload Here</a><br />
- Extracted BoW_histogram_feature_descriptions: <a href="https://www.dropbox.com/sh/67xvwp18te69wlf/AACF8kzu2aG2kX9C3RcEiyTfa?dl=0">Here</a><br />
(or you could extract this kind of feature by running BoWCodeWordBuilder.py and histogram_feature_extraction.py)<br />
#### References: <a href="https://www.facebook.com/tranquoclong.t1">Tran Quoc Long</a><br />
