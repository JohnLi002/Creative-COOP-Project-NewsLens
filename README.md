# NewsLens

The overall purpose of NewsLens is to help highlight the bias present in articles. Currently this repository only contains files related to a machine learning algorithms with the purpose of predicting the news outlet based on the text given to it as well as various other files with the purpose of bringing our project to life. We are far from done however we hope that in the future we can complete it.

There are several csv files we used but are not present within this github. I you want to use the datasets we used, everything originates from the "All the News" in [Kaggle](https://www.kaggle.com/snapcrack/all-the-news). You can use our Categorize_Datasets.py and Create_Test_Dataset.py files withiin the Datasets folder to help create csv files that were used.

## Libraries were installed
If you are having trouble with using any of these libraries despite installing the necessary them, please try to use anaconda and create an environment that solely for the specific packages needed. That is what I had done to run some of these files. The following libraries mentioned are linked to their Github pages. Check their README.md for more information and links to their documentation as well as how to install their libraries. Some of these libraries have prerequisite but I only mentioned the ones below because they are the libraries that I have specified to be imported in some of the code. 

### Libraries used to create Neural Network
[Sci-Kit Learn](https://github.com/scikit-learn/scikit-learn)

[TensorFlow](https://github.com/tensorflow/tensorflow/)

[TensorFlow-Hub](https://github.com/tensorflow/hub)
-Was used for Google's Universal Sentence Encoder

[Keras](https://github.com/keras-team/keras)
-For this project I downloaded it, and you can learn to do so [here](https://pypi.org/project/keras/) as the documentation within does not mention how to do so. However there seems to also be the option of using Keras by simply importing it through TensorFlow. You will need to modify some parts of the code if you choose to use Keras through TensorFlow.

**Following four libraries that were used to create heatmaps so we could visually see of results**

[Seaborn](https://github.com/mwaskom/seaborn) *Seaborn also required the 3 libaries below*

[Numpy](https://github.com/numpy/numpy)

[Matlabplot](https://github.com/matplotlib/matplotlib)

[pandas](https://github.com/pandas-dev/pandas)


### Libraries used for Keyword Extraction
[scikit learn](https://github.com/scikit-learn/scikit-learn)
-For TF-IDF

[rake-nltk](https://github.com/csurfer/rake-nltk)

[gensim](https://github.com/RaRe-Technologies/gensim)
-For TextRank method

[YAKE](https://github.com/LIAAD/yake)

[KeyBERT](https://github.com/MaartenGr/KeyBERT)