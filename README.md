# Machine Learning from Scratch
Common Machine Learning algorithms implemented from Scratch

### Index:
1. [Linear Regression using Normal Equation](#linear_reg_eqn)
2. [Linear Regression with Gradient Descent](#linear_reg_gd)
3. [Logistic Regression with Gradient Descent](#log_reg_gd)
4. [Decision Trees](#dec_tree)
5. [Regression Tree](#reg_tree)
6. [Logistic Regression using Newton's method](#log_reg_newm)
7. [Linear Regression with Ridge Regularization](#lin_reg_rreg)
8. [Perceptron](#percp)
9. [Autoencoder NN from scratch and using Tensorflow](#auto_enc)
10. [Classifier Neural Network (square loss) from scratch and using Tensorflow](#class_sq)
11. [Classifier Neural Network (cross-entropy) from scratch and using Tensorflow](#class_cross_enp)
12. [Gaussian Discriminant Analysis](#gda)
13. [Naive Bayesian Classifier](#naive_bayes)
14. [Expectation Maximization](#exp_max)
15. [AdaBoost](#ada_boost)
16. [AdaBoost with Active Learning](#active_learn)
17. [AdaBoost with missing data (on UCI datasets)](#adaboost_uci)
18. [Error Correcting Output Codes](#ecoc)
19. [Gradient Boosted Trees](#gd_trees)
20. [Feature Selection](#feature-select)
21. [PCA for Feature Reduction](#pca)
22. [Logistic Regression with regularization](#ridge_lasso)
23. [HAAR Image feature extraction](#haar)
24. [Support Vector Machine](#svm)
25. [Dual Perceptron](#dual_percp)

## <a id="linear_reg_eqn"></a>Linear Regression (Normal Equation)
Linear Regression with Mean Squared Error cost function. The weight training is done with Normal Equation (closed-form solution).

[Linear Regression for predicting Housing Price](https://github.com/prasad-madhale/machine-learning/blob/master/Linear%20Regression/Housing(Linear_Reg).ipynb)

[Linear Regression for Email Spam detection](https://github.com/prasad-madhale/machine-learning/blob/master/Linear%20Regression/SpamBase(Linear_Reg).ipynb)

## <a id="linear_reg_gd"></a>Linear Regression (using Batch Gradient Descent)
Linear Regression for House Price and Spam Email prediction using Batch Gradiet Descent.

[Linear Regression with Gradient Descent-Spambase](https://github.com/prasad-madhale/machine-learning/blob/master/Gradient%20Descent/Gradient_Descent_Spambase.py)

*Cost function*

<img src="https://github.com/prasad-madhale/machine-learning/blob/master/Gradient%20Descent/plot/spambase_cost_func.png" width="600">

*ROC curve*

<img src="https://github.com/prasad-madhale/machine-learning/blob/master/Gradient%20Descent/plot/spambase_ROC.png" width="600">

***

[Linear Regression with Gradient Descent-Housing](https://github.com/prasad-madhale/machine-learning/blob/master/Gradient%20Descent/Gradient_Descent_Housing.py)

*Cost function*

<img src="https://github.com/prasad-madhale/machine-learning/blob/master/Gradient%20Descent/plot/housing_cost_func.png" width="600">

## <a id="log_reg_gd"></a>Logistic Regression with Batch Gradient Descent
[Logistic Regression - Spambase dataset](https://github.com/prasad-madhale/machine-learning/blob/master/Logistic%20Regression/Logistic_reg_Spambase.py)

*Log Likelihood*

<img src="https://github.com/prasad-madhale/machine-learning/blob/master/Logistic%20Regression/plot/log_likehood.png" width="600">

*ROC curve*

<img src="https://github.com/prasad-madhale/machine-learning/blob/master/Logistic%20Regression/plot/ROC_curve.png" width="600">

## <a id="dec_tree"></a>Decision Tree

Decision Tree to classify data points in the Spambase dataset.

[Decision Tree](https://github.com/prasad-madhale/machine-learning/blob/master/Decision%20Tree/Decision_Tree.ipynb)

## <a id="reg_tree"></a>Regression Tree

Regression Tree to predict continuous valued data for Housing price dataset.

[Regression Tree](https://github.com/prasad-madhale/machine-learning/blob/master/Regression%20Tree/Regression_Tree.ipynb)

## <a id="log_reg_newm"></a>Logistic Regression using Newton's Method

Train logistic regression using Newton's method (solution in closed form)

[Logistic Regression with Newton's method](https://github.com/prasad-madhale/machine-learning/blob/master/Newton's%20Method/Log_reg_Newtons_method_Spambase.py)

*Log likelihood*

<img src="https://github.com/prasad-madhale/machine-learning/blob/master/Newton's%20Method/plot/log_likelihood.png" width="600">

## <a id="lin_reg_rreg"></a>Linear Regression with Ridge Regularization

Train Linear Regression with Ridge regularization to control the weights

[Linear Regression Ridge regularization - Housing](https://github.com/prasad-madhale/machine-learning/blob/master/Regularization/Housing_Ridge_Regularization.ipynb)

[Linear Regression Ridge regularization - Spambase](https://github.com/prasad-madhale/machine-learning/blob/master/Regularization/SpamBase_Ridge_Regularization.ipynb)

## <a id="percp"></a>Perceptron

Single layer pereptron to classify 01 labelled dataset

[Perceptron](https://github.com/prasad-madhale/machine-learning/blob/master/Perceptron/perceptron.py)

*Mistakes per Iteration*

<img src="https://github.com/prasad-madhale/machine-learning/blob/master/Perceptron/plot/mistakes.png" width="600">

## <a id="auto_enc"></a>Autoencoder Neural Network (scratch & tf)

Implemented Multilayer perceptron Neural Network for Autoencoder from scratch and using Tensorflow

[AutoEncoder from Scratch](https://github.com/prasad-madhale/machine-learning/blob/master/AutoEncoder/scratch_autoencoder.py)

*Loss per Epoch*

<img src="https://github.com/prasad-madhale/machine-learning/blob/master/AutoEncoder/plots/loss_autoencoder_scratch.png" width="600">

[AutoEncoder using Tensorflow](https://github.com/prasad-madhale/machine-learning/blob/master/AutoEncoder/autoencoder_tf.py)

*Loss per Epoch*

<img src="https://github.com/prasad-madhale/machine-learning/blob/master/AutoEncoder/plots/autoencoder_loss_tf.png" width="600">

## <a id="class_sq"></a>Square Loss Classifier Neural Network (scratch & tf)

Implemented a Multilayer perceptron Neural Network for Classification from scratch and using Tensorflow.
Uses sigmoid activation along with a square loss.

[Classifier with Square Loss from scratch](https://github.com/prasad-madhale/machine-learning/blob/master/NeuralNetwork/wine_classify_scratch.py)

*Loss per Epoch*

<img src="https://github.com/prasad-madhale/machine-learning/blob/master/NeuralNetwork/plots/wine_classify_loss.png" width="600">

[Classifier with Square Loss using tensorflow](https://github.com/prasad-madhale/machine-learning/blob/master/NeuralNetwork/wine_classify(tf).py)


*Loss per Epoch*

<img src="https://github.com/prasad-madhale/machine-learning/blob/master/NeuralNetwork/plots/wine_classify_loss_tf.png" width="600">

## <a id="class_cross_enp"></a>Cross-Entropy Classifier Neural Network

Implemented a Multilayer perceptron Neural Network for Classification from scratch and using Tensorflow.
Uses sigmoid activation with softmax at the output layer along with a cross entropy loss.

[Classifier with Cross Entropy Loss from scratch](https://github.com/prasad-madhale/machine-learning/blob/master/NeuralNetwork/wine_likelihood_scratch.py)

*Loss per Epoch*

<img src="https://github.com/prasad-madhale/machine-learning/blob/master/NeuralNetwork/plots/wine_classify_likelihood.png" width="600">


[Classifier with Cross Entropy Loss using Tensorflow](https://github.com/prasad-madhale/machine-learning/blob/master/NeuralNetwork/wine_likelihood(tf).py)

*Loss per Epoch*

<img src="https://github.com/prasad-madhale/machine-learning/blob/master/NeuralNetwork/plots/wine_log_likelihood_tf.png" width="600">

## <a id="gda"></a>Gaussian Discriminant Analysis

Implemented GDA which learns a distribution to form discriminant function for prediction.

[GDA](https://github.com/prasad-madhale/machine-learning/blob/master/Gaussian%20Discriminant%20Analysis/gda.py)

## <a id="naive_bayes"></a>Naive Bayesian Classifier

Implemented a Naive Bayesian Classifier which uses Baye's rule to learn a gaussian using given data probabilities.

[Naive Bayes Classifier (Gaussian)](https://github.com/prasad-madhale/machine-learning/blob/master/Naive%20Bayesian/gaussian_naive_bayes.py)

[Naive Bayes Classifier with Bernoulli](https://github.com/prasad-madhale/machine-learning/blob/master/Naive%20Bayesian/bernoulli_naive_bayes.py)

[Naive Bayes Classifier for broken down into 9 bins](https://github.com/prasad-madhale/machine-learning/blob/master/Naive%20Bayesian/9bin_naive_bayes.py)

[Naive Bayes Classifier for broken down into 4 bins](https://github.com/prasad-madhale/machine-learning/blob/master/Naive%20Bayesian/4bin_naive_bayes.py)

[Naive Bayes Classifier on Polluted Dataset](https://github.com/prasad-madhale/machine-learning/blob/master/Naive_Polluted/gaussian_naive_bayes.py)

[Naive Bayes on missing data](https://github.com/prasad-madhale/machine-learning/blob/master/Naive_Bayes_missing_data/missing_bernoulli_naive_bayes.py)

We implement Naive Bayes on missing data by ignoring it in Bernoulli distribution probability calculations.

## <a id="exp_max"></a>Expectation Maximization

Given data which is a mixture of Gaussian. EM algorithm accurately predicts to which Gaussian the data point belongs

[Expectation Maximization (mixture of Gaussian)](https://github.com/prasad-madhale/machine-learning/blob/master/EM/gaussian_em_scratch.py)

[Expectation Maximization for Coin Flipping example](https://github.com/prasad-madhale/machine-learning/blob/master/EM/coin_flip.py)

*Given two biased coins and datapoints derived by picking one coin at random and flipping it d times.
EM algorithm helps to predict which coin is used to create that datapoint*

## <a id="ada_boost"></a> AdaBoost

Boosting is a methodology where by combining multiple weak learners we get a strong model for prediction.
I have used simple **1-split Decision Tree** as weak learner for this AdaBoosting implementation.

[AdaBoost with Optimal Thresholding](https://github.com/prasad-madhale/machine-learning/blob/master/AdaBoost/adaboost.py)

*Optimal thresholding signifies going through all the decision stumps => (feature,threshold) combinations to find the one
that gives maximum improvement in predictions*

*Error at Each Round*

<img src="https://github.com/prasad-madhale/machine-learning/blob/master/AdaBoost/plots/round_error.png" width="600">

*Train/Test Error*

<img src="https://github.com/prasad-madhale/machine-learning/blob/master/AdaBoost/plots/train_test_error.png" width="600">

*AUC curve*

<img src="https://github.com/prasad-madhale/machine-learning/blob/master/AdaBoost/plots/auc.png" width="600">

*ROC curve*

<img src="https://github.com/prasad-madhale/machine-learning/blob/master/AdaBoost/plots/roc.png" width="600">

[AdaBoost with Random Thresholding](https://github.com/prasad-madhale/machine-learning/blob/master/AdaBoost/adaboost_random.py)

*Random thresholding signifies picking a decision stump => (feature,threshold) combination at random*

*Error at Each Round*

<img src="https://github.com/prasad-madhale/machine-learning/blob/master/AdaBoost/plots/random_round_error.png" width="600">

*Train/Test Error*

<img src="https://github.com/prasad-madhale/machine-learning/blob/master/AdaBoost/plots/random_train_test_error.png" width="600">

*AUC curve*

<img src="https://github.com/prasad-madhale/machine-learning/blob/master/AdaBoost/plots/random_auc.png" width="600">

*ROC curve*

<img src="https://github.com/prasad-madhale/machine-learning/blob/master/AdaBoost/plots/random_roc.png" width="600">

[AdaBoost on Polluted Data](https://github.com/prasad-madhale/machine-learning/blob/master/Adaboost_with_bad_features/adaboost_polluted.py)

Implemented AdaBoost with Optimal decision stump on polluted dataset. Output stored in ./logs/out_polluted.txt 

## <a id="active_learn"></a> [AdaBoost with Active Learning](https://github.com/prasad-madhale/machine-learning/blob/master/Active-Learning/adaboost_active_learning.py)

Active learning is a technique in which we start with some percent of random data in the training set and then keep adding data points with least error to the training set. I have used Adaboost with Optimal Decision stumps to implement the Active Learning starting with 5, 10, 15, 20, 30, 50 percent of random data. 

## <a id="adaboost_uci"></a> [AdaBoost with missing data on UCI datasets](https://github.com/prasad-madhale/machine-learning/tree/master/AdaBoost_UCI)

Implemented Adaboost on popular UCI datasets. Also, handled missing data in both datasets. Tested the performance by using some fixed percent of data selected at random. 

## <a id="ecoc"></a> [Error Correcting Output Codes](https://github.com/prasad-madhale/machine-learning/blob/master/ECOC/Ecoc.py)

Error Correcting Output Codes uses a coding matrix to use provide a way to use a binary classifier on a multi-label dataset. ECOC uses this coding matrix such that each column in the matrix represents a subset of labels and each of these has it's own model (for our case we use Adaboost as the individual learner).

## <a id="gd_trees"></a>[Gradient Boosted Trees](https://github.com/prasad-madhale/machine-learning/blob/master/Gradient-Boosting/gradient_boost.py)

## <a id="bagging"></a>[Bagging](https://github.com/prasad-madhale/machine-learning/blob/master/Bagging/bootstrap_aggr.py)

Bagging involves creating small bags of x% data picked randomly with replacement. A model is trained on each bag and predictions are made based on either the average/mode of predictions over all models depending on the type of labels.

## <a id="feature-select"></a>[Feature Selection using Margin analysis](https://github.com/prasad-madhale/machine-learning/blob/master/Adaboost_with_bad_features/adaboost_bad_features.py)

Select top important features based on margins analysis. [margins](http://www.ccs.neu.edu/home/vip/teach/MLcourse/5_features_dimensions/lecture_notes/boosting_margin_jay/jay_margin_feature.pdf)

## <a id="pca"></a>[PCA for feature reduction](https://github.com/prasad-madhale/machine-learning/blob/master/Naive_bayes_PCA/gaussian_naive_bayes_PCA.py)

PCA allows us to reduce the number of features in the dataset by creating features which are a linear combination of each other. We used sklearn's PCA implementation to reduce the number of features in our dataset to 100 and then ran Naive Bayes on these features to obtain good results.

## <a id="ridge_lasso"></a>Logistic Regression with regularization

[Logistic Regression with Ridge regularization](https://github.com/prasad-madhale/machine-learning/blob/master/Logistic_with_regularization/Logistic_reg_Ridge_reg.py)

[Logistic Regression with LASSO regularization](https://github.com/prasad-madhale/machine-learning/blob/master/Logistic_with_regularization/Logistic_reg_lasso_sk.py)

## <a id="haar"></a>[HAAR Image feature extraction](https://github.com/prasad-madhale/machine-learning/blob/master/HAAR/HAAR_feature.py)

We extracted features from MNIST dataset using HAAR methodology. 
1. Create 100 rectangles randomly distributed inside image pixel sizes.
2. Extract 2 features each by imposing these 100 rectangles on each image.
3. We get 200 features per image.
4. Feed these features and labels into a multi-class classifier to obtain predictions(in our case we use AdaBoost with ECOC)

## <a id="svm"></a>Support Vector Machine

[SVM using sklearn](https://github.com/prasad-madhale/machine-learning/blob/master/SVM/svm_sklearn.py)
[SVM with HAAR features](https://github.com/prasad-madhale/machine-learning/blob/master/SVM/SVM_mnist_HAAR.py)
Use the extracted HAAR features from MNIST dataset in SVM
[SVM with SMO from scratch](https://github.com/prasad-madhale/machine-learning/blob/master/SMO/svm_smo.py)
[SVM on MNIST with HAAR features](https://github.com/prasad-madhale/machine-learning/blob/master/SMO/smo_mnist.py)

## <a id="knn"></a>K-Nearest Neighbors

#### 1. [KNN with Fixed neighbors](https://github.com/prasad-madhale/machine-learning/blob/master/KNN/knn_radius_digits.py)
Implemented KNN with different number of nearest neighbors. Also, used different kernels like Gaussian, Cosine and Polynomial.

#### 2. [KNN with probabilities](https://github.com/prasad-madhale/machine-learning/blob/master/KNN/knn_probs_digits.py)
Implemented KNN with probability density estimator

#### 3. [KNN with a fixed range of radius](https://github.com/prasad-madhale/machine-learning/blob/master/KNN/knn_radius_digits.py)
Implemented KNN with fixed range of radius measured with Euclidean distance

#### 4. [KNN with Relief algorithm](https://github.com/prasad-madhale/machine-learning/blob/master/KNN/knn_with_relief.py)
Implemented KNN with feature selection by independently assessing the quality Weights of each feature by adjusting the weights with each instance

## <a id="dual_percp"></a>[Dual Perceptron](https://github.com/prasad-madhale/machine-learning/blob/master/Dual-Perceptron/dual_perceptron.py)

Implemented Dual Perceptron with linear and gaussian kernel. 


