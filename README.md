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
