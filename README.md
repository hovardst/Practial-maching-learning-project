# Practical Machin Learning course - Assignment Week 4

## Assignment
The goal of your project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases.


## Report

To complete the assignment I decided to try two different classification models, Linear Discriminant Analysis (LDA) and Random Forest.


### Code
The final R-code for the assignment is in the file "Week 4 assignement.R" in this GIT HUB repository. 

The main sections of the code is

* Loading the data
  **	Data for test and training was read from  http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har
* Cleaning data
  * The data was cleaned to facilitate training the different models with the following methods
    - Removing variables with near zero variance
    - Removing variables that don't add any value to prediction ( variables removed,  X,  user_name and cvtd_timestamp) 
    - Some of the variables has a large proportion of NA values. Remove all variables that have more than 80% NA values. 
- Split training data 80%/20% to get training  and cross validation set from cleaned data set
- Train and evaluate model 
- Print prediction of best model based on test data

### Analysis
First I tried to train a model based on the Linear Discriminant Analysis (LDA) method. 

The confusion matrix for the model is

![Confusion Matrix LDA modell](https://github.com/hovardst/Practial-maching-learning-project/blob/master/Confusion%20matrix%20-%20LDA%20modell.png)

As the accuracy is only 0.733, the out of sample error is estimated to 0,267. This is hence not a very accurate model. 


Running the same dataset with the Random Forrest (RF) method gave the following confusion matrix

![Confusion Matrix RF modell](https://github.com/hovardst/Practial-maching-learning-project/blob/master/Confusion%20matrix%20-%20RF%20modell.png)

With this method, the accuracy is 0.998, indicating an out of sample error of 0,002. This seems like a much more accurate model, and I used the RF model to predict the 20 different test cases. The result is in the file “Final prediction.csv” in this repository. 
