# Industrial_Downtime_Prediction

I am working on model to predict the downtime of an Industrial asset will have a failure in next 48hrs based on its everyday sensor data.


The Dataset has 20 predictor variables.I did the below steps as part of data cleaning,transformation and integration.
- Filtering on Asset ID- After loading the Indicator reading dataset we need to first filter it on HAGC Asset Number which is 540-630.
- Pivoting the Data- All the Indicator names are stored in a single columns ,we need to pivot the data such that each indicator name is an individual column and its readings for each date as rows
3. Type Conversions-  The Numeric readings are converted into integer format.
4. Filling the Null Values- While filling the null values I have used interpolation method for numeric values which have some kind of auto-correlation.And for other numeric values I have used the mean value.While filling nulls for categorical values I used a backfill.
5. Calculating Moving Average and Sum-Calculated moving average and sum to fill in nulls values for fluid loss readings.
6. Filtering the Year- Filtered the indicator data to year 2012 as the downtimes are only available for this period.So we have a dataset of 365 days.Am filtering the year after filling the nulls because the moving averages gets filled from previous years data.
7. Merging with Downtime dates- We fetch the downtime dates 2 days before to the downtime and mark as failure.
8. Creating Dummies for filter variables-In Python we convert the categorical variables to binary indicators using dummies function.
9. Dropping the records of actual Downtime- Dropping the actual downtime dates and its previous dates indicator readings as it can disturb the model.



As part of feature selection,I choose to run the random forest classification algorithm and get the feature importance.I also used spearman-rank correlation method.



I performed the below 4 machine learning algorithms
- Logistic Regression- It uses regularized regression using the liblinear solver sklearn which is preferred for smaller dataset.I used l2 penalty using scikit learn
- Gaussian Naive Bayes-  GaussianNB implements the Gaussian Naive Bayes algorithm for classification. The likelihood of the features is assumed to be Gaussian
3. Random Forest-A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and use averaging to improve the predictive accuracy and control over-fitting.
4. Gradient Boosting Classifier- Gradient Boosting builds an additive model in a forward stage-wise fashion; it allows for the optimization of arbitrary differentiable loss functions. 

Apart from the above 4 algorithms,I tried using SVM and KNN classifier algorithms,But their performance scores(recall) are too low.

I have used Grids search in python to search for best parameters to run the model.I have performed grid search along with cross validation in order to avoid over fitting of the data.
In order to run the files run

./Data_transformation.sh

and then run model.sh with Argument as 
- LR for Logistic Regression  (./model LR)
- NB for Naive Bayes          (./model NB)  
- RF for Random Forest        (./model RF)
- GB for gradient boosting    (./model GB)

Please read the Predicting the Downtime document to find the results of this model.
