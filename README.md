# Credit_Risk_Analysis

## Project Overview

Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. Therefore, we need to employ different techniques to train and evaluate models with unbalanced classes. In this project, we use Python to build and evaluate several supervised machine learning models to predict credit risk.

We adopted the following procedure:

- Oversample the data using the **RandomOverSampler** and **SMOTE** algorithms.
- Undersample the data using the **ClusterCentroids** algorithm.
- Use a combinatorial approach of over- and under sampling using the **SMOTEENN** algorithm.
- Compare two machine learning models that reduce bias, **BalancedRandomForestClassifier** and **EasyEnsembleClassifier**.

We will evaluate the performance of these models and make a recommendation on whether they should be used to predict credit risk.

## Resources
Data Source: LoanStats_2019Q1.csv
Software: Python, Anaconda Navigator, Jupyter Notebook 

## Results

We are using Loan Status file for this analysis and apply following ML models to predit high / low credit risk customers.


### Oversampling - RandomOverSampler

![image](https://user-images.githubusercontent.com/83181834/131159729-fec86c9b-5247-4304-a839-91705cc9db0f.png)

![image](https://user-images.githubusercontent.com/83181834/131159594-a20f891d-0a35-4170-b97d-0de964b003dc.png)

![image](https://user-images.githubusercontent.com/83181834/131159652-f5e8de06-16a2-4ce2-87b1-b6186e2c9d5a.png)

- Balanced Accuracy score is 65%
- Precision of high risk is 1 % because of less samples with sensitivity of 67%
- Precision of low risk is 100 % because of high samples with sensitivity of 64%

Low risk precision is good along with sensitivity but high risk precision is very low of 1%

### Oversampling - SMOTE

![image](https://user-images.githubusercontent.com/83181834/131159767-6e582e2f-67eb-4eb9-a580-d43462d71c46.png)

![image](https://user-images.githubusercontent.com/83181834/131159792-9f6b8185-9269-4725-9506-210398dd8739.png)

![image](https://user-images.githubusercontent.com/83181834/131159839-3ece4731-d36e-4a70-8c34-ecdf3efde6ae.png)

- Balanced Accuracy score is 63%
- Precision of high risk is 1 % because of less samples with sensitivity of 61%
- Precision of low risk is 100 % because of high samples with sensitivity of 64%

Results are similar as above. Low risk precision is good along with sensitivity but high risk precision is very low of 1%

### Undersampling - ClusterCentroids

![image](https://user-images.githubusercontent.com/83181834/131159885-4b4813bf-7a0c-490f-9ca7-5f0f9172706a.png)

![image](https://user-images.githubusercontent.com/83181834/131159902-7de1d534-189b-4f1c-8f3a-51a38349d91b.png)

![image](https://user-images.githubusercontent.com/83181834/131159974-2aa6bce2-0920-46b9-a4b1-a98de8e13c23.png)

- Balanced Accuracy score is 52 %
- Precision of high risk is 1 % because of less samples with sensitivity of 57 %
- Precision of low risk is 100 % because of high samples with sensitivity of 46 %

Low risk precision is good however sensitivity is only 46% because of high number of false positive. High risk precision is still 1%

### Combination - SMOTEENN

![image](https://user-images.githubusercontent.com/83181834/131160029-c94a0a6b-1e97-44f0-b032-41f270b7e29e.png)

![image](https://user-images.githubusercontent.com/83181834/131160054-ea30fd83-d8e6-4808-9d7e-9c42ee1e7684.png)

![image](https://user-images.githubusercontent.com/83181834/131160099-3b0ed307-0586-4338-aac6-9cf4894e5af4.png)

- Balanced Accuracy score is 62%
- Precision of high risk is 1 % because of less samples with sensitivity of 71 %
- Precision of low risk is 100 % because of high samples with sensitivity of 54 %

Pretty much similar results as above, low risk precision is good however sensitivity is only 54 % because of high number of false positive. High risk precision is still 1%

### Ensemble - BalancedRandomForestClassifier

![image](https://user-images.githubusercontent.com/83181834/131160166-d04cd3e5-606b-4790-96e7-699830f48c17.png)

![image](https://user-images.githubusercontent.com/83181834/131160185-213c584c-17eb-4243-af54-edbc98ff50c2.png)

![image](https://user-images.githubusercontent.com/83181834/131160321-80c8b1df-2b29-4a1e-b96f-0fdd974dc5ac.png)

- Balanced Accuracy score is 79 %
- Precision of high risk is 4 % because of less samples with sensitivity of 67 %
- Precision of low risk is 100 % because of high samples with sensitivity of 91 %

Almost all stats have improved with ensemble ML but high risk precision is still 4 % which is way too low.

### Ensemble - EasyEnsembleClassifier

![image](https://user-images.githubusercontent.com/83181834/131160216-b694136f-f70d-47e1-b0d3-56fcf233db62.png)

![image](https://user-images.githubusercontent.com/83181834/131160241-ad197c72-7f66-475a-871d-be9a011980bd.png)

![image](https://user-images.githubusercontent.com/83181834/131160262-db699414-3e82-40f6-ae07-695da66eafbd.png)

- Balanced Accuracy score is 93 %
- Precision of high risk is 7 % because of less samples with sensitivity of 91 %
- Precision of low risk is 100 % because of high samples with sensitivity of 94 %

Accuracy score improved to 93% but high risk precision is still 7 %. Everything else is looking good in this model.
  
## Summary

All the 6 models used to perform the Credit Risk analysis shows weak precision for credit high risk. Ensemble model improved the overall accuracy score , especially EasyEnsembleClassifier has near perfect score in all the stats except Precision of High risk.

### Recommendation

On the other hand, all the models are tagging a lot of low risk as high risk which would impact Bank's revenue and business strategy in falsely tagging low risk as high risk. Because of this reason, I would not recommend the bank to use any of these models to predict credit risk. 

