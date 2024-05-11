# Naive-Bayes-Classifier
To detect the presence of heart disease from the medical records of patients using Naive Bayes Classifier

Implementation of a Naïve Bayes classifier from first principles and evaluation on the dataset available from UCI. 
Comparison is made with Gaussian Naive Bayes Classifier of sklearn library. 
I have used 5 random partitions of training and testing dataset to evaluate the implementation and results are given for each partition.
The following outputs are shown:
1. Confusion matrix
2. F1 score
3. ROC curve

Dataset for evaluating Naïve Bayes classifier: Heart Disease Data Set, available from: 
https://archive.ics.uci.edu/ml/datasets/Heart+Disease

OUTPUT   

Naïve Bayes Classifier from Scratch
Test Train Split no.  1 

 Accuracy:  64.47368421052632 %

Confusion Matrix
[[39  4  1  1  1]
 [ 4  4  1  2  1]
 [ 0  2  0  0  3]
 [ 0  0  2  3  2]
 [ 0  1  0  2  3]]

F1 Score
0.6518226287506748

 
Test Train Split no.  2 

 Accuracy:  61.8421052631579 %

Confusion Matrix
[[31  6  0  1  1]
 [ 1  7  1  3  2]
 [ 0  0  3  1  1]
 [ 0  1  2  4  4]
 [ 0  1  1  2  3]]

F1 Score
0.6532670006731253

 
Test Train Split no.  3 

 Accuracy:  63.1578947368421 %

Confusion Matrix
[[39  4  0  0  0]
 [ 4  3  1  4  2]
 [ 0  1  2  2  1]
 [ 0  1  1  2  2]
 [ 0  1  1  2  3]]

F1 Score
0.6444976076555023


 
Test Train Split no.  4 

 Accuracy:  53.94736842105263 %

Confusion Matrix
[[27  9  2  0  2]
 [ 2  5  0  3  1]
 [ 1  1  1  1  4]
 [ 0  3  2  4  0]
 [ 0  2  0  2  4]]

F1 Score
0.563081498967478

 
Test Train Split no.  5 

 Accuracy:  72.36842105263158 %

Confusion Matrix
[[40  4  1  0  0]
 [ 4  5  0  3  2]
 [ 0  2  5  0  1]
 [ 0  1  0  4  1]
 [ 0  0  0  1  2]]

F1 Score
0.7409242639875402
 

Naïve Bayes Classifier using function
Test Train Split no.  1 

Gaussian Naive Bayes model accuracy(in %): 43.42105263157895

Confusion Matrix
[[27  9  1  2  1]
 [ 6  4  4  3  1]
 [ 1  5  1  3  0]
 [ 0  5  1  1  0]
 [ 0  1  0  0  0]]

F1 Score
0.45617409481573573

 
Test Train Split no.  2 

Gaussian Naive Bayes model accuracy(in %): 52.63157894736842

Confusion Matrix
[[32  3  2  0  1]
 [ 5  4  0  2  3]
 [ 0  4  1  3  3]
 [ 0  2  5  2  2]
 [ 0  1  0  0  1]]

F1 Score
0.53108341028008

 

Test Train Split no.  3 

Gaussian Naive Bayes model accuracy(in %): 44.73684210526316

Confusion Matrix
[[29  3  0  0  2]
 [ 7  5  2  1  5]
 [ 0  3  0  2  2]
 [ 1  1  4  0  4]
 [ 0  1  2  2  0]]

F1 Score
0.4452007098411842

 
Test Train Split no.  4 

Gaussian Naive Bayes model accuracy(in %): 57.89473684210527

Confusion Matrix
[[36  5  2  0  0]
 [ 7  3  2  1  0]
 [ 0  3  3  1  0]
 [ 0  3  3  2  3]
 [ 0  0  0  2  0]]

F1 Score
0.5782593739250085

 
Test Train Split no.  5 

Gaussian Naive Bayes model accuracy(in %): 47.368421052631575

Confusion Matrix
[[31  3  1  0 10]
 [ 3  3  1  0  6]
 [ 0  1  1  0  6]
 [ 1  0  0  0  7]
 [ 0  0  0  1  1]]

F1 Score
0.5309808612440191


 

