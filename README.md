# Title-Alignment

In title_alignment_classifier.py file , I train a logistic regression classifier on Ted dataset, which contains a list
of 1161‬ projects and, for each project, a set of 5 candidate documents obtained by an automatic pre-process.
The projects are highlighted in green and have the value "Ted" in the Source column. The 5 rows below each green line
include the candidate documents for that project. The dataset includes the judgment of 4 annotators and there should be
3 annotations for every document. The annotations are combined in the Aggregate column. Use the values of this column as
the target labels to train and test you models.

1. The train accuracy of model is 0.9231800766283524
2. The test accuracy of model is  0.9367521367521368


# Application

In the Application.py file, I applied trained model on eis-title data file, contains 14,643 titles. I paired two
titles on it and got millions of title pairs. These title pairs does not have true labels. I need to use logistic
regression classifier to predict them. I created a csv file that includes title pairs, corresponding cosine
similarity and predicted labels.



