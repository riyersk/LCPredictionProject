# LCPredictionProject
Predicting whether or a loan on lending club will charge off.

In this project, I use data on lending club loans to predict whether or not a given loan will charge off (lending club gives up on collecting payments on it.)

Guide:

LCLoadAndClean.ipynb : This python notebook file outlines how I loaded and cleaned the data.

LCgridSearchR.py : This python script was run on an AWS instance. It runs model gridsearches on subsets of the data and writes output results to text files. This script was run on subsets that had balanced classes, so rebalancing was deemed unnecessary.

LCgridSearchNR.py : This python script was also run on an AWS instance. It runs model gridsearches on subsets of the data and writes output results to text files. This script was run on subsets with imbalanced classes, so it rebalances before running the gridsearches.

LCModelResults.ipynb : This python notebook file explores the results of the modeling.

All the text files: These text files were written by the two python scripts already mentioned. The ones that begin with the prefix 'lr' represent logistic regression models, while those that begin with the prefix 'rf' represent random forest models. The next two characters represent the subset of the lending club data the models were built on. Each text file contains a confusion matrix of the model being tested on a test set, as well as a score, which represents the rate of return you would achieve if you invested according to the model.
