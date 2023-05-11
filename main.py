import os
from A.A import data_preprocessing
from A.A import SVM
from A.A import createEnsemble
from A.A import trainEnsemble
from A.A import evaluateEnsemble
from A.A import createTransformer
from A.A import trainTransformer
from A.A import evaluateTransformer


# ======================================================================================================================
# Data preprocessing
os.listdir("Datasets") 
data_train, data_val, data_test = data_preprocessing()
# ======================================================================================================================

# Baseline test for SVM
#acc_A_train, acc_A_test = SVM(data_train, data_test)

# Build and train model object.
print("Creating the model.......")
model_A, dataset_train, dataset_val, dataset_test = createTransformer(data_train, data_val, data_test)
# Train model based on the training set
print("Training the model.......")
acc_A_train = trainTransformer(model_A, dataset_train, dataset_val) 
# Test model based on the test set.
print("Testing the model.......")
acc_A_test = evaluateTransformer(model_A, dataset_test, data_test)


# ======================================================================================================================
# Print out results
print('TA:{},{}'.format(acc_A_train, acc_A_test))