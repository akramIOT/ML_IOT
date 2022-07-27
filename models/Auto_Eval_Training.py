
## Evaluating Different Models by using  the  Auto-ML  framework  ""EVALML""  in this  module.

print("\nImporting to Auto-ML based Training ...##")

import evalml   ## AutoML  technique to be used here  This package is  required only if you are doing  automatic Data cleaning and Pre-processing without any Manual steps.
from PreProcess_Data import Xtrain,Xtest,Ytrain,Ytest
from evalml import AutoMLSearch
evalml.problem_types.ProblemTypes.all_problem_types

from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

X_train, X_test, y_train, y_test = Xtrain,Xtest,Ytrain,Ytest

print("\n\n\tRunning Auto ML based  training\n")

automl = AutoMLSearch(X_train=Xtrain, y_train=Ytrain, problem_type='binary')
print(automl.search())

automl.rankings
print(automl.best_pipeline)

best_pipeline=automl.best_pipeline

print(best_pipeline)

#GeneratedPipeline(parameters={'Imputer':{'categorical_impute_strategy': 'most_frequent', 'numeric_impute_strategy': 'mean',
#                            'categorical_fill_value': None, 'numeric_fill_value': None}, 'Logistic Regression Classifier'
#:{'penalty': 'l2', 'C': 1.0, 'n_jobs': -1, 'multi_class': 'auto', 'solver': 'lbfgs'},})

automl.describe_pipeline(automl.rankings.iloc[0]["id"])

### Evaluate on hold out of the data samples
best_pipeline.score(X_test, y_test, objectives=["auc","f1","Precision","Recall"])

automl_auc.rankings
automl_auc.describe_pipeline(automl_auc.rankings.iloc[0]["id"])

best_pipeline_auc = automl_auc.best_pipeline
# get the score on holdout data
best_pipeline_auc.score(X_test, y_test,  objectives=["auc"])

## Pickling the trained model
best_pipeline.save("AutomML_Eval_model.pkl")

check_model=automl.load('model.pkl')
check_model.predict_proba(X_test).to_dataframe()





