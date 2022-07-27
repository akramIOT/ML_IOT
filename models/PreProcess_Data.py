
import pandas as pd
import numpy as  np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

X = pd.read_csv("/Users/akram/AKRAM_CODE_FOLDER/ML/Washington_ML/serverless-machine-learning/ML_Proj_Template/ml1/data/external/benign_traffic.csv")
y = pd.read_csv("/Users/akram/AKRAM_CODE_FOLDER/ML/Washington_ML/serverless-machine-learning/ML_Proj_Template/ml1/data/external/scan.csv")
print(" ::Reading of Input Data is Sucessfull:: \n")
print(X)
print(X.head())
print(f"The shape of  Input  dataset is : {X.shape}")
print(f"The shape of  Input malicious  dataset is : {y.shape}")

## As per the Data Description of this Data Set, security threat infected (malicious) Traffic to be marked as "1" and Clean(benign) traffic to be marked
## as  "0". Ref: https://archive.ics.uci.edu/ml/datasets/detection_of_IoT_botnet_attacks_N_BaIoT# ,  https://archive.ics.uci.edu/ml/machine-learning-databases/00442/

X['Out'] = 1
y['Out'] = 0

print(f"Clean/ Benign Traffic is  {X['Out']}")
print(f"Malicious Traffic is  {y['Out']}")

combined = pd.concat([X,y], axis =0)
print(f"Concatenated Data Shape is {combined.shape}")

print(f" combined1 shape is {combined.shape}")

## Shuffling the  Input Data Set
combined = shuffle(combined)
Output = combined['Out']
combined=combined.drop(['Out'],axis=1)
combined=combined.drop(['HpHp_L0.01_pcc'],axis=1)
combined1 =combined.iloc[:,:28] # Removing all the Outlier Columns

Output=np.array(Output).flatten()

print("After remove: ",combined.shape)
print("\nThe OUTPUT is : \n",Output)
print("\nOUTPUT SHAPE : ",Output.shape)

#Standardization
comb_std=(combined-(combined.mean()))/(combined.std())
comb_std_arra=np.array(comb_std)

#Using SKLearn
scale=StandardScaler()
scale.fit(combined1)
scale.transform(combined1)
#X1.to_csv('combined1.csv')

#Normalization
comb_norm=(X-(X.min()))/((X.max())-X.min())
#comb_norm_arra=np.array(comb_norm)

#print("After Norm : ",comb_norm_arra.shape)

#Testing/Training Data
Xtrain,Xtest,Ytrain,Ytest=train_test_split(combined1,Output,test_size=0.3,random_state=1)##########################