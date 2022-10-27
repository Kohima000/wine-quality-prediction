
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

wine_dataset=pd.read_csv("winequality-red.csv")
#print(wine_dataset.shape)
#print(wine_dataset.head())
#print(wine_dataset.isnull().sum())
#print(wine_dataset.describe)
#print(sns.catplot(x="quality",data=wine_dataset,kind="count"))
plot=plt.figure(figsize=(5,5))
print(sns.barplot(x="quality",y="volatile acidity",data=wine_dataset))
print(sns.barplot(x="quality",y="citric acid",data=wine_dataset))
#correlation:finding proportionality relation
#correlation=wine_dataset.corr()
#plt.figure(figsize=(10,10))
#print(sns.heatmap(correlation,cbar=True,fmt=".1f",annot=True,annot_kws={"size":8},cmap="Blues"))

#LABEL BINARIZATION OR LABEL ENCODING: to divide the label(the column on which we are predicting) into good or bad or 0 or 1)
X=wine_dataset.drop("quality",axis=1)
Y=wine_dataset["quality"].apply(lambda y_value: 1 if y_value>=7 else 0)
#
##TRAIN AND TEST SPLIT
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=2)
print(Y.shape,Y_test.shape,Y_train.shape)
#
##MODEL TRAINING: rANDOM MODEL CLASSIFIER
model=RandomForestClassifier(n_estimators=100)
model.fit(X_train,Y_train)
X_test_pred=model.predict(X_test)
test_data_accuracy=accuracy_score(X_test_pred,Y_test)
print("Accuracy: ",test_data_accuracy)

#Building a predictive system
input_data=(7.3,0.65,0.0,1.2,0.065,15.0,21.0,0.9946,3.39,0.47,10.0)
input_data_numpy=np.asarray(input_data)
#print(input_data_numpy)
inpt_data_reshape=input_data_numpy.reshape(1,-1)
prediction=model.predict(inpt_data_reshape)
print(prediction)
if prediction[0]==1:
    print("Good Quality")
else:
    print("Bad quality")




#
#
