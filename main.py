### Load Modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
### Prepare/collect data
import os
dataset = os.path.join(os.getcwd(),"Dataset")
training = os.path.join(dataset, "Training")
testing = os.path.join(dataset, "Testing")
classes = {'Tumor':0, 'Normal':1}
print(os.getcwd(), dataset, training, testing)
import cv2
X = []
Y = []
for cls in classes:
 pth = os.path.join(training,cls)
 for j in os.listdir(pth):
 img = cv2.imread(pth+'/'+j, 0)
 img = cv2.resize(img, (200,200))
 X.append(img)
 Y.append(classes[cls])
X = np.array(X)
Y = np.array(Y)
X_updated = X.reshape(len(X), -1)
np.unique(Y)
pd.Series(Y).value_counts()
X.shape, X_updated.shape
### Visualize data
plt.imshow(X[0], cmap='gray')
### Prepare data
X_updated = X.reshape(len(X), -1)
X_updated.shape
### Split Data
xtrain, xtest, ytrain, ytest = train_test_split(X_updated, Y, random_state=10,
 test_size=.20)
xtrain.shape, xtest.shape
### Feature Scaling
print(xtrain.max(), xtrain.min())
print(xtest.max(), xtest.min())
xtrain = xtrain/255
xtest = xtest/255
print(xtrain.max(), xtrain.min())
print(xtest.max(), xtest.min())
### Feature Selection: PCA
from sklearn.decomposition import PCA
print(xtrain.shape, xtest.shape)
pca = PCA(.98)
# pca_train = pca.fit_transform(xtrain)
# pca_test = pca.transform(xtest)
pca_train = xtrain
pca_test = xtest
# print(pca_train.shape, pca_test.shape)
# print(pca.n_components_)
# print(pca.n_features_)
### Train Model
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')
lg = LogisticRegression(C=0.1)
lg.fit(xtrain, ytrain)
sv = SVC()
sv.fit(xtrain, ytrain)
### Evaluation
print("Training Score:", lg.score(xtrain, ytrain))
print("Testing Score:", lg.score(xtest, ytest))
print("Training Score:", sv.score(xtrain, ytrain))
print("Testing Score:", sv.score(xtest, ytest))
### Prediction
pred = sv.predict(xtest)
misclassified=np.where(ytest!=pred)
misclassified
print("Total Misclassified Samples: ",len(misclassified[0]))
print(pred[36],ytest[36])
### TEST MODEL
dec = {0:'Tumor', 1:'Normal'}
plt.figure(figsize=(12,8))
c=1
tumor = os.path.join(testing,"Tumor")
for i in os.listdir(tumor)[:8]:
 plt.subplot(4,4,c)
 
 img = cv2.imread(os.path.join(tumor,i),0)
 img1 = cv2.resize(img, (200,200))
 img1 = img1.reshape(1,-1)/255
 p = sv.predict(img1)
 plt.title(dec[p[0]])
 plt.imshow(img, cmap='gray')
 plt.axis('off')
 c+=1
plt.figure(figsize=(12,8))
c=1
normal = os.path.join(testing, "Normal")
for i in os.listdir(normal)[:6]:
 plt.subplot(4,4,c)
 img = cv2.imread(os.path.join(normal, i),0)
 img1 = cv2.resize(img, (200,200))
 img1 = img1.reshape(1,-1)/255
 p = sv.predict(img1)
 plt.title(dec[p[0]])
 plt.imshow(img, cmap='gray')
 plt.axis('off')
 c+=1
# Save the Model to disk
import joblib
joblib.dump(sv, "tumormodel.joblib")
