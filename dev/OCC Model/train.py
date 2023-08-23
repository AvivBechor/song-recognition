r'''
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report
import numpy as np 
import pandas as pd 
import pickle

file_name = "model.model"
train_data_path = r"C:\Users\avivb\Desktop\School\Collage\14th grade\Final Project\Data Sets\Data\audio_features.csv"
train_data = pd.read_csv(train_data_path)
path = r".\{file_name}"
train_data = train_data.iloc[:, 1:len(list(train_data.columns)) - 1:1]
  
input_data = np.array(train_data.values.tolist())

lof_novelty = LocalOutlierFactor(n_neighbors=5, novelty=True).fit(input_data)

pickle.dump(lof_novelty, open(file_name,"wb"))
'''

from sklearn import svm
import numpy as np 
import pandas as pd 
import pickle

file_name = "mixed_model.model"
train_data_path = r"C:\Users\avivb\Desktop\School\Collage\14th grade\Final Project\OCC Model\mixed_audio_features1.csv"
train_data = pd.read_csv(train_data_path)
path = r".\{file_name}"
train_data = train_data.iloc[:, 1:len(list(train_data.columns)) - 1:1]
train_data=train_data.fillna(0)
input_data = np.array(train_data.values.tolist())

clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
model = clf.fit(input_data)
pickle.dump(model, open(file_name,"wb"))
