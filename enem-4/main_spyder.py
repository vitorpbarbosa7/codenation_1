import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(f'Shape Test: {test.shape} and Shape Train: {train.shape}')
    
test_features = list(test)
train_features = list(train)

newtrain_features = list(set(train_features).intersection(set(test_features)))

newtrain_features.append('IN_TREINEIRO')

train = train[newtrain_features]

### Retornar o número das colunas que devem receber o tratamento labelencoder 

train = train.drop('NU_INSCRICAO', axis = 1)
test = test.drop('NU_INSCRICAO', axis = 1)

object_features = list(train.select_dtypes(include = 'object'))

feat_le = [train.columns.get_loc(a) for a in object_features]
[feat_le.remove(c) for c in [32,33]]

train_array = train.values
#test_array = test.values

from sklearn.preprocessing import LabelEncoder
labelencoder_previsores = LabelEncoder()

for ft in feat_le:
  train_array[:,ft] = labelencoder_previsores.fit_transform(train_array[:,ft])