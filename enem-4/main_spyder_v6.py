import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#%% Filtrar train com as features de test
test_features = list(test)
train_features = list(train)

newtrain_features = list(set(train_features).intersection(set(test_features)))

newtrain_features.append('IN_TREINEIRO')

train = train[newtrain_features]

### Retornar o número das colunas que devem receber o tratamento labelencoder 

train = train.drop('NU_INSCRICAO', axis = 1)
test = test.drop('NU_INSCRICAO', axis = 1)

object_features_train = list(train.select_dtypes(include = 'object'))
object_features_test = list(test.select_dtypes(include = 'object'))

# Converter para string todos do tipo object (isso não deveria ser necessário né, mas foi, sem isso dava erro):
def obj2str(df, features):
    for a in features:
        df[a] = df[a].astype(str)
    return df

train = obj2str(train, object_features_train)
test = obj2str(test, object_features_test)

feat_le_train = [train.columns.get_loc(a) for a in object_features_train]
feat_le_test = [test.columns.get_loc(a) for a in object_features_test]
#[feat_le.remove(c) for c in [32,33]]

train_array = train.values
test_array = test.values

from sklearn.preprocessing import LabelEncoder

def label_encoder(df, feat_le):
    le = LabelEncoder()
    for ft in feat_le:
        df[:,ft] = le.fit_transform(df[:,ft])
    return df 

train_array = label_encoder(train_array, feat_le_train)
test_array = label_encoder(test_array, feat_le_test)
# %% Preencher os valores missing com 0:
from sklearn.impute import SimpleImputer

train_imputer = SimpleImputer(missing_values=np.nan, strategy='constant',fill_value = 0)
test_imputer = SimpleImputer(missing_values=np.nan, strategy='constant',fill_value = 0)

train_array = train_imputer.fit_transform(train_array)
test_array = test_imputer.fit_transform(test_array)

#%%Separação X_train e target
X_train = train_array[:,:-1]
y_train = train_array[:,-1]

submission = test_array


# %% Normalização dos dados:
from sklearn.preprocessing import StandardScaler
ss_X_train = StandardScaler()
ss_X_test = StandardScaler()

X_train_scaled = ss_X_train.fit_transform(X_train)
X_submission_scaled = ss_X_test.fit_transform(submission)

# %%Separação entre dados de teste e de treinamento (na base de treinamento )
# Ainda não é a previsão final sobre os dados de test (submission) para serem submetidos

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train_scaled, y_train, 
                                                    test_size = 0.3, 
                                                    random_state = 42)


# %%Regressão Logistica:
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

lr.fit(X_train, y_train)




