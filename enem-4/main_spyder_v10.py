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


# %%Label Encoder
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

# %%Bypass label encoder e selecionar só algumas features

train  = train[['TP_ST_CONCLUSAO', 'NU_IDADE', 'TP_ANO_CONCLUIU', 'TP_ESCOLA','IN_TREINEIRO']]
test = train[['TP_ST_CONCLUSAO', 'NU_IDADE', 'TP_ANO_CONCLUIU', 'TP_ESCOLA']]
train_array =  train.values
test_array = test.values
#%%Bypass porque vamos utilizar PCA
train_array =  train.values
test_array = test.values

# %% Preencher os valores missing com 0:
from sklearn.impute import SimpleImputer

train_imputer = SimpleImputer(missing_values=np.nan, strategy='constant',fill_value = 0)
test_imputer = SimpleImputer(missing_values=np.nan, strategy='constant',fill_value = 0)

train_array = train_imputer.fit_transform(train_array)
test_array = test_imputer.fit_transform(test_array)

#%%Separação X_train e target
# Foi necessário converter para float, porque na regressão logística indicava que 
# y_train era object
X_train = train_array[:,:-1].astype(float)
y_train = train_array[:,-1].astype(float)

submission = test_array


# %% Normalização dos dados:
from sklearn.preprocessing import StandardScaler
ss_X_train = StandardScaler()
ss_X_test = StandardScaler()

X_train_scaled = ss_X_train.fit_transform(X_train)
X_submission_scaled = ss_X_test.fit_transform(submission)


# %%Bypass StandardScaler

X_train_scaled = X_train
X_submission_scaled = submission


# %%SMOTE para reamostrar e resolver a questão da calsse minoritária
from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy='minority')

X_smote, y_smote =  smote.fit_resample(X = X_train_scaled, y = y_train)


# %% Bypass smote
X_smote = X_train_scaled
y_smote = y_train
# %%PCA:
from sklearn.decomposition import PCA
pca_train = PCA(n_components=5)
pca_fitted_train = pca_train.fit(X_smote)
pca_data_train = pca_fitted_train.transform(X_smote)

pca_submission = PCA(n_components = 5)
pca_fitted_submission = pca_submission.fit(X_submission_scaled)
pca_data_submission = pca_fitted_submission.transform(X_submission_scaled)

#sns.scatterplot(pca_data[:,0], pca_data[:,1], hue = y_smote)

X_smote = pca_data_train
X_test = pca_data_submission
# %%Separação entre dados de teste e de treinamento (na base de treinamento )
# Ainda não é a previsão final sobre os dados de test (submission) para serem submetidos

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, 
                                                    test_size = 0.3, 
                                                    random_state = 42)

# %%Random Forest:
from sklearn.ensemble import RandomForestClassifier

model =  RandomForestClassifier(n_estimators=40,
                                criterion = 'entropy',
                                   random_state=42)

model.fit(X_train, y_train)

# %% Predição
predicao = model.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score

matriz = confusion_matrix(y_test, predicao); matriz

acuracia = accuracy_score(y_test, predicao); acuracia

# %%Previsão para os dados submission:
    
submission_predict = model.predict(X_submission_scaled)

#Neural:
submission_predict = np.round(model.predict(X_submission_scaled)).reshape(-1)

test = pd.read_csv('test.csv')

new_df = np.array(list(zip(test.NU_INSCRICAO.values, submission_predict)))

df_answer = pd.DataFrame(new_df, 
                         columns = ['NU_INSCRICAO','IN_TREINEIRO'])

df_answer.to_csv('answer.csv', sep = ',', index = False)




