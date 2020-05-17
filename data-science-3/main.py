#!/usr/bin/env python
# coding: utf-8

# # Desafio 5
# 
# Neste desafio, vamos praticar sobre redução de dimensionalidade com PCA e seleção de variáveis com RFE. Utilizaremos o _data set_ [Fifa 2019](https://www.kaggle.com/karangadiya/fifa19), contendo originalmente 89 variáveis de mais de 18 mil jogadores do _game_ FIFA 2019.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


from math import sqrt

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
import statsmodels.api as sm
import statsmodels.stats as st
from sklearn.decomposition import PCA

from loguru import logger


# In[2]:


fifa = pd.read_csv("fifa.csv")


# In[3]:


columns_to_drop = ["Unnamed: 0", "ID", "Name", "Photo", "Nationality", "Flag",
                   "Club", "Club Logo", "Value", "Wage", "Special", "Preferred Foot",
                   "International Reputation", "Weak Foot", "Skill Moves", "Work Rate",
                   "Body Type", "Real Face", "Position", "Jersey Number", "Joined",
                   "Loaned From", "Contract Valid Until", "Height", "Weight", "LS",
                   "ST", "RS", "LW", "LF", "CF", "RF", "RW", "LAM", "CAM", "RAM", "LM",
                   "LCM", "CM", "RCM", "RM", "LWB", "LDM", "CDM", "RDM", "RWB", "LB", "LCB",
                   "CB", "RCB", "RB", "Release Clause"
]

try:
    fifa.drop(columns_to_drop, axis=1, inplace=True)
except KeyError:
    logger.warning(f"Columns already dropped")


# ## Inicia sua análise a partir daqui

# In[4]:


# Sua análise começa aqui.


# ## Questão 1
# 
# Qual fração da variância consegue ser explicada pelo primeiro componente principal de `fifa`? Responda como um único float (entre 0 e 1) arredondado para três casas decimais.

# #### Análise de consistência das variáveis:

# In[5]:


cons = pd.DataFrame({'colunas': fifa.columns, 
                    'tipo': fifa.dtypes,
                    'missing': fifa.isna().sum(),
                    'size': fifa.shape[0],
                    'unicos': fifa.nunique()})

cons['percentual'] = round(cons['missing'] / cons['size'],4)


# #### Percentual de missing 

# In[6]:


cons.percentual.plot.hist(bins = 3)


# #### Remover valores missing

# In[7]:


fifa.dropna(inplace = True)


# In[8]:


#Inicializando o objeto PCA:
pca = PCA().fit(fifa)


# In[9]:


def q1():
    return round(pca.explained_variance_ratio_[0],3)
    # Retorne aqui o resultado da questão 1.
    pass
q1()


# ## Questão 2
# 
# Quantos componentes principais precisamos para explicar 95% da variância total? Responda como un único escalar inteiro.

# In[10]:


cumulative_evr = np.cumsum(pca.explained_variance_ratio_)
component_95 = np.argmax(cumulative_evr >=0.95) + 1
component_95


# #### Visualização por Screeplot

# In[11]:


evr = pca.explained_variance_ratio_
pcascreeplot = pd.DataFrame({'var': evr, 'PC':np.arange(1,len(evr)+1)})
fig = sns.lineplot(np.arange(len(evr)), np.cumsum(evr))
sns.barplot(x = 'PC', y = 'var', data = pcascreeplot)
fig.axes.axhline(0.95, ls = "--", color = "red")
plt.xlabel('Componentes')
plt.ylabel('Variância explicada')


# In[12]:


def q2():
    return component_95
    # Retorne aqui o resultado da questão 2.
    pass
q2()


# ## Questão 3
# 
# Qual são as coordenadas (primeiro e segundo componentes principais) do ponto `x` abaixo? O vetor abaixo já está centralizado. Cuidado para __não__ centralizar o vetor novamente (por exemplo, invocando `PCA.transform()` nele). Responda como uma tupla de float arredondados para três casas decimais.

# In[13]:


x = [0.87747123,  -1.24990363,  -1.3191255, -36.7341814,
     -35.55091139, -37.29814417, -28.68671182, -30.90902583,
     -42.37100061, -32.17082438, -28.86315326, -22.71193348,
     -38.36945867, -20.61407566, -22.72696734, -25.50360703,
     2.16339005, -27.96657305, -33.46004736,  -5.08943224,
     -30.21994603,   3.68803348, -36.10997302, -30.86899058,
     -22.69827634, -37.95847789, -22.40090313, -30.54859849,
     -26.64827358, -19.28162344, -34.69783578, -34.6614351,
     48.38377664,  47.60840355,  45.76793876,  44.61110193,
     49.28911284
]


# #### Expressar este vetor x, que equivale a uma observação, com 37 variáveis, em 2 componentes principais extraídos a partir da base de dados fifa

# In[14]:


#Utilizaremos, portanto, apenas 2 componentes
pca = PCA(n_components=2)
pca.fit(fifa)


# In[15]:


#Já possuímos as duas combinações lineares das variáveis que representam a maior porcentagem da variância total. 
pca.components_


# #### Sabe-se que:
# $$Z_{1} = \phi_{11}X_{1} + \phi_{21}X_{2} + \cdots + \phi_{p1}X_{p} = \sum_{1 \leq j \leq p} \phi_{j1}X_{j}$$
# 
# Portanto, basta multiplicar os loadings, ou seja, os coeficientes que expressam PC1 e PC2 em função de X1, X2, ... , Xn para obter as coordenadas de `x` nos novos componentes principais 

# In[16]:


def q3():
    return tuple([round(x,3) for x in pca.components_.dot(x)])
    # Retorne aqui o resultado da questão 3.
    pass

q3()


# ## Questão 4
# 
# Realiza RFE com estimador de regressão linear para selecionar cinco variáveis, eliminando uma a uma. Quais são as variáveis selecionadas? Responda como uma lista de nomes de variáveis.

# In[17]:


from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

X_train =fifa.drop(columns='Overall') 
y_train =fifa['Overall']

regressor.fit(X_train, y_train)


# In[18]:


from sklearn.feature_selection import RFE


#step = 1  significa remover as variáveis uma a uma até chegar em 5, neste caso 
rfe = RFE(regressor, n_features_to_select=5, step = 1)

rfe.fit(X_train, y_train)


# In[19]:


df_rfe = pd.DataFrame({'features':list(X_train),
                      'selecionadas':rfe.get_support()})


# In[20]:


def q4():
    return list(df_rfe[df_rfe.selecionadas == True]['features'])
    # Retorne aqui o resultado da questão 4.
    pass

q4()

