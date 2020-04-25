#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[3]:


type(black_friday.index)


# In[4]:


#Q1: Número de linhas no dataset:

a1 = (black_friday.shape[0], black_friday.shape[1])
a1


# In[5]:


black_friday.sample(10)


# In[6]:


#Q2
list(black_friday)


# In[7]:


mulheres = black_friday[black_friday['Gender'] == 'F'].shape[0]
homens = black_friday[black_friday['Gender'] == 'M'].shape[0]
print('Número de mulheres é de ', str(mulheres), 'e de homens de :', str(homens), '\n', 
     'Este total é de', str(mulheres + homens), 'E o total de linhas no dataframe é de:', str(black_friday.shape[0]))


# In[8]:


#O negócio já tava em bin e eu aqui me matando parar criar condições baseadas em ints
# f_f = black_friday['Gender'] == 'F'
# f_f_26_35 = black_friday['Age'] == '26-35'
# all_filters = f_f & f_f_26_35
# a2 = black_friday[all_filters]
a2 = black_friday[(black_friday['Gender'] == 'F') & (black_friday['Age'] == '26-35')].shape[0]


# In[42]:


#Q3
a3 = black_friday['User_ID'].nunique()
a3


# In[10]:


#Q4
a4 = black_friday.dtypes.unique().shape[0]
a4


# In[11]:


#Q5
black_friday.isnull().sum(axis = 0)


# In[12]:


black_friday.isnull().any(axis = 1).sum()


# In[13]:


linhas_vazias = black_friday.isnull().any(axis = 1).sum()
a5 = linhas_vazias/black_friday.shape[0]
a5


# In[14]:


print("Há um total de ",linhas_vazias, "registros com pelo menos uma coluna vazia, de um total de ", black_friday.shape[0], "registros")


# In[15]:


#Q6
#Já sei qual é a coluna que contém a maior quantidade de null, então:
a6 = black_friday['Product_Category_3'].isnull().sum()
a6


# In[49]:


#Q7
#Valor mais frequente em uma série de dados é claro que é a moda
a7 = black_friday['Product_Category_3'].mode()[0]
a7


# In[17]:


#Q8
from sklearn import preprocessing
import numpy as np


# In[18]:


X = black_friday['Purchase'].values
X


# In[19]:


X.shape


# In[20]:


min_max_value = preprocessing.MinMaxScaler()
X = black_friday['Purchase'].values.astype(float)


# In[21]:


X = np.reshape(X, (-1,1))
X


# In[22]:


purchase_norm = min_max_value.fit_transform(X)


# In[23]:


black_friday['Purchase'].mean()


# In[24]:


a8 = np.mean(purchase_norm)


# In[25]:


print("A média antes da normalização era de ", black_friday['Purchase'].mean(), 
      "e agora a média após a normalização é de", np.mean(purchase_norm))


# In[26]:


#Q8
scaler = preprocessing.StandardScaler()
standard_purchase = scaler.fit_transform(X)
import matplotlib.pyplot as plt
plt.hist(standard_purchase)


# In[27]:


print("Após a normalização, a média do valor de compra passou a ser de ", np.mean(standard_purchase), 
      "E o desvio padrão de", np.std(standard_purchase))


# In[28]:


df_purchase_stand = pd.DataFrame(standard_purchase)


# In[29]:


df_purchase_stand.columns = ['Valores']
df_purchase_stand


# In[30]:


a9 = df_purchase_stand[(df_purchase_stand['Valores'] > -1) & (df_purchase_stand['Valores']< 1)].shape[0]
a9


# In[31]:


#Q10


# In[32]:


a10 = True


# In[35]:


a10


# In[50]:


lista = [a1,a2,a3,a4,a5,a6,a7,a8,a9,a10]


# In[51]:


lista


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[126]:


def q1():
    return a1
    # Retorne aqui o resultado da questão 1.
    pass


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[127]:


def q2():
    a2
    return a2# Retorne aqui o resultado da questão 2.
    pass


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[128]:


def q3():
    a3
    return a3# Retorne aqui o resultado da questão 3.
    pass


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[129]:


def q4():
    a4
    return a4# Retorne aqui o resultado da questão 4.
    pass


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[130]:


def q5():
    a5
    return a5
    # Retorne aqui o resultado da questão 5.
    pass


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[131]:


def q6():
    a6
    return a6
    # Retorne aqui o resultado da questão 6.
    pass


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[132]:


def q7():
    a7
    return a7
    # Retorne aqui o resultado da questão 7.
    pass


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[133]:


def q8(): 
    a8
    return a8
    # Retorne aqui o resultado da questão 8.
    pass


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[134]:


def q9():
    a9
    return a9# Retorne aqui o resultado da questão 9.
    pass


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[135]:


def q10():
    a10
    return a10
    # Retorne aqui o resultado da questão 10.
    pass

