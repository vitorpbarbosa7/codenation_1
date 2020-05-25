#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk


# # Algumas configurações para o matplotlib.

# ## _Setup_ geral

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk


# In[4]:


countries = pd.read_csv("countries.csv")


# In[5]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.head(5)


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# #### Temos 2 variáveis categóricas, as quais são as duas primeiras colunas e 18 variáveis numéricas, todas à direita 

# In[6]:


countries.dtypes


# In[7]:


#Convertiz tudo para string primeiro, porque ele dizia que os object eram float64
countries = countries.astype(str)
countries.info()


# In[8]:


float_columns = list(countries) 
del float_columns[0:4]


# In[9]:


float_columns


# In[10]:


for col in float_columns:
    countries[col] = countries[col].str.replace(',','.').astype(float)


# In[11]:


for col in ['Population','Area']:
    countries[col] = countries[col].astype(int)


# In[12]:


countries.dtypes


# In[13]:


countries.head()


# ### Retirar os espaços das features Country e Region

# In[14]:


countries_ = countries


# In[15]:


countries_['Country'] = countries_['Country'].str.strip()
countries_['Region'] = countries['Region'].str.strip()


# ### Base com retirada dos espaços e substituição das vírgulas por pontos:

# In[16]:


countries_.head()


# ## Inicia sua análise a partir daqui

# In[17]:


# Sua análise começa aqui.


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[18]:


region_list = list(countries_.Region.unique())


# In[20]:


region_list.sort()


# In[21]:


def q1():
    # Retorne aqui o resultado da questão 1.
    return region_list
    

q1()


# In[22]:


type(q1())


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[23]:


from sklearn.preprocessing import KBinsDiscretizer


# In[24]:


#Diferença entre chamar com ponto . e com [['']]:

# O 'Discretizer' pede dimensão 2


# In[25]:


countries_['Pop_density']


# In[26]:


countries_.Pop_density.shape


# In[27]:


countries_[['Pop_density']].shape


# In[28]:


discretizer = KBinsDiscretizer(n_bins=10, encode = 'ordinal', strategy='quantile')

discretizer.fit(countries_[['Pop_density']])

Pop_density_bins = pd.DataFrame(discretizer.transform(countries_[['Pop_density']]), columns = ['Pop_density_bins'])


# In[29]:


countries_ = pd.concat([countries_, Pop_density_bins], axis = 1)


# In[30]:


#Desta maneira consigo retornar o valor do índice, utilizando [] para referenciar, como está presente na resposta da questão
countries_.Pop_density_bins.value_counts().index


# In[31]:


def q2():
    return countries_.Pop_density_bins.value_counts()[9]# Retorne aqui o resultado da questão 2.
    pass
q2()


# In[32]:


type(q2())


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[33]:


#Nuḿero de categorias (se missing)
climate_attr = countries_.Climate.nunique(); climate_attr


# In[34]:


#Missing também seria uma categoria? 
countries_[['Climate']].isna().sum().sum()


# In[35]:


#Adicionando a categoria missing
climate_attr = climate_attr + 1; climate_attr


# In[36]:


countries_.Region.nunique()


# In[37]:


#Número de categorias da Region
region_attr = countries_.Region.nunique(); region_attr


# In[38]:


#Há missing em Region? não, então não será adicionada nova categoria
countries_[['Region']].isna().sum().sum()


# In[39]:


from sklearn.preprocessing import OneHotEncoder
countries_['Climate'].fillna(0, inplace = True)
one_hot_encoder = OneHotEncoder(sparse=False, dtype = np.int)
encoded_region_climate = one_hot_encoder.fit_transform(countries_[['Climate','Region']])


# In[40]:


#Sâo portanto 18 novos atributos, como indicava a análise anterior
encoded_region_climate.shape


# In[41]:


a3 = np.int(encoded_region_climate.shape[1])


# In[42]:


def q3():
    # Retorne aqui o resultado da questão 3.
    return a3


# In[43]:


q3()


# In[44]:


type(q3())


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[45]:


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# ### Criação do pipeline

# In[46]:


pipeline = Pipeline(steps = [
    ('SimpleImputer', SimpleImputer(strategy = 'median')),
    ('StandardScaler', StandardScaler())
])


# Aplicação do pipeline:

# In[47]:


pipe_float = pipeline.fit_transform(countries_.iloc[:,list(range(2,20))]); pipe_float


# In[48]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# Tratamento para poder aplicar o transform na nova observação test_country:

# In[49]:


df_test_country = pd.DataFrame(test_country).T; df_test_country


# Voltar os nomes das colunas:

# In[50]:


df_test_country.columns = list(countries); df_test_country


# Aplicar o transform do pipeline que foi criado com o fit na base completa:

# In[51]:


test_country_pipe = pipeline.transform(df_test_country.drop(columns = ['Country','Region'], axis = 1))


# Facilitar visualização com um dataframe, de modo a visualizar rapidamente qual é a variável Arable

# In[52]:


df_test_country_pipe = pd.DataFrame(test_country_pipe, columns = list(countries.drop(columns = ['Country','Region'], axis = 1)))
df_test_country_pipe


# In[53]:


def q4():
    return round(df_test_country_pipe.loc[0,'Arable'], 3)
    # Retorne aqui o resultado da questão 4.
    pass

q4()


# In[54]:


type(q4())


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# ### Visualização de outliers através do boxplot

# In[55]:


sns.boxplot(countries_.Net_migration, orient = 'vertical')


# ### Visualização de outliers através de histograma

# In[56]:


sns.distplot(countries_.Net_migration)


# Visualização dos 5 primeiros países com maior net migration 

# In[57]:


countries_.sort_values(by=['Net_migration'], ascending = False).head(5)


# Visualização dos 5 países com menor Net Migration

# In[58]:


countries_.sort_values(by = ['Net_migration'], ascending = True).head(5)


# ### Não deve-se remover estes outliers, eles são dados reais que indicam as taxas de Imigração de um país.
# ### Os outliers só indicam que os países com valores acima de 15 por exemplo são países que entram muito mais pessoas do que saem e vice-versa para aqueles com valores abaixo de -15 aproximadamente

# In[59]:


#NÃO denominei o quant1 e quant3 de q1 e q3, porque na hora de submeter o resultado do desafio, iria dar conflito e erro com a função q1() e q3()
quant1 = countries_.Net_migration.quantile(0.25)
quant3 = countries_.Net_migration.quantile(0.75)
iqr = quant3-quant1

outlier_limits = [quant1-1.5*iqr, quant3+1.5*iqr]

outliers_inferior = countries_[countries_['Net_migration'] < outlier_limits[0]]
outliers_superior = countries_[countries_['Net_migration'] > outlier_limits[1]]


# In[60]:


def q5():
    return (outliers_inferior.shape[0], outliers_superior.shape[0], False)
    # Retorne aqui o resultado da questão 4.
    pass

q5()


# In[61]:


type(q5())


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[62]:


from sklearn.datasets import fetch_20newsgroups


# In[63]:


#Definição das bibliotecas de documentos para importar:
categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']


# In[64]:


#Carregando o subset de treino? (mas não era para carregar o de teste?)
#selecionando as categorias
#"Embaralhando" os documentos do corpus com random_state= 42 

newsgroups = fetch_20newsgroups(subset = 'train', 
                               categories = categories,
                               shuffle = True,
                               random_state = 42)


# ### Aplicar o Count Vectorizer para descobrir quantas vezes a palavra j aparece no documento i 

# In[65]:


from sklearn.feature_extraction.text import CountVectorizer


# In[66]:


count_vectorizer = CountVectorizer()
newsgroups_counts = count_vectorizer.fit_transform(newsgroups.data)


# In[67]:


#Passando para dataframe:
df_newsgroups_vectorizer = pd.DataFrame(newsgroups_counts.toarray(), columns= count_vectorizer.get_feature_names())
df_newsgroups_vectorizer
                                       


# In[68]:


#Contabilizar o número de vezes que aparece a palavra phone


# In[69]:


def q6():
    return df_newsgroups_vectorizer['phone'].sum()
    # Retorne aqui o resultado da questão 4.
    pass

q6()


# In[70]:


type(q6())


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[71]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[72]:


tfidf_vectorizer = TfidfVectorizer()

tfidf_vectorizer.fit(newsgroups.data)

newsgroups_tfidf_vectorized = tfidf_vectorizer.transform(newsgroups.data)


# In[73]:


df_newsgroups_tfidf_vectorized = pd.DataFrame(newsgroups_tfidf_vectorized.toarray(),
                                              columns = tfidf_vectorizer.get_feature_names())


# In[74]:


df_newsgroups_tfidf_vectorized


# In[75]:


def q7():
    return round(df_newsgroups_tfidf_vectorized['phone'].sum(),3)
    # Retorne aqui o resultado da questão 4.
    pass

q7()


# In[76]:


type(q7())


# ## Qual o maior tfidf?

# In[77]:


tfidf_todas = [df_newsgroups_tfidf_vectorized.iloc[:,i].sum() for i in list(range(0,df_newsgroups_tfidf_vectorized.shape[1]))]


# In[78]:


names = tfidf_vectorizer.get_feature_names()


# In[79]:


len(names)


# In[80]:


len(tfidf_todas)


# In[81]:


df_tfidf_importance = pd.DataFrame({'Palavras': names, 'Tfidf': tfidf_todas})


# In[82]:


df_tfidf_importance.sort_values(by='Tfidf', ascending = False).head(10)

