#!/usr/bin/env python
# coding: utf-8

# # Desafio 4
# 
# Neste desafio, vamos praticar um pouco sobre testes de hipóteses. Utilizaremos o _data set_ [2016 Olympics in Rio de Janeiro](https://www.kaggle.com/rio2016/olympic-games/), que contém dados sobre os atletas das Olimpíadas de 2016 no Rio de Janeiro.
# 
# Esse _data set_ conta com informações gerais sobre 11538 atletas como nome, nacionalidade, altura, peso e esporte praticado. Estaremos especialmente interessados nas variáveis numéricas altura (`height`) e peso (`weight`). As análises feitas aqui são parte de uma Análise Exploratória de Dados (EDA).
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns


# In[2]:


df = pd.read_csv("athletes.csv")


# In[3]:


def get_sample(df, col_name, n=100, seed=42):
    """Get a sample from a column of a dataframe.
    
    It drops any numpy.nan entries before sampling. The sampling
    is performed without replacement.
    
    Example of numpydoc for those who haven't seen yet.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Source dataframe.
    col_name : str
        Name of the column to be sampled.
    n : int
        Sample size. Default is 100.
    seed : int
        Random seed. Default is 42.
    
    Returns
    -------
    pandas.Series
        Sample of size n from dataframe's column.
    """
    np.random.seed(seed)
    
    random_idx = np.random.choice(df[col_name].dropna().index, size=n, replace=False)
    
    ##Retornar as linhas random_idx com a coluna col_name
    return df.loc[random_idx, col_name]


# ## Inicia sua análise a partir daqui

# In[4]:


# Sua análise começa aqui.


# ## Questão 1
# 
# Considerando uma amostra de tamanho 3000 da coluna `height` obtida com a função `get_sample()`, execute o teste de normalidade de Shapiro-Wilk com a função `scipy.stats.shapiro()`. Podemos afirmar que as alturas são normalmente distribuídas com base nesse teste (ao nível de significância de 5%)? Responda com um boolean (`True` ou `False`).

# In[5]:


#Subset de acordo com o solicitado de 3000 amostras da coluna height
sub_1 = get_sample(df, 'height', n = 3000)


# #### Aplicação do teste de normalidade de Shapiro-Wilk

# In[6]:


shap_stat, shap_pvalue = sct.shapiro(sub_1); shap_pvalue


# ##### Se Ho (hipótese nula) indica que a distribuição é normal, um valor p pequeno como o de 5.681722541339695e-07, indica que a probabilidade de rejeitar a hipótese nula de a distribuição ser normal, é muito pequena, logo neste caso, rejeita-se a hipótese nula

# In[7]:


shap_pvalue > 0.05


# In[8]:


def q1():
    return (shap_pvalue > 0.05)# Retorne aqui o resultado da questão 1.
    pass

q1()


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Plote o qq-plot para essa variável e a analise.
# * Existe algum nível de significância razoável que nos dê outro resultado no teste? (Não faça isso na prática. Isso é chamado _p-value hacking_, e não é legal).

# ##### Histograma com bins = 25
# A forma do gráfico lembra uma normal, mas o resultado do teste traz um valor de p-value pequeno, o que indica a rejeição de Ho, ou seja, a rejeijão da consideração de a distribuição ser considerada normal. Portanto a minha conclusão visual ao olhar o gráfico e o resultado do teste não são condizentes

# In[9]:


sns.distplot(sub_1, bins = 25)


# ##### Análise qqplot
# Ao análisar o gráfico qqplot, a distribuição aparenta ser uma normal, devido ao seu ajuste com a linha de inclicação de 45°, mas o teste de Shapiro-Wilk indica o contrário, que devemos rejeitar Ho, ou seja, rejeitar a hipótese de ser uma distribuição normal, e que a probabilidade de cometer erro ao rejeitar Ho é muito baixa

# In[10]:


import statsmodels.api as st
st.qqplot(sub_1, fit= True, line = '45')


# ##### P-value hacking:
# Com um nível de significância abaixo de 5.681722541339695e-07, pode-se dizer que distribuição é uma normal

# ## Questão 2
# 
# Repita o mesmo procedimento acima, mas agora utilizando o teste de normalidade de Jarque-Bera através da função `scipy.stats.jarque_bera()`. Agora podemos afirmar que as alturas são normalmente distribuídas (ao nível de significância de 5%)? Responda com um boolean (`True` ou `False`).

# In[11]:


jb_t, jb_pvalue = sct.jarque_bera(sub_1); jb_pvalue


# #### Pode-se rejeitar $H_{0}$
# Com o valor de 0.001478366424594868 para o *p-value*, pode-se rejeitar $H_{0}$ porque a probabilidade de erro será ainda muito pequena (e ainda inferior ao nível de significância de 5%)

# In[12]:


def q2():
    return (jb_pvalue > 0.05)
    # Retorne aqui o resultado da questão 2.
    pass

q2()


# __Para refletir__:
# 
# * Esse resultado faz sentido?

# <font size = '5' color = 'green'>Este resultado é qualitativamente igual ao resultado fornecido pelo teste de Shapiro-Wilk, diferindo apenas quantitativamente em relação ao valor p, portanto, faz sentido. </font>

# ## Questão 3
# 
# Considerando agora uma amostra de tamanho 3000 da coluna `weight` obtida com a função `get_sample()`. Faça o teste de normalidade de D'Agostino-Pearson utilizando a função `scipy.stats.normaltest()`. Podemos afirmar que os pesos vêm de uma distribuição normal ao nível de significância de 5%? Responda com um boolean (`True` ou `False`).

# In[13]:


sub_weight = get_sample(df, 'weight', n = 3000)


# In[14]:


ap_t, ap_pvalue = sct.normaltest(sub_weight); ap_pvalue


# In[15]:


def q3():
    return (ap_pvalue > 0.05)
    # Retorne aqui o resultado da questão 3.
    pass

q3()


# In[16]:


normalsim = sct.norm.rvs(loc = 10, scale = 3, size = 100000, random_state = 42)
sct.normaltest(normalsim)


# <font color = 'cian'> Neste caso da célula imediatamente acima desta eu não rejeitaria $H_{0}$, porque a probabilidade estar cometendo um erro seria de aproximadamente 85% ao rejeitar a hipótese nula que indica que a distribuição é normal </font>

# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Um _box plot_ também poderia ajudar a entender a resposta.

# In[17]:


#Histograma:
sns.distplot(sub_weight, bins = 25)


# #### A forma do gráfico e o resultado são condizentes
# O gráfico possui uma cauda à direita, de modo que não é uma distribuição simétrica, como seria a normal  
# Ademais, ao plotar o boxplot, observa-se que há outliers (de acordo com o critério utilizado por boxplots, claro) apenas para um lado da curva, o que indica uma distribuição assimétrica, que novamente, difere da normal

# In[18]:


sns.boxplot(sub_weight)


# ## Questão 4
# 
# Realize uma transformação logarítmica em na amostra de `weight` da questão 3 e repita o mesmo procedimento. Podemos afirmar a normalidade da variável transformada ao nível de significância de 5%? Responda com um boolean (`True` ou `False`).

# In[19]:


#Log na base 10 ou e? o np.log é base e
sub_weight_log = np.log(sub_weight)


# In[20]:


sns.distplot(sub_weight_log)


# In[21]:


#Normal test:
sub_weight_log_t, sub_weight_log_pvalue = sct.normaltest(sub_weight_log); sub_weight_log_pvalue


# In[22]:


def q4():
    return (sub_weight_log_pvalue > 0.05)
    # Retorne aqui o resultado da questão 4.
    pass

q4()


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Você esperava um resultado diferente agora?

# #### Parecia ser um log-normal, pela cauda à direita, mas aparentemnte o teste não indicou isso. 
# Link para explicaçãozinha rapida sobre distribuição log-normal: https://www.youtube.com/watch?v=K_96-AMmEqQ

# > __Para as questão 5 6 e 7 a seguir considere todos testes efetuados ao nível de significância de 5%__.

# ## Questão 5
# 
# Obtenha todos atletas brasileiros, norte-americanos e canadenses em `DataFrame`s chamados `bra`, `usa` e `can`,respectivamente. Realize um teste de hipóteses para comparação das médias das alturas (`height`) para amostras independentes e variâncias diferentes com a função `scipy.stats.ttest_ind()` entre `bra` e `usa`. Podemos afirmar que as médias são estatisticamente iguais? Responda com um boolean (`True` ou `False`).

# In[23]:


list(df)


# In[24]:


df['nationality'].unique()


# In[25]:


sub_nationalities = df[df.nationality.isin(['BRA','USA','CAN'])]
sub_bra = sub_nationalities[sub_nationalities['nationality'] == 'BRA']
sub_usa = sub_nationalities[sub_nationalities['nationality'] == 'USA']


# In[26]:


sub_bra['height'].isnull().value_counts()


# In[27]:


sub_usa['height'].isnull().value_counts()


# #### Número de graus de liberdade: $N_{1}$ + $N_{2}$ - 2 (Por t-student) por Welch's t-test a expressão é outra
# #### Alpha (nível de significância) = 0.05
# Além disso: equal_var. bool, optional
# If True (default), perform a standard independent 2 sample test that assumes equal population variances [1]. If False, perform Welch’s t-test, which does not assume equal population variance
# 
# #### Devemos então utilizar equal_varbool = False, para realizar o Welch's t-test, o qual considera variâncias diferentes para as populações

# In[28]:


sct.ttest_ind(a=sub_bra['height'], b=sub_usa['height'], equal_var = False)


# In[29]:


bra_usa_height_t, bra_usa_height_pvalue = sct.ttest_ind(a=sub_bra['height'], b=sub_usa['height'],
                                                        equal_var = False, nan_policy = 'omit'); bra_usa_height_pvalue


# In[30]:


def q5():
    return (bra_usa_height_pvalue >= 0.05)
    # Retorne aqui o resultado da questão 5.
    pass

q5()


# ## Questão 6
# 
# Repita o procedimento da questão 5, mas agora entre as alturas de `bra` e `can`. Podemos afimar agora que as médias são estatisticamente iguais? Reponda com um boolean (`True` ou `False`).

# In[31]:


sub_can = sub_nationalities[sub_nationalities['nationality']=='CAN']


# In[32]:


print("A média de altura dos canadenses é de:", sub_can['height'].mean(), "e a média de altura dos brasileiros é de:", sub_bra['height'].mean())


# In[33]:


bra_can_height_t, bra_can_height_pvalue = sct.ttest_ind(a=sub_bra['height'], b=sub_can['height'],
                                                        equal_var = False, nan_policy = 'omit'); bra_can_height_pvalue


# <font size = '5' color = 'teal'> Pode-se dizer que as médias são iguais, porque neste caso não rejeita-se $H_{0}$, dado que a probabilidade de erro ao rejeitá-la seria de 52,3%, a qual é muito superior ao nível de significância de 5 % definido previamente. </font>

# In[34]:


def q6():
    return (bra_can_height_pvalue >= 0.05)
    # Retorne aqui o resultado da questão 6.
    pass
q6()


# ## Questão 7
# 
# Repita o procedimento da questão 6, mas agora entre as alturas de `usa` e `can`. Qual o valor do p-valor retornado? Responda como um único escalar arredondado para oito casas decimais.

# In[35]:


print("A média e variância da altura dos canadenses é de:", sub_can['height'].mean(), " e", sub_can['height'].var(),
      "e a média e variância da altura dos americanos é de:", sub_usa['height'].mean(), " e ", sub_usa['height'].var())


# #### Visualização das distribuições por distplot:

# In[36]:


sns.distplot(sub_can['height'], color = 'skyblue', label = 'Canadá', bins = 30)
sns.distplot(sub_usa['height'], color = 'red', label = 'Estados Unidos', bins = 30)
plt.legend()


# In[37]:


usa_can_height_t, usa_can_height_pvalue = sct.ttest_ind(a=sub_usa['height'], b=sub_can['height'],
                                                        equal_var = False, nan_policy = 'omit'); a7 = round(usa_can_height_pvalue, 8)


# In[38]:


def q7():
    return a7
    # Retorne aqui o resultado da questão 7.
    pass

q7()


# __Para refletir__:
# 
# * O resultado faz sentido?
# * Você consegue interpretar esse p-valor?
# * Você consegue chegar a esse valor de p-valor a partir da variável de estatística?

# #### O resultado faz sentido porque as médias possuem valores distintos e variâncias pequenas ? (não tenho certeza)

# #### O p-valor representa a probabilidade de rejeitar-se $H_{0}$ quando o correto seria não rejeitá-la. Como o valor de 0.000466 é muito baixo, inclusive  
# #### inferior ao nível de significância de 5 %, a probabilidade de erro ao rejeitar-se $H_{0}$ é muito baixa, e neste caso, pode-se rejeitá-la. 

# ### Valor do teste estatístico:

# In[39]:


usa_can_height_t


# O teste t-student e o Welch's Test assumem a normalidade dos dados. Sabe-se também que o teste é bicaudal. Desta forma, o valor de 3.516987632488539 obtido, representa o desvio padrão para esquerda e para d direita. A partir destes limites, pode-se obter o valor das probabilidades
# 

# #### Créditos da função para graus de liberdade https://pythonfordatascience.org/welch-t-test-python-pandas/

# In[40]:


sub_usa_notnull = sub_usa['height'].notnull()
sub_can_notnull = sub_can['height'].notnull()


# In[41]:


def welch_dof(x,y):
        dof = (x.var()/x.size + y.var()/y.size)**2 / ((x.var()/x.size)**2 / (x.size-1) + (y.var()/y.size)**2 / (y.size-1))
        print(f"Welch-Satterthwaite Degrees of Freedom= {dof:.4f}")
        return dof
        
df = welch_dof(sub_usa_notnull, sub_can_notnull)


# In[42]:


2*sct.t.cdf(-usa_can_height_t, df = df)


# O valor de *p-value* não deu exatamente igual né, não sei se é assim que faz então 
