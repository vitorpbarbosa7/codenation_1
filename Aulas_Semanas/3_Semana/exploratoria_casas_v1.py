import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from plotnine import *
import seaborn as sea

dados = pd.read_csv('houses_to_rent_v2.csv')

dados.rename(columns = {"rent amount (R$)" : 'valor_aluguel'}, inplace=True)

dados.info()

#%%Análises gerais:
corr_geral = dados.corr(method = 'spearman')
sea.heatmap(corr_geral, cmap='coolwarm', annot=True)

#%%QUal média de aluguel mais alta:

#Mas há valores missing?
dados.isnull().sum(axis = 0)
#Não tem valores missing, mas isso é muito estranho

#listinha:
dados.groupby('city')['valor_aluguel'].median().reset_index().sort_values('valor_aluguel', ascending = False)
    
#Plot do boxplot, ou seja, visualizar a distribuição das variáveis de valor de aluguel por cidade. 
(ggplot(dados) + aes(x = 'city', y = 'valor_aluguel')) + geom_boxplot() + theme_bw()

dados.groupby('city')['valor_aluguel'].median()

#E quantos dados há de cada um? 
dados.city.value_counts()
#%% Banheiros e valor do aluguel
dados.valor_aluguel.describe()

#Definição, alugueis 
sub_banheiros_caros = dados[dados['valor_aluguel'] > 5000]

sub_banheiros_caros.bathroom.describe()

#ggplot em python é o crime 
(ggplot(dados) + aes(x = 'valor_aluguel', y = 'bathroom') + geom_point()) 

#Maneira sinistra de fazer:
dados['aluguel_alto'] = ['Alto' if x > 5000 else 'Baixo' for x in dados['valor_aluguel']]

dados.groupby('aluguel_alto')['bathroom'].mean()

#%% Groupby por porcentagem, tô besta, deu certo 
fur = dados.groupby('aluguel_alto')['furniture'].value_counts().groupby(level=0).apply(lambda x: 100*x/float(x.sum()))
