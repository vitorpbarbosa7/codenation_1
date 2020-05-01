import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import mode

dados = pd.read_csv('desafio1.csv')

#Há valores missing na pontuação crédito? 
dados.pontuacao_credito.isnull().sum()
#Então não há nenhum valor missing

#Vamos obter os valores por estado então. 
SC = pd.DataFrame(dados[dados['estado_residencia'] == "SC"])
RS = dados[dados['estado_residencia'] == 'RS']
PR = dados[dados['estado_residencia'] == 'PR']

def estatisticas(serie):
    lista = [serie.mode(),
             serie.mean(),
             serie.median(),
             serie.std()]
    return lista

SC_list = estatisticas(SC.pontuacao_credito)
RS_list = estatisticas(RS.pontuacao_credito)
PR_list = estatisticas(PR.pontuacao_credito)    

print(SC_list)
print(RS_list)
print(PR_list)

plt.hist(RS.pontuacao_credito)
mode(SC.pontuacao_credito)

pontuacao_credito = dados.groupby('estado_residencia')['pontuacao_credito'].mean()
