import streamlit as st
import pandas as pd

def main():
    st.title('API de informações iniciais sobre dataframes')
    st.header('Importação da base de dados:')
    # file = st.file_uploader('Escolha seu arquivo csv',type = 'csv')
    # if file is not None:
    # df = pd.read_csv(file)
    df = pd.read_csv('IRIS.csv')
    
    #Visualização inicial dos dados
    linhas = st.slider(label = 'Escolha o número de linhas iniciais', 
                       min_value = 0,
                       max_value = 100,
                       value = 5)
    
    st.dataframe(df.head(linhas))    

    #Número de linhas e de colunas:
    st.markdown('**Número de linhas:**')
    st.markdown(df.shape[0])
    st.markdown('**Número de colunas:**')
    st.markdown(df.shape[1])
    
    st.markdown('Outra maneira de visualização')
    st.table(df.head(linhas))

    # Estatísticas básicas:
    # Seleção de colunas que não são objeto, portanto são numéricas:
    tipos = pd.DataFrame({"colunas": df.columns, "tipos": df.dtypes})
    numericas = list(tipos[tipos['tipos'] != 'object']['colunas'])

    colunas = st.selectbox('Selecione a feature que deseja analisar:', numericas)
    if colunas is not None:
    	st.markdown('Seleciona qual estatística básica deseja obter:')
    	mean = st.checkbox('Mediana')
    	if mean:
        	st.markdown(df[colunas].mean())


    
    
if __name__ == '__main__':
    main()
