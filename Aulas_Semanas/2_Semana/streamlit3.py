import streamlit as st
import pandas as pd

def main():
    st.title('API de informações iniciais sobre dataframes')
    st.header('Importação da base de dados:')
    # file = st.file_uploader('Escolha seu arquivo csv',type = 'csv')
    # if file is not None:
    # df = pd.read_csv(file)
    df = pd.read_csv('IRIS.CSV')
    
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


    
    
if __name__ == '__main__':
    main()
