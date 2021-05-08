import streamlit as st
import pandas as pd

def main():
    # st.title('Hello World')
    st.markdown('Botão')
    botao = st.button('Botão')
    #Ação de clicar no botão:
    if botao:
        st.markdown('Botão Clicado')
        
    
    check = st.checkbox('Checkbox')
    if check:
        st.markdown('Checkbox clicado')
    
    #Lista de opções:
    radio = st.radio('Escolha as opções',
                     ('Opt 1','Opt 2'))
    if radio == 'Opt 1':
        st.markdown('Opt 1')
    if radio == 'Opt 2':
        st.markdown('Opt 2')
        
    select = st.selectbox('Choose opt 1 ',
                          ('Opt 1 ', 'Opt 2 '))
    if select == 'Opt 1':
        st.markdown('Opt 1 ')
    if select == 'Opt 2':
        st.markdown('Opt 2 ')
        
        
    multi= st.multiselect('Choose :',
                          ('Opt 1','Opt 2'))
    
    st.write('You selected:', multi)
    
    
    file = st.file_uploader('Escolha seu arquivo csv', 
                     type = 'csv')
    if file is not None:
        st.markdown('Arquivo carregado')
    

if __name__ == '__main__':
    main()
