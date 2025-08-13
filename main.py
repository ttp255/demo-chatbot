import streamlit as st
from langchain_core.messages import HumanMessage,AIMessage
from rag import *

def  main():


    st.set_page_config(page_title='Chat bot',page_icon='🤖')

    with st.sidebar:
        st.subheader('Drop your document')
        pdf_file=st.file_uploader(
            'Upload your file in here',
            accept_multiple_files=False
        )
       
        if st.button('Process'):
            with st.spinner('Processing'):
                file_path='./file/'+pdf_file.name
                if os.path.exists(file_path):
                    os.remove(file_path)
                with open(file_path,'wb') as f:
                     f.write(pdf_file.read())

                
             
      
                raw_text=load_data(file_path=file_path)
                chunks=split_document(raw_text)
                save_db(chunks)
                st.write('Load success!')
        
             
               


    if 'chat_history' not in st.session_state:
        st.session_state.chat_history=[]
   
    


    st.title('Chat bot 🤖')

    user_query=st.chat_input('Your message')

    for message in st.session_state.chat_history:
        if isinstance(message,HumanMessage):
            with st.chat_message('Human'):
                st.markdown(message.content)
        else:
            with st.chat_message('AI'):
                st.markdown(message.content)


    if user_query is not None and user_query!='':
        st.session_state.chat_history.append(HumanMessage(user_query))

        with st.chat_message('Human'):
            st.markdown(user_query)

        with st.chat_message('AI'):
            with st.spinner('......'):
                model=GoogleGenerativeAI(model='gemini-2.5-flash',api_key=google_api_key)
                vector_store=Chroma(
                    persist_directory=path_db,
                    embedding_function=embeddings
                )
                retriever=vector_store.as_retriever(search_kwargs={'k':3})
                ai_response=response(model,retriever,question=user_query)
                st.markdown(ai_response)
        st.session_state.chat_history.append(AIMessage(ai_response))


if __name__=='__main__':

    main()