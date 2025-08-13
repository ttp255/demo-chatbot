from langchain_google_genai import GoogleGenerativeAI
from langchain_chroma import Chroma 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.prompts import ChatPromptTemplate
import os
import shutil
from dotenv import load_dotenv


load_dotenv()
google_api_key=os.getenv('GOOGLE_GEN_AI_KEY')
api_huggingface=os.getenv('API_HUGGINGFACE')
path_db='./chroma'
embeddings=HuggingFaceEndpointEmbeddings(
        model='sentence-transformers/all-MiniLM-L6-v2',




        huggingfacehub_api_token=api_huggingface,

    )
def load_data(file_path):
  loader=PyPDFLoader(file_path=file_path)

  docs=loader.load()

  return docs



def split_document(docs):
  text_splitter=RecursiveCharacterTextSplitter(
      chunk_size=1000,
      chunk_overlap=500,
      length_function=len,
      add_start_index=True,

    

  )
  return text_splitter.split_documents(docs)



def save_db(chunks,embeddings=embeddings,path_db=path_db):
    if os.path.exists(path_db):
        shutil.rmtree(path_db)
    os.mkdir(path_db)

    vector_store=Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=path_db


    )









def response(model:GoogleGenerativeAI,retriever,question):

  template="""
    You are the best in .....
    Here are some relevant reviews: {reviews}
    Here are some relevant answer: {question}

  """
  prompt=ChatPromptTemplate.from_template(template)

  chat_bot=prompt|model
  reviews=retriever.invoke(question)
  result=chat_bot.invoke({'reviews':reviews,'question':question})
  return result





