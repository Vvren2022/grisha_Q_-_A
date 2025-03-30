# -*- coding: utf-8 -*-

from langchain.vectorstores import FAISS
import tf_keras as keras
import pandas as pd
# from langchain_openai import ChatOpenAI
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
# from langchain_community.vectorstores import FAISS
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
# from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')

# Load environment variables from .env file


os.environ["OPENAI_API_KEY"]=grish_key


# Initialize the OpenAI LLM
llm = ChatOpenAI(temperature=0.1)

# Initialize instructor embeddings using the Hugging Face model
# instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
openai_embeddings = OpenAIEmbeddings()
vectordb_file_path = "faiss_index"

def create_vector_db():
    # Load data from FAQ sheet


    # df = pd.read_csv('Grisha_faqs.csv', encoding="ISO-8859-1")  # Read with alternative encoding
    # df.to_csv("cleaned_faqs.csv", index=False, encoding="utf-8")  # Save in UTF-8
    #
    # # Now load with LangChain
    # loader = CSVLoader(file_path="cleaned_faqs.csv")
    # data = loader.load()

    loader = CSVLoader(file_path='Grisha_faqs.csv', source_column="prompt",encoding='cp1252')
    data = loader.load()

    # Create a FAISS instance for vector database from 'data'
    vectordb = FAISS.from_documents(documents=data, embedding=openai_embeddings)

    # Save vector database locally
    vectordb.save_local(vectordb_file_path)

def get_qa_chain():
    # Load the vector database from the local folder
    vectordb = FAISS.load_local(vectordb_file_path, openai_embeddings,allow_dangerous_deserialization=True)

    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer, try to provide as much text as possible from the "response" section in the source document context without making many changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": PROMPT})

    return chain

if __name__ == "__main__":
    create_vector_db()
    chain = get_qa_chain()
    # print(chain("Do you have a JavaScript course?"))

