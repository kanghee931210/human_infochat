import streamlit as st

# LLM Model

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

import asyncio

# PDF preprocess
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Embedding Model
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.vectorstores import FAISS
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
# Memory buffer
from langchain.memory import ConversationBufferMemory
from langchain.memory import StreamlitChatMessageHistory

# Vector DB

from langchain.vectorstores import FAISS
from langchain.chat_models import ChatVertexAI
from langchain.embeddings import VertexAIEmbeddings

from langchain.callbacks import get_openai_callback

import yaml
import numpy as np
# from loguru import logger
import pandas as pd
import os

from langchain.callbacks import get_openai_callback



def main():
    st.set_page_config(
        page_title = 'Hist rag chat make_kh',
        page_icon = ':flag-kh:'
    )
    with st.sidebar:
        uploaded_files = st.file_uploader("CSV file을 업로드 하세요", type=['csv'], accept_multiple_files=False)
        GCP_files = st.file_uploader("Key file을 업로드 하세요", type=['json'], accept_multiple_files=False)
        process = st.button("실행")

    st.title(":blue[Hist] _Chat_ :robot_face:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    # if 'checker' not in st.session_state:
    #     st.session_state.checker = None
    # if 'emb' not in st.session_state:
    #     st.session_state.emb = None
    # start
    if process:
        df = pd.read_csv(uploaded_files, index_col=0)
        with open('GCP.json', mode='wb') as w:
            w.write(GCP_files.getvalue())
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'GCP.json'
        files_text = get_text(df)
        vetorestore = get_vectorstore(files_text)
        st.session_state.conversation = conversation_chat(vetorestore)

        st.session_state.processComplete = True

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role" : "assistant",
                                         "content" : "안녕하세요 궁금한 점이 있으신가요?"}]

    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    history = StreamlitChatMessageHistory(key='chat_messages')

    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content" : query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            chain = st.session_state.conversation
            response = chain(query)['answer']
            st.markdown(response)
            # with st.spinner("답을 찾는 중입니다..."):
            #     result = chain(query)
            #     with get_openai_callback() as cb:
            #         st.session_state.chat_history = result['chat_history']
            #     print(st.session_state.chat_history)
            #     response = result['answer']
            #     source_documents = result['source_documents']
            #
            #     print(source_documents)
            #     st.markdown(response)
                # with st.expander("참고 문서 확인"):
                    # st.markdown(source_documents[0].metadata['source'], help=source_documents[0].page_content)
                    # st.markdown(source_documents[1].metadata['source'], help=source_documents[1].page_content)
                    # st.markdown(source_documents[2].metadata['source'], help=source_documents[2].page_content)
        # st.session_state.messages.append({'role' : "assistant", 'content' : result})

def get_text(df):
    # df = pd.read_csv('result2.csv', index_col=0)
    text = []
    for i in range(len(df)):
        New_data = zip(df.columns, df.iloc[i])
        text.append(' '.join([f"{key} : {value}" for key, value in New_data]))
    return text

def get_vectorstore(text):
    embedding = VertexAIEmbeddings(model_name='textembedding-gecko-multilingual@latest')
    vectordb = FAISS.from_texts(text, embedding)
    return vectordb

def conversation_chat(vectordb):
    llm = ChatVertexAI(
        model_name="chat-bison@001",
        max_output_tokens=1024,
        temperature=0.2,
        top_p=0.8,
        top_k=40)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        # combine_docs_chain_kwargs={"prompt": template},
        chain_type='stuff',
        retriever=vectordb.as_retriever(vervose=True, k=3),
        # retriever=vectordb.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.01,
        #                                                                                          'k': 3}),
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
        # anwser는 답변만 답겠다.
        get_chat_history=lambda h: h,
        return_source_documents=True,
        # verbose=True
    )
    return conversation_chain
if __name__ == '__main__':
    main()
