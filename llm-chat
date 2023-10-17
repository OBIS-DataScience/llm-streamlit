

import streamlit as st
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
import dropbox
import tempfile
import os
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import FewShotPromptTemplate
from langchain import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings

# Initialize Streamlit page
st.set_page_config(page_title='🦜🔗 Ask the Doc App')
st.title('🦜🔗 Ask the Doc App')

# Dropbox access token input field
dropbox_access_token = st.text_input("Enter your Dropbox Access Token:")

if dropbox_access_token:
    dbx = dropbox.Dropbox(dropbox_access_token)
    
    def generate_response(openai_api_key, query_text):
        # Split documents into chunks
        temp_dir = tempfile.mkdtemp()
        # Download files with allowed extensions to a temporary directory
        allowed_extensions = ['.pdf', '.xlsx', '.csv', '.docx', '.txt']
        for entry in dbx.files_list_folder('/test-llm').entries:
            if isinstance(entry, dropbox.files.FileMetadata):
                file_extension = os.path.splitext(entry.name)[-1].lower()
                if file_extension in allowed_extensions:
                    file_path = os.path.join(temp_dir, entry.name)
                    dbx.files_download_to_file(file_path, entry.path_lower)
        
        os.environ['OPENAI_API_KEY'] = openai_api_key
        
        loader_txt = DirectoryLoader(
            temp_dir,  # my local directory
            glob='**/*.txt'  # include pdf, csv, and xlsx files
        )
        
        loader_pdf = DirectoryLoader(
            temp_dir,  # my local directory
            glob='**/*.pdf'  # include pdf, csv, and xlsx files
        )
        
        loader_csv = DirectoryLoader(
            temp_dir,  # my local directory
            glob='**/*.csv'  # include pdf, csv, and xlsx files
        )
        
        loader_docx = DirectoryLoader(
            temp_dir,  # my local directory
            glob='**/*.docx'  # include pdf, csv, and xlsx files
        )
        
        loader_pptx = DirectoryLoader(
            temp_dir,  # my local directory
            glob='**/*.pptx'  # include pdf, csv, and xlsx files
        )
        
        docs = loader_txt.load() + loader_pdf.load() + loader_csv.load() + loader_docx.load() + loader_pptx.load()
        
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=0
        )
        docs_split = text_splitter.split_documents(docs)
        faiss_index = FAISS.from_documents(docs_split, OpenAIEmbeddings(model="text-embedding-ada-002"))
        retriever = faiss_index.as_retriever()
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True,
            output_key='answer')
        
        # Prompt
        tech_template = """As a support bot, your goal is to provide accurate
        and helpful information. 
        Keep the response complete. Keep response within 200 words.
        Always respond in English words. Do not show any other language.
        {context}

        Q: {question}
        A: """
        
        PROMPT = PromptTemplate(
            template=tech_template, input_variables=["context", "question"]
        )
        
        chain = ConversationalRetrievalChain.from_llm(
            llm=OpenAI(temperature=0,model='text-davinci-003'),
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": PROMPT}
        )
        
        return chain.run(query_text)

    query_text = st.text_input('Enter your question:',
                              placeholder='Ask me anything from the documents provided in Dropbox.')

    # Form input and query
    result = []
    with st.form('myform', clear_on_submit=True):
        openai_api_key = st.text_input('OpenAI API Key', type='password')
        submitted = st.form_submit_button('Submit')
        if submitted and openai_api_key.startswith('sk-'):
            with st.spinner('Retrieving response...'):
                response = generate_response(openai_api_key, query_text)
                result.append(response)
                del openai_api_key

    if len(result):
        st.info(response)
