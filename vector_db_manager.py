import os
import torch
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
import nltk
nltk.download('punkt')
import pandas as pd
import numpy as np

def get_embedding_function(model_name="BAAI/bge-m3"):
    model_kwargs = {'device': 'cuda', 'trust_remote_code': True}
    encode_kwargs = {'normalize_embeddings': False}

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    return embeddings

def load_documents(data_path):
    document_loader = TextLoader(data_path)
    return document_loader.load()

def load_directory(directory_path):
    directory_loader = DirectoryLoader(directory_path, glob="**/*.txt")
    return directory_loader.load()

def split_documents(documents, chunk_size=800, chunk_overlap=80):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False
    )
    return text_splitter.split_documents(documents)

def create_vector_db(splitted_doc, save_path, embedding_function=get_embedding_function()):
    page_content = []
    source = []
    for idx, prueba in enumerate(splitted_doc):
        temp = prueba.dict()
        p_cont = temp["page_content"]
        page_content.append(p_cont)
        source.append(temp["metadata"]["source"])

    embeddings = []
    for chunk in page_content:
        embeddings.append(embedding_function.embed_query(chunk))

    df = pd.DataFrame.from_dict({"page_content": page_content, "source": source, "embeddings": embeddings})
    save_vector_db(df, save_path)
    return df

def save_vector_db(vector_db, save_path):
    vector_db.to_parquet(save_path)

def load_vector_db(save_path):
    df = pd.read_parquet(save_path)
    embeddings = torch.tensor(np.array(df['embeddings'].tolist(), dtype=np.float32))
    return embeddings
