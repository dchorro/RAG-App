import sys
from vector_db_manager import load_vector_db, get_embedding_function, create_vector_db, split_documents, load_directory
from llm_interaction import ask, load_model
import textwrap

def print_wrapped(text, wrap_length=80):
    wrapped_text = textwrap.fill(text, wrap_length)
    print(wrapped_text)

def main():
    DIRECTORY_PATH = "transcriptionsBorjaBandera"
    vector_db_path = 'vector_db.parquet'
    doc = load_directory(DIRECTORY_PATH)
    splitted_doc = split_documents(doc)
    df = create_vector_db(splitted_doc=splitted_doc, save_path=vector_db_path)
    embeddings = load_vector_db(vector_db_path)
    embedding_function = get_embedding_function()
    
    # Load model and tokenizer
    llm_model, tokenizer = load_model()
    print("Welcome! Type 'exit' to quit.")
    while True:
        query = input("Enter your query: ")
        if query.lower() == 'exit':
            break
        # Ask the LLM
        response = ask(query=query, embeddings=embeddings, embedding_function=embedding_function, llm_model=llm_model, tokenizer=tokenizer, df=df)
        print()
        print_wrapped(response)
        print()
        print()

if __name__ == "__main__":
    main()
