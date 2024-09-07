import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from vector_db_manager import get_embedding_function, split_documents
import time

def retrieve_relevant_resources(query, embeddings, embedding_model, top_k=5, print_time=True):
    embedded_query = torch.tensor(embedding_model.embed_query(query))
    start = time.time()
    cosine_similarity_results = F.cosine_similarity(embedded_query, embeddings, dim=1)
    end = time.time()
    
    if print_time:
        print(f"[INFO] Time taken to get scores on {len(embeddings)} embeddings: {end-start:.5f} seconds")
    
    values, indices = torch.topk(cosine_similarity_results, k=top_k)
    return values, indices

def load_model(model_id="UCLA-AGI/Gemma-2-9B-It-SPPO-Iter3"):
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id, return_token_type_ids=False)
    llm_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_id, torch_dtype=torch.float16, 
                                                     quantization_config=quantization_config, low_cpu_mem_usage=True, 
                                                     device_map="auto", use_safetensors=True)
    return llm_model, tokenizer

def prompt_formatter(query, context_items):
    context = "- " + "\n- ".join([item for item in context_items])
    base_prompt = f"""Eres un asistente parte de una aplicación RAG que habla español. 
    Aquí hay elementos de contexto relevantes para la consulta: 
    {context} 
    \nPregunta del usuario: {query}\nRespuesta:"""
    
    return base_prompt

def ask(query, embeddings, embedding_function, llm_model, tokenizer, df, top_k=5, temperature=0.7, max_new_tokens=256):
    scores, indices = retrieve_relevant_resources(query=query, embeddings=embeddings, embedding_model=embedding_function, top_k=top_k)
    context_items = [df.iloc[int(i)]["page_content"] for i in indices.tolist()]
    # context_items = [doc["page_content"] for doc in indices]
    
    prompt = prompt_formatter(query=query, context_items=context_items)
    
    input_ids = tokenizer(prompt, return_token_type_ids=False, return_tensors="pt").to("cuda")
    outputs = llm_model.generate(**input_ids, temperature=temperature, do_sample=True, max_new_tokens=max_new_tokens)
    output_text = tokenizer.decode(outputs[0]).strip()
    # Replace prompt and special tokens
    output_text = output_text.replace(prompt, "").replace("<bos>", "").replace("<eos>", "").replace("</s>", '').replace('<end_of_turn>', '')
    output_text = output_text.strip()
    
    return output_text