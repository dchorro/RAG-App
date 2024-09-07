import torch
import time
import pandas as pd
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import DirectoryLoader
import nltk
nltk.download('punkt')
import langchain
import torch
import numpy as np
import torch.nn.functional as F
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import is_flash_attn_2_available
from transformers import BitsAndBytesConfig




def get_embedding_function(model_name="BAAI/bge-m3"):
    model_kwargs = {'device': 'cuda', 'trust_remote_code': True}
    encode_kwargs = {'normalize_embeddings': False}

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

    return embeddings

embedding_function = get_embedding_function()

def load_documents(data_path):
    document_loader = TextLoader(data_path)
    return document_loader.load()

def load_directory(directory_path):
    # directory_loader = DirectoryLoader(data_path)
    directory_loader = DirectoryLoader(directory_path, glob="**/*.txt")
    return directory_loader.load()

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 80,
        length_function=len,
        is_separator_regex=False
    )
    return text_splitter.split_documents(documents)

def load_model(model_id="UCLA-AGI/Gemma-2-9B-It-SPPO-Iter3", 
               use_quantization_config=True, 
               load_in_4bit=True, 
               bnb_4bit_compute_dtype=torch.float16,
               device="cuda"):
    
    quantization_config = BitsAndBytesConfig(load_in_4bit=load_in_4bit,
                                            bnb_4bit_compute_dtype=bnb_4bit_compute_dtype)
    use_quantization_config = True

    # Bonus: flash attention 2 = faster attention mechanism
    # Flash Attention 2 requires a GPU with a compute capability score of 8.0+ (Ampere, Ada Lovelace, Hopper and above): https://developer.nvidia.com/cuda-gpus
    if (is_flash_attn_2_available()) and (torch.cuda.get_device_capability(0)[0] >= 8):
        attn_implementation = "flash_attention_2"
    else:
        attn_implementation = "sdpa" # scaled dot product attention
    print(f"Using attention implementation: {attn_implementation}")

    # 2. Pick a model we'd like to use
    # model_id = "nvidia/Nemotron-4-Minitron-4B-Base"
    # model_id = "UCLA-AGI/Gemma-2-9B-It-SPPO-Iter3"

    # 3. Instantiate tokenizer (tokenizer turns text into tokens)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id, return_token_type_ids=False)

    # 4. Instantiate the model
    llm_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_id,
                                                    torch_dtype=torch.float16,
                                                    quantization_config=quantization_config if use_quantization_config else None,
                                                    #  low_cpu_mem_usage=False, # use as much memory as we can
                                                    low_cpu_mem_usage=True,
                                                    device_map="auto",
                                                    use_safetensors=True,
                                                    attn_implementation=attn_implementation)

    if not use_quantization_config:
        llm_model.to(device)


def retrieve_relevant_resources(query, embeddings, embedding_model, top_k=5, print_time=True):

    embedded_query = torch.tensor(embedding_model.embed_query(query))

    start = time.time()
    cosine_similarity_results = F.cosine_similarity(embedded_query, embeddings, dim=1)
    end = time.time()

    if print_time:
        print(f"[INFO] Time taken to get scores on {len(embeddings)} embeddings: {end-start:.5f} seconds")
    values, indices = torch.topk(cosine_similarity_results, k=top_k)

    return values, indices


def prompt_formatter(query: str,
                     context_items: list[dict]) -> str:
    context = "- " + "\n- ".join([item for item in context_items])

    base_prompt = """Eres un asistente parte de una aplicación RAG que habla español.
    Con base en los siguientes elementos de contexto, responde la consulta.
    Date tiempo para pensar extrayendo pasajes relevantes del contexto antes de responder la consulta.
    Ten en cuenta únicamente el contexto que se te proporciona.
    No devuelvas el razonamiento, solo devuelve la respuesta.
    Asegúrate de que tus respuestas sean lo más explicativas posible.
    Utilice los siguientes ejemplos como referencia para el estilo de respuesta ideal.
    Tu texto no debería de estar cogido literalmente del contexto, sino parafraseado para que quede mejor redactado, eliminando las palabras que no aportan ningún significado como "emm, eh" y expresiones del estilo que se utilizan al hablar.
    \nEjemplo 1:
    Consulta: ¿Cuáles son las vitaminas liposolubles?
    Respuesta: Las vitaminas liposolubles incluyen la vitamina A, la vitamina D, la vitamina E y la vitamina K. Estas vitaminas se absorben junto con las grasas en la dieta y se pueden almacenar en el tejido graso del cuerpo y en el hígado para su uso posterior. La vitamina A es importante para la visión, la función inmunológica y la salud de la piel. La vitamina D desempeña un papel fundamental en la absorción de calcio y la salud ósea. La vitamina E actúa como antioxidante, protegiendo a las células del daño. La vitamina K es esencial para la coagulación sanguínea y el metabolismo óseo.
    \nEjemplo 2:
    Consulta: ¿Cuáles son las causas de la diabetes tipo 2?
    Respuesta: La diabetes tipo 2 suele estar asociada a una sobrealimentación, en particular al consumo excesivo de calorías que conduce a la obesidad. Los factores incluyen una dieta alta en azúcares refinados y grasas saturadas, que pueden provocar resistencia a la insulina, una afección en la que las células del cuerpo no responden de manera eficaz a la insulina. Con el tiempo, el páncreas no puede producir suficiente insulina para controlar los niveles de azúcar en sangre, lo que da lugar a la diabetes tipo 2. Además, la ingesta calórica excesiva sin suficiente actividad física exacerba el riesgo al promover el aumento de peso y la acumulación de grasa, en particular alrededor del abdomen, lo que contribuye aún más a la resistencia a la insulina.
    \nEjemplo 3:
    Consulta: ¿Cuál es la importancia de la hidratación para el rendimiento físico?
    Respuesta: La hidratación es crucial para el rendimiento físico porque el agua desempeña un papel fundamental en el mantenimiento del volumen sanguíneo, la regulación de la temperatura corporal y la garantía del transporte de nutrientes y oxígeno a las células. Una hidratación adecuada es esencial para el funcionamiento óptimo de los músculos, la resistencia y la recuperación. La deshidratación puede provocar una disminución del rendimiento, fatiga y un mayor riesgo de enfermedades relacionadas con el calor, como el golpe de calor. Beber suficiente agua antes, durante y después del ejercicio ayuda a garantizar un rendimiento físico y una recuperación óptimos. \nAhora use los siguientes elementos de contexto para responder la consulta del usuario:
    {context}
    \nPasajes relevantes: <extraiga aquí los pasajes relevantes del contexto>
    Consulta del usuario: {query}
    Respuesta:
    """

    base_prompt = base_prompt.format(context=context,
                                     query=query)

    # Create prompt template for instruction-tuned model
    dialogue_template = [
        {"role": "user",
         "content": base_prompt}
    ]

    # Apply the chat template
    # prompt = tokenizer.apply_chat_template(conversation=dialogue_template,
    #                                        tokenize=False,
    #                                        add_generation_prompt=True)

    return base_prompt

# query = random.choice(query_list)
query = "Indica las ventajas y desventajas de un dólar fuerte"
print(f"Query: {query}")

# Get relevant resources
scores, indices = retrieve_relevant_resources(query=query,
                                              embeddings=embeddings,
                                              embedding_model=get_embedding_function())

for score, idx in zip(scores, indices):
    print(f"Score: {score:.4f}")
    print(f"Index: {idx}")
# Create a list of context items
context_items = [df.iloc[int(i)]["page_content"] for i in indices.tolist()]

# Format our prompt
prompt = prompt_formatter(query=query,context_items=context_items)
print(prompt)

def ask(query: str,
        temperature: float=0.7,
        max_new_tokens:int=256,
        top_k: int=5,
        format_answer_text=True,
        return_answer_only=True):
    """
    Takes a query, finds relevant resources/context and generates an answer to the query based on the relevant resources.
    """

    # RETRIEVAL
    # Get just the scores and indices of top related results
    scores, indices = retrieve_relevant_resources(query=query,
                                                  embeddings=embeddings,
                                                  embedding_model=embedding_function,
                                                  top_k=top_k)

    # Create a list of context items
    context_items = [df.iloc[int(i)]["page_content"] for i in indices.tolist()]

    # Add score to context item
    # for i, item in enumerate(context_items):
    #     print(item)
    #     item["score"] = scores[i].cpu()

    # AUGMENTATION
    # Create the prompt and format it with context items
    prompt = prompt_formatter(query=query,
                              context_items=context_items)

    # GENERATION
    # Tokenize the prompt
    input_ids = tokenizer(prompt, return_token_type_ids=False, return_tensors="pt").to("cuda")

    # Generate an output of tokens
    outputs = llm_model.generate(**input_ids,
                                 temperature=temperature,
                                 do_sample=True,
                                 max_new_tokens=max_new_tokens)

    # Decode the tokens into text
    output_text = tokenizer.decode(outputs[0])

    # Format the answer
    if format_answer_text:
        # Replace prompt and special tokens
        output_text = output_text.replace(prompt, "").replace("<bos>", "").replace("<eos>", "").replace("</s>", '').replace('<end_of_turn>', '')
        output_text = output_text.strip()

    # Only return the answer without context items
    if return_answer_only:
        return output_text

    return output_text, context_items