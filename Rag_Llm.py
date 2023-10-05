#%% Import necessary libraries and set up GPU environment
import os
from torch import cuda, bfloat16
import transformers

#%% Import necessary libraries and set up GPU environment and Huggingface token for model access
os.environ["CUDA_VISIBLE_DEVICES"] = "GPUNUMBER"
model_id = 'mistralai/Mistral-7B-Instruct-v0.1'
token = "TOKEN"

# Configure 4-bit quantization for the model
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,  # Load model weights in 4-bit precision
    bnb_4bit_quant_type='nf4',  # Use nf4 quantization type
    bnb_4bit_use_double_quant=False,  # Disable nested quantization
    bnb_4bit_compute_dtype=bfloat16
)

# Initialize the tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

# Initialize the model with quantization config
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map='auto',
)
model.eval()
#%% Initialize a text generation pipeline
generator = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    task='text-generation',
    temperature=0.0,
    max_new_tokens=256,
    repetition_penalty=1.1
)
#%% Load data from a CSV file
from langchain.document_loaders.csv_loader import CSVLoader
source = "Source CSV"
abstract_loader = CSVLoader(file_path=source, source_column='Abstract', encoding='ISO-8859-1')
abstract_data = abstract_loader.load()
#%% Split the abstracts into smaller chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
)
abstract_documents = text_splitter.transform_documents(abstract_data)
#%% Initialize embeddings and vector store
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.embeddings.cache import CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore

store = LocalFileStore("./cache/")
embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'
embedding_model = HuggingFaceEmbeddings(model_name=embed_model_id)
embedder = CacheBackedEmbeddings.from_bytes_store(embedding_model, store, namespace=embed_model_id)
vector_store = FAISS.from_documents(abstract_documents, embedder)
#%%  Initialize LangChain components
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.callbacks import StdOutCallbackHandler

llm = HuggingFacePipeline(pipeline=generator)
retriever = vector_store.as_retriever()
handler = StdOutCallbackHandler()
#%% Create a RetrievalQA pipeline
rag = RetrievalQA.from_chain_type(
    llm=llm, chain_type='stuff',
    retriever=retriever,
    callbacks=[handler],
    return_source_documents=True
)
#%% Perform text generation with llm
print(llm('Vaccines cause autism.'))
#%% Perform retrieval-based QA with LangChain
rag('Vaccines cause autism.')

# %%
