import os
import openai
from dotenv import load_dotenv
from llama_index.core import GPTVectorStoreIndex, SimpleDirectoryReader, VectorStoreIndex, StorageContext, ServiceContext
from llama_index.core import PromptHelper
# from langchain_community.llms.openai import AzureOpenAI
from langchain_openai.llms.azure import AzureOpenAI
# from langchain.embeddings import OpenAIEmbeddings
# from llama_index.embeddings import LangchainEmbedding
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore



# Load env variables (create .env with OPENAI_API_KEY and OPENAI_API_BASE)
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')
pinecone_index = os.environ.get("PINECONE_API_INDEX")
cohere_api_key = os.environ["COHERE_API_KEY"]

# Configure OpenAI API
api_type = "azure"
api_version = "2023-07-01-preview"
api_base = os.getenv('AZURE_OPENAI_ENDPOINT')
api_key = os.getenv("OPENAI_API_KEY")

# Create LLM via Azure OpenAI Service
llm = AzureOpenAI(
    model="OpenAIGPT35Turbo",
    # deployment_name="my-custom-llm",
    api_key=api_key,
    azure_endpoint=api_base,
    api_version=api_version,
)

embed_model = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name="OpenAIEmbeddings",
    api_key=api_key,
    azure_endpoint=api_base,
    api_version=api_version,
)

from llama_index.core import Settings
Settings.llm = llm
Settings.embed_model = embed_model

pinecone_vector_store = PineconeVectorStore(api_key=PINECONE_API_KEY, index_name=pinecone_index, environment=PINECONE_API_ENV, namespace="alice_1")
vector_index = VectorStoreIndex.from_vector_store(vector_store=pinecone_vector_store)

# query = "What are the mx480 system configurations?"
# query_engine = vector_index.as_query_engine()
# answer = query_engine.query(query)

# print(answer.get_formatted_sources())
# print("query was:", query)
# print("answer was:", answer)

# Define prompt helper
max_input_size = 3000
num_output = 256
chunk_size_limit = 1000 # token window size per document
max_chunk_overlap = 0.5 # overlap for each token fragment
prompt_helper = PromptHelper(context_window=max_input_size, num_output=num_output, chunk_overlap_ratio=max_chunk_overlap, chunk_size_limit=chunk_size_limit)

storage_context = StorageContext.from_defaults(vector_store=pinecone_vector_store)
service_context = ServiceContext.from_defaults(embed_model=embed_model)
# Read txt files from data directory
documents = SimpleDirectoryReader('data').load_data()

index = GPTVectorStoreIndex(documents, llm_predictor=llm, embed_model=embed_model, prompt_helper=prompt_helper)

# index = GPTVectorStoreIndex.from_documents(documents, storage_context=storage_context,  service_context=service_context)

# index.save_to_disk("index.json")

# Query index with a question
retriver = index.as_retriever()
response = retriver.retrieve("What is azure openai service?")
print(response)