from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone
from llama_index.core.retrievers import VectorIndexRetriever
from dotenv import load_dotenv,find_dotenv
import os

load_dotenv(find_dotenv())
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')
pinecone_index = os.environ.get("PINECONE_API_INDEX")
cohere_api_key = os.environ["COHERE_API_KEY"]

pinecone_vector_store = PineconeVectorStore(api_key=PINECONE_API_KEY, index_name=pinecone_index, environment=PINECONE_API_ENV, namespace="alice_new_1")


# Configure OpenAI API
api_type = "azure"
api_version = "2023-07-01-preview"
api_base = os.getenv('AZURE_OPENAI_ENDPOINT')
api_key = os.getenv("OPENAI_API_KEY")

# Create LLM via Azure OpenAI Service
from langchain_openai.llms.azure import AzureOpenAI
llm = AzureOpenAI(
    model="OpenAIGPT35Turbo",
    # deployment_name="my-custom-llm",
    api_key=api_key,
    azure_endpoint=api_base,
    api_version=api_version,
)

# from llama_index.embeddings.cohere import CohereEmbedding
# embed_model = CohereEmbedding(
#     cohere_api_key=cohere_api_key,
#     model_name="embed-english-v3.0",
#     input_type="search_query",
# )

from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
embed_model = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name="OpenAIEmbeddings",
    api_key=api_key,
    azure_endpoint=api_base,
    api_version=api_version,
)

# from llama_index.core import Settings
# Settings.llm = llm
# Settings.embed_model = embed_model


# Instantiate VectorStoreIndex object from your vector_store object
vector_index = VectorStoreIndex.from_vector_store(vector_store=pinecone_vector_store, embed_model=embed_model)

retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=3)

# Query vector DB
answer = retriever.retrieve('Which protocol can we configure on mx480?')
# answer = retriever.retrieve('What are the mx480 system configurations?')

# answer = retriever.retrieve('What are the operating systems?')
# answer = retriever.retrieve('IPv6 Address Requirements in a Subscriber Access Network')


# Inspect results
print([i.get_content() for i in answer])

# >>> ['some relevant search result 1', 'some relevant search result 1'...]