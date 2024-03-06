from llama_index.core import Document, VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone
from llama_index.embeddings.cohere import CohereEmbedding
from IPython.display import Markdown, display
from dotenv import load_dotenv,find_dotenv
import os


load_dotenv(find_dotenv())
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')
cohere_api_key = os.environ["COHERE_API_KEY"]

# embed_model = CohereEmbedding(
#     cohere_api_key=cohere_api_key,
#     model_name="embed-english-v3.0",
#     input_type="search_query",
# )


# pinecone_vector_store = PineconeVectorStore(api_key=PINECONE_API_KEY, index_name="awsknowledgebase2", environment=PINECONE_API_ENV, namespace="alice")

# storage_context = StorageContext.from_defaults(vector_store=pinecone_vector_store)

# index = VectorStoreIndex.from_documents(
#     nodes, storage_context=storage_context, embed_model=embed_model
# )

# print(index)


# Pincone connection with llama index
# from llama_index import download_loader

from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.ingestion import IngestionPipeline

pinecone_api_key = os.environ.get('PINECONE_API_KEY')
openai_api_key = os.environ.get('OPENAI_API_KEY')
pinecone_index = os.environ.get("PINECONE_API_INDEX")


# embed_model = CohereEmbedding(
#     cohere_api_key=cohere_api_key,
#     model_name="embed-english-v3.0",
#     input_type="search_query",
# )

from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
# Configure OpenAI API
api_type = "azure"
api_version = "2023-07-01-preview"
api_base = os.getenv('AZURE_OPENAI_ENDPOINT')
api_key = os.getenv("OPENAI_API_KEY")

embed_model = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name="OpenAIEmbeddings",
    api_key=api_key,
    azure_endpoint=api_base,
    api_version=api_version,
)


data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
print(data_dir)

documents = SimpleDirectoryReader(data_dir).load_data()
parser = SentenceSplitter(chunk_size=250, chunk_overlap=40) #(chunk_size=80, chunk_overlap=20, embed_model=embed_model)
# parser = SemanticSplitterNodeParser(chunk_size=80, chunk_overlap=20, embed_model=embed_model)
nodes = parser.get_nodes_from_documents(documents, show_progress=True)



pinecone_vector_store = PineconeVectorStore(api_key=PINECONE_API_KEY, index_name=pinecone_index, environment=PINECONE_API_ENV, namespace="alice_new_1")

pipeline = IngestionPipeline(
    transformations=[
        # SemanticSplitterNodeParser(
        #     buffer_size=2,
        #     breakpoint_percentile_threshold=95, 
        #     include_metadata=True, 
        #     embed_model=embed_model,
        #     ),
        embed_model,
        ],
        vector_store=pinecone_vector_store 
    )


pipeline.run(nodes=nodes)
# pipeline.run(documents=documents)

print()


# query_engine = index.as_query_engine()
# response = query_engine.query("What are the mx480 system configurations?")
# display(Markdown(f"<b>{response}</b>"))
# print()