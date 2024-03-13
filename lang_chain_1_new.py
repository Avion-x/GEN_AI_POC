import os

from dotenv import load_dotenv, find_dotenv
# from langchain.llms import AzureOpenAI
from langchain_community.llms import AzureOpenAI
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms.bedrock import LLM
from langchain_openai.llms.azure import AzureOpenAI
from langchain_openai import AzureOpenAIEmbeddings

# from langchain.llms.azure import AzureOpenAIEmbedding
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from pinecone import Pinecone

# Load environment variables from a `.env` file (if present)
load_dotenv(find_dotenv())

# Retrieve environment variables
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')
pinecone_index = os.environ.get("PINECONE_API_INDEX")
AZURE_OPENAI_ENDPOINT = os.environ.get('AZURE_OPENAI_ENDPOINT')
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Create Pinecone vector store
pinecone_vector_store = PineconeVectorStore(
    api_key=PINECONE_API_KEY,
    index_name=pinecone_index,
    environment=PINECONE_API_ENV,
    namespace="alice_new_5"
)

# Create Azure OpenAI embedding model
embed_model = AzureOpenAIEmbeddings(
    model="text-embedding-ada-002",
    deployment="OpenAIEmbeddings",
    api_key=OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version="2023-07-01-preview"
)

# Create VectorStoreIndex and retriever
vector_index = VectorStoreIndex.from_vector_store(vector_store=pinecone_vector_store, embed_model=embed_model)
retriever = VectorIndexRetriever(vector_index, similarity_top_k=3)  # Retrieve top 3 results

# Query vector database
query = "Which protocol can we configure on mx480?"

# query = "How to install the routing engine for mx480 router"
answer = retriever.retrieve(query)

# Combine top retrieved document content (if any) for LLM processing
top_content = ""
docs = []
if answer:
    for retrieved_doc in answer:
        res = retrieved_doc.get_content() + "\n" 
        top_content += res # Concatenate with newlines
        docs.append(res)
        

print("result from vector database is :  ")
for i in docs:
    print(i)
    print("--------------")


from langchain_community.chat_models import ChatCohere
from langchain_core.messages import AIMessage, HumanMessage

cohere_chat_model = ChatCohere(cohere_api_key=os.environ["COHERE_API_KEY"])

current_message_with_prompt = [
    HumanMessage(content=f"Here is some information retrieved from the knowledge base for the query {query} : {docs}. Give me the summarized results in one or two sentences")
]


a = cohere_chat_model.invoke(current_message_with_prompt)

print(current_message_with_prompt)
print()
print(a.content)

prompt = f"Generate 2 unit test cases and 2 python test scripts for mx480 router having knowledge base data for query {query} is {a.content}."

current_message_with_prompt = [
    HumanMessage(content= prompt+''' Each testcase and testscript should be encapsulated within its own separate JSON object, and it is an object under the "testname" key. All these JSON objects should be assembled within a Python list, resulting in [\{ "testname":"", "testcase":{}, "testscript":{}}] Each test case should include a testname, objective, steps (list of expected results), relevant test data. Make sure each test script JSON object includes the following fields: 'testname', 'objective', 'file_name', 'init_scripts'. The 'init_scripts' field should contain pip install commands for all required packages, 'script' (given in triple quotes), 'run_command' (a command to execute the python file) and 'expected_result'. The Python list with the JSON objects should not include any unrelated context or information. Find the starting delimiter as ###STARTLIST### and the ending delimiter as ###ENDLIST###''')
]

b = cohere_chat_model.invoke(current_message_with_prompt)

print(current_message_with_prompt)
print()
print(b.content)






