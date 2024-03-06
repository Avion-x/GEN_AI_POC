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
    namespace="alice_new_1"
)

# Configure Azure OpenAI LLM
# llm = AzureOpenAI(
#     # model="gpt-35-turbo",
#     deployment_name = "OpenAIGPT35Turbo",
#     api_key=OPENAI_API_KEY,
#     azure_endpoint=AZURE_OPENAI_ENDPOINT,
#     api_version="2023-07-01-preview"
# )



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
answer = retriever.retrieve(query)

# Combine top retrieved document content (if any) for LLM processing
top_content = ""
docs = []
if answer:
    for retrieved_doc in answer:
        res = retrieved_doc.get_content() + "\n" 
        top_content += res # Concatenate with newlines
        docs.append(res)
        

from azure.core.credentials import AzureKeyCredential
credential = AzureKeyCredential(OPENAI_API_KEY)


from azure.ai.textanalytics import TextAnalyticsClient
text_analytics_client = TextAnalyticsClient(
    endpoint=AZURE_OPENAI_ENDPOINT,
    credential=credential
)


# Send combined content to LLM for further processing or summarization
llm_prompt = f"Here is some information retrieved from the knowledge base:\n{top_content}\n"
llm_prompt += "Can you summarize the key points or provide additional insights?"
# llm_response = llm.run(llm_prompt)

# print(docs)


documents = [{"id": "1", "text": llm_prompt}]
tasks = ["summarization"]  # Replace with other desired tasks (e.g., "key_phrases")
# response = text_analytics_client.analyze_sentiment(documents=docs)


# response = text_analytics_client.analyze_text(documents=[llm_prompt], tasks=["summarization"])
# for document in response.documents:
#     for summary in document.summarization:
#         llm_response = summary.sentence


from langchain_community.chat_models import ChatCohere
from langchain_core.messages import AIMessage, HumanMessage

cohere_chat_model = ChatCohere(cohere_api_key=os.environ["COHERE_API_KEY"])

# # Send a chat message without chat history
# current_message = [HumanMessage(content="Generate unit test cases for router mx480")]
# print(cohere_chat_model(current_message))

# Send a chat message with chat history, note the last message is the current user message
current_message_with_prompt = [
    HumanMessage(content=f"Mx480 protocol configurations are {top_content}"),
    AIMessage(content="You are a best network test engineer?"),
    HumanMessage(content="Generate unit test cases for router mx480") ]

current_message_with_prompt = [
    HumanMessage(content=f"The knowledge base results for {query} are {docs}. Give me the summarized results in one or two sentences")
]


a = cohere_chat_model.invoke(current_message_with_prompt)

print(current_message_with_prompt)

print(a.content)


# llm_response = llm.invoke(llm_prompt)

# Print LLM response
# print(llm_response)





