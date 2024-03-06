from langchain.chains import StuffDocumentsChain, LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain


from langchain.chains.retrieval_qa.base import (
    RetrievalQA,
    VectorDBQA,
)

# document_prompt = PromptTemplate(
#     input_variables=["page_content"],
#     template="{page_content}"
# )
# document_variable_name = "context"
# llm = OpenAI()
# # The prompt here should take as an input variable the
# # `document_variable_name`
# prompt = PromptTemplate.from_template(
#     "Summarize this content: {context}"
# )
# llm_chain = LLMChain(llm=llm, prompt=prompt)
# chain = StuffDocumentsChain(
#     llm_chain=llm_chain,
#     document_prompt=document_prompt,
#     document_variable_name=document_variable_name
# )


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

from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
embed_model = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name="OpenAIEmbeddings",
    api_key=api_key,
    azure_endpoint=api_base,
    api_version=api_version,
)

vector_index = VectorStoreIndex.from_vector_store(vector_store=pinecone_vector_store, embed_model=embed_model)

retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=5)

# qa = RetrievalQA.from_chain_type(llm=llm,chain_type="stuff", retriever=retriver)


query = "What are mx480 router system configurations?"
docs = retriever.retrieve('Which protocol can we configure on mx480?')

# for doc in docs:
#     setattr(doc,'page_content', doc.get_content())

chain=load_qa_chain(llm,chain_type="stuff")
# response=chain.run(input_documents=docs,question=query)
# response = chain.invoke(input=docs, question=query)
# print(response)


from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def format_docs(docs):
    return "\n\n".join(doc.get_content() for doc in docs)

context = query
prompt = PromptTemplate.from_template(
    "Summarize this content: {context}"
)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)