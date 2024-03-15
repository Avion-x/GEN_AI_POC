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
    namespace="alice_new_61"
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

# registry = {
#     "Bootup process" : "",
#     "Routing protocols (OSPF, BGP, IS-IS, etc)" : "",
#     "Firewall filters" : "",
#     "NAT configuration" : "",
#     "VPN configuration" : "",
#     "Access control lists" : "",
#     "QoS configuration" : "",
#     "Interface configuration" : "",
#     "VLAN configuration" : "",
#     "Static routes" : "",
#     "Policy configuration" : "",
#     "SNMP configuration" : "",
#     "NTP configuration" : "",
#     "Logging and monitoring" : "",
#     "Image install and upgrade" : "",
#     "Configuration rollback" : "",
#     "Chassis redundancy" : "",
#     "Line card redundancy" : "",
#     "Power supply redundancy" : "",
#     "Fabric redundancy" : "",
#     "Control plane protection" : "",
#     "Link aggregation" : "",
#     "IPsec tunnel connectivity" : "",
#     "GRE tunnel connectivity" : "",
#     "Multicast routing" : "",
#     "MPLS configuration" : "",
#     "VRRP configuration" : "",
#     "IPv6 configuration" : "",
#     "DNS configuration" : "",
#     "RADIUS authentication" : "",
#     "User authentication" : "",
#     "SSH connectivity" : "",
#     "NETCONF connectivity" : "",
#     "ZTP functionality" : "",
#     "L2C/L2VPN functionality" : "",
#     "L3C/L3VPN functionality" : "",
#     "EVPN functionality" : "",
#     "VxLAN functionality" : "",
#     "SD-WAN functionality" : "",
#     "Class of service" : "",
#     "Forwarding class" : "",
#     "Schedulers" : "",
#     "Shapers" : "",
#     "Policers" : "",
#     "Filters - firewall, route, etc" : "",
#     "Forwarding options" : "",
#     "Sampling configuration" : "",
#     "Auto-RP configuration" : "",
#     "MD5 authentication" : "",
#     "Media gateways" : "",
#     "SRX chassis cluster" : "",
#     "Session table utilization" : "",
#     "Next hop group testing" : "",
#     "Destination NAT" : "",
#     "Source NAT" : "",
#     "Port address translation" : "",
#     "Unified threat management" : "",
#     "Intrusion detection/prevention" : "",
#     "Antivirus scanning" : "",
#     "Traffic shaping per application" : "",
#     "Threat prevention policy" : "",
#     "Screen options" : "",
#     "Traceoptions" : "",
#     "Root authentication" : "",
#     "Password policies" : "",
#     "Zone-based firewall" : "",
#     "Content filtering" : "",
#     "Botnet detection" : "",
#     "Application identification" : "",
#     "URL filtering" : "",
#     "Anti-spam filtering" : "",
#     "Fan redundancy" : "",
#     "Graceful restart" : "",
#     "Bidirectional forwarding detection" : "",
#     "Spanning tree protocol" : "",
#     "DHCP configuration" : "",
#     "TACACS+ authentication" : "",
#     "ECMP load balancing" : "",
#     "Sticky MAC configuration" : "",
#     "Proxy ARP" : "",
#     "IP directed broadcast" : "",
#     "Denial of service protection" : "",
#     "Loopback interface connectivity" : "",
#     "Management interface connectivity" : "",
#     "Inline tap interface connectivity" : "",
#     "MACsec configuration" : "",
#     "Lawful intercept configuration" : "",
#     "Port mirroring" : "",
#     "sFlow monitoring" : "",
#     "Top talkers monitoring" : "",
#     "Interface counters and statistics" : "",
#     "Routing table validation" : "",
#     "Cisco EVPN Interoperability" : "",
#     "VMware NSX integration" : "",
#     "Memory utilization" : "",
#     "CPU utilization" : "",
#     "Hardware sensors/environmentals" : "",
#     "Jitter buffer configuration" : "",
#     "Auto-VoIP configuration" : "",
#     "SIP ALG configuration" : "",
# }

registry = {
    "GRE Tunnel Connecectivity" : "Tunnel Interface, Tunnel Endpoint, Tunnel Source, Tunnel Destination, Tunnel Mode, Tunneling Protocol, Tunnel Encapsulation, GRE tunnel connectivity: Tunnel Interface, Tunnel Endpoint, Tunnel Source, Tunnel Destination, Tunnel Mode, Tunneling Protocol, Tunnel Encapsulation,",
    "NAT Configuration" : 'Source NAT (SNAT), Destination NAT (DNAT), Static NAT, Dynamic NAT.',
    "ACL Configuration" : "ACL (Access Control List), Permit, Deny, Source Address, Destination Address.",
    "Routing Protocols" : ""
}

# Query vector database
# key = "Ip Security"
# key = "Routing Protocols"
key = "ports"
query = f"what are the {key}?"

# query = f"How many {key}?"

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

cohere_chat_model = ChatCohere(cohere_api_key=os.environ["COHERE_API_KEY"], temaperature=0.1)

# print(query)
# print(docs)

current_message_with_prompt = [
    HumanMessage(content=f"you are only allowed to search answer to my question within following tripple backticks, find {query} : ```{docs}```. result should as short as possible without information loss, If you don't find answer of my question in the given text simply say 'NOT FOUND' also when the given text is null or empty list say 'Not Found'")
]


a = cohere_chat_model.invoke(current_message_with_prompt)

# print(current_message_with_prompt)
print()
knowledge_base_summary = f"{registry.get(key, '')}" if ('Not Found'.lower() in a.content.lower()) else a.content

print( f"Query : {query} \nResult : {knowledge_base_summary}")

prompt = f"Generate 4 unit test cases for {key} along with python script for each test cases for mx480 router having knowledge base data for query {query} is {knowledge_base_summary}."

current_message_with_prompt = [
    HumanMessage(content= prompt+''' Each testcase and testscript should be encapsulated within its own separate JSON object, and it is an object under the "testname" key. All these JSON objects should be assembled within a Python list, resulting in [\{ "testname":"", "testcase":{}, "testscript":{}}] Each test case should include a testname, objective, steps (list of expected results), relevant test data. Make sure each test script JSON object includes the following fields: 'testname', 'objective', 'file_name', 'init_scripts'. The 'init_scripts' field should contain pip install commands for all required packages, 'script' (given in triple quotes), 'run_command' (a command to execute the python file) and 'expected_result'. The Python list with the JSON objects should not include any unrelated context or information. Find the starting delimiter as ###STARTLIST### and the ending delimiter as ###ENDLIST###''')
]

b = cohere_chat_model.invoke(current_message_with_prompt)

# print(current_message_with_prompt)
print()
print(b.content)






