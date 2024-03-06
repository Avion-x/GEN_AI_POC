from llama_index.llms import LlamaCPP
from llama_index.embeddings import HuggingFaceEmbedding

llm = LlamaCPP(
    model_path="path/to/your/model/Mixtral_8x7B_Instruct_v0.1.gguf",  # https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF
    context_window=32000,
    max_new_tokens=1024,
    model_kwargs={'n_gpu_layers': 1},
    verbose=True
)

embedding_model = HuggingFaceEmbedding(model_name="WhereIsAI/UAE-Large-V1")  # https://huggingface.co/WhereIsAI/UAE-Large-V1

from llama_index import ServiceContext

service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embedding_model,
    system_prompt='You are a bot that answers questions about podcast transcripts'
)