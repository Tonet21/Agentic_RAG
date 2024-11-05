from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
import os
from typing import List, Optional
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import SummaryIndex, VectorStoreIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector


documents = SimpleDirectoryReader(input_files=["metagpt.pdf"]).load_data()
splitter = SentenceSplitter(chunk_size=1024)
nodes = splitter.get_nodes_from_documents(documents)
##HF_TOKEN: Optional[str] = os.getenv("HUGGING_FACE_TOKEN") ##set hf token env var
HF_TOKEN = "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"
Settings.llm = HuggingFaceInferenceAPI(
    model_name="HuggingFaceH4/zephyr-7b-alpha", token=HF_TOKEN
)



Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")



summary_index = SummaryIndex(nodes)
vector_index = VectorStoreIndex(nodes, embed_model=Settings.embed_model)

summary_query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize", use_async=True, llm=Settings.llm
)
vector_query_engine = vector_index.as_query_engine(llm=Settings.llm)





summary_tool = QueryEngineTool.from_defaults(
    query_engine=summary_query_engine,
    description=("Useful for summarization questions related to syllos"),
)

vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    description=("Useful for retrieving specific context from the syllos paper."),
)




query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[
        summary_tool,
        vector_tool,
    ],
    verbose=True,
)


response = query_engine.query("What is the summary of the document?")
print(str(response))
