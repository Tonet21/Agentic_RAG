import os
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.llms.mistralai import MistralAI
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.core import SummaryIndex, VectorStoreIndex
from llama_index.core.vector_stores import MetadataFilters
from typing import List
from llama_index.core.vector_stores import FilterCondition
from llama_index.core.tools import FunctionTool
from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner


documents = SimpleDirectoryReader(input_files=["metagpt.pdf"]).load_data()
splitter = SentenceSplitter(chunk_size=1024)
nodes = splitter.get_nodes_from_documents(documents)


mistral_api_key = os.environ.get("MISTRAL_API_KEY")


Settings.llm = MistralAI(
    api_key=mistral_api_key,
    model="mistral-large-latest",  # or "mistral-medium", "mistral-small"
)


Settings.embed_model = MistralAIEmbedding(
    api_key=mistral_api_key, model="mistral-embed"
)

summary_index = SummaryIndex(nodes)
vector_index = VectorStoreIndex(nodes)

summary_query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize", use_async=True
)
vector_query_engine = vector_index.as_query_engine()





summary_tool = QueryEngineTool.from_defaults(
    query_engine=summary_query_engine,
    description=("Useful for summarization questions related to MetaGPT"),
)

vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    description=("Useful for retrieving specific context from the MetaGPT paper."),
)


agent_worker = FunctionCallingAgentWorker.from_tools(
    [vector_tool, summary_tool], llm=Settings.llm, verbose=True
)
agent = AgentRunner(agent_worker)
response = agent.query(
    "Tell me about the agent roles in MetaGPT, "
    "and then how they communicate with each other."
)

print(str(response))
