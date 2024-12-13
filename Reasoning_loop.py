import os
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.llms.mistralai import MistralAI
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.core import SummaryIndex, VectorStoreIndex
from llama_index.core.vector_stores import MetadataFilters
from typing import List, Optional
from llama_index.core.vector_stores import FilterCondition
from llama_index.core.tools import FunctionTool
from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner


documents = SimpleDirectoryReader(input_files=["papers/metagpt.pdf"]).load_data()
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

vector_index = VectorStoreIndex(nodes)


def vector_query(query: str, page_numbers: Optional[List[str]] = None) -> str:
    """Use to answer questions over a given paper.

    Useful if you have specific questions over the paper.
    Always leave page_numbers as None UNLESS there is a specific page you want to search for.

    Args:
        query (str): the string query to be embedded.
        page_numbers (Optional[List[str]]): Filter by set of pages. Leave as NONE
            if we want to perform a vector search
            over all pages. Otherwise, filter by the set of specified pages.

    """

    page_numbers = page_numbers or []
    metadata_dicts = [{"key": "page_label", "value": p} for p in page_numbers]

    query_engine = vector_index.as_query_engine(
        similarity_top_k=2,
        filters=MetadataFilters.from_dicts(
            metadata_dicts, condition=FilterCondition.OR
        ),
    )
    response = query_engine.query(query)
    return response


vector_query_tool = FunctionTool.from_defaults(name="vector_tool", fn=vector_query)

summary_index = SummaryIndex(nodes)
summary_query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize",
    use_async=True,
)
summary_tool = QueryEngineTool.from_defaults(
    name="summary_tool",
    query_engine=summary_query_engine,
    description=("Useful if you want to get a summary of MetaGPT"),
)
agent_worker = FunctionCallingAgentWorker.from_tools(
    [vector_query_tool, summary_tool], llm=Settings.llm, verbose=True
)
agent = AgentRunner(agent_worker)
response = agent.query(
    "Tell me about the agent roles in MetaGPT, "
    "and then how they communicate with each other."
)

print(str(response))
