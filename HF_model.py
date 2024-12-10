import os
from llama_index.core import Settings
from llama_index.llms.gemini import Gemini 
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.agent import ReActAgent  # Use ReActAgent instead
import os
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.llms.gemini import Gemini 
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core import SummaryIndex, VectorStoreIndex
from llama_index.core.vector_stores import MetadataFilters
from typing import List
from llama_index.core.vector_stores import FilterCondition
from llama_index.core.tools import FunctionTool
from llama_index.core.tools import QueryEngineTool

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") 
Settings.llm = Gemini(
    model="models/gemini-1.5-flash", api_key=GOOGLE_API_KEY
)
Settings.embed_model = GeminiEmbedding(
    model_name="models/embedding-001", api_key=GOOGLE_API_KEY
)
documents = SimpleDirectoryReader(input_files=["metagpt.pdf"]).load_data()
splitter = SentenceSplitter(chunk_size=1024)
nodes = splitter.get_nodes_from_documents(documents) 
vector_index = VectorStoreIndex(nodes, embed_model=Settings.embed_model)


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



# Use ReActAgent with your tools
agent = ReActAgent.from_tools(
    [vector_tool, summary_tool], 
    llm=Settings.llm, 
    verbose=True
)
response = agent.chat(
    "Tell me about all the evaluation datasets used in MetaGPT."
)
response = agent.chat("Tell me the results over one of the above datasets.")

print(str(response))