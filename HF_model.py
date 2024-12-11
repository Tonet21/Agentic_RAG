import os
from llama_index.core import Settings
from llama_index.llms.gemini import Gemini 
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.agent import ReActAgent  # Use ReActAgent instead
import os
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import SummaryIndex, VectorStoreIndex
from llama_index.core.vector_stores import MetadataFilters
from typing import List
from llama_index.core.vector_stores import FilterCondition
from llama_index.core.tools import FunctionTool
from llama_index.core.tools import QueryEngineTool
from pathlib import Path



GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") 

documents = SimpleDirectoryReader(input_files=["metagpt.pdf"]).load_data()
splitter = SentenceSplitter(chunk_size=1024)
nodes = splitter.get_nodes_from_documents(documents) 

Settings.llm = Gemini(
    model="models/gemini-1.5-flash", api_key=GOOGLE_API_KEY

)



Settings.embed_model = GeminiEmbedding(
    model_name="models/embedding-001", api_key=GOOGLE_API_KEY
)

papers = [
    "metagpt.pdf",
    "longlora.pdf",
    "loftq.pdf",
    "swebench.pdf",
    "selfrag.pdf",
    "zipformer.pdf",
    "values.pdf",
    "finetune_fair_diffusion.pdf",
    "knowledge_card.pdf",
    "metra.pdf",
    "vr_mcl.pdf"
]

paper_to_tools_dict = {}
for paper in papers:
    documents = SimpleDirectoryReader(input_files=[paper]).load_data()
    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)
    summary_index = SummaryIndex(nodes)
    vector_index = VectorStoreIndex(nodes)

    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize", use_async=True
    )
    vector_query_engine = vector_index.as_query_engine()





    summary_tool = QueryEngineTool.from_defaults(
        name=f"summary_tool_{paper}",
        query_engine=summary_query_engine,
        description=(f"Useful for summarization questions related to {paper}"),
    )

    vector_tool = QueryEngineTool.from_defaults(
        name=f"vector_tool_{paper}",
        query_engine=vector_query_engine,
        description=(f"Useful for retrieving specific context from the {paper} paper."),
    )
    paper_to_tools_dict[paper] = [vector_tool, summary_tool]

all_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]
# define an "object" index and retriever over these tools
from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex

obj_index = ObjectIndex.from_objects(
    all_tools,
    index_cls=VectorStoreIndex,
)
obj_retriever = obj_index.as_retriever(similarity_top_k=3)
# Use ReActAgent with your tools
agent = ReActAgent.from_tools(
    tool_retriever=obj_retriever, 
    llm=Settings.llm, 
    verbose=True
)
response = agent.query(
    "Tell me about the evaluation dataset used "
    "in MetaGPT and compare it against SWE-Bench"
)
print(str(response))

## see how I can make optional the page number on the query_index_tool
## and if it works on reasoning_loop with gemini (create a new file) if it works do also
## the mutli_doc using gemini
## AFTER THIS I WOULD LIKE TO CREATE THE SAME AGENT USING HF MODEL  