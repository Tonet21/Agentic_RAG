import os
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.llms.mistralai import MistralAI
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.core import SummaryIndex, VectorStoreIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner
from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex
import requests

urls = [
    "https://openreview.net/pdf?id=VtmBAGCN7o",
    "https://openreview.net/pdf?id=6PmJoRfdaK",
    "https://openreview.net/pdf?id=LzPWWPAdY4",
    "https://openreview.net/pdf?id=VTF8yNQM66",
    "https://openreview.net/pdf?id=hSyW5go0v8",
    "https://openreview.net/pdf?id=9WD9KwssyT",
    "https://openreview.net/pdf?id=yV6fD7LYkF",
    "https://openreview.net/pdf?id=hnrB5YHoYu",
    "https://openreview.net/pdf?id=WbWtOYIzIK",
    "https://openreview.net/pdf?id=c5pwL0Soay",
    "https://openreview.net/pdf?id=TpD2aG1h0D",
]

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
    "vr_mcl.pdf",
]


mistral_api_key = os.environ.get("MISTRAL_API_KEY")


Settings.llm = MistralAI(
    api_key=mistral_api_key,
    model="mistral-large-latest",  # or "mistral-medium", "mistral-small"
)


Settings.embed_model = MistralAIEmbedding(
    api_key=mistral_api_key, model="mistral-embed"
)


paper_to_tools_dict = {}
# Loop through URLs and save each one

# Loop through URLs and save each one
for url, filename in zip(urls, papers):
    try:
        # Download the paper
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Save the paper
        with open(filename, "wb") as file:
            file.write(response.content)
        print(f"Downloaded {filename} successfully")

        # Process the downloaded paper
        try:
            # Load documents from the downloaded file
            documents = SimpleDirectoryReader(input_files=[filename]).load_data()

            # Split documents into nodes
            splitter = SentenceSplitter(chunk_size=1024)
            nodes = splitter.get_nodes_from_documents(documents)

            # Create indexes
            vector_index = VectorStoreIndex(nodes)
            summary_index = SummaryIndex(nodes)

            # Create query engines
            summary_query_engine = summary_index.as_query_engine(
                response_mode="tree_summarize", use_async=True
            )
            vector_query_engine = vector_index.as_query_engine()

            # Create tools
            summary_tool = QueryEngineTool.from_defaults(
                query_engine=summary_query_engine,
                description=(
                    f"Useful for summarization questions related to {filename}"
                ),
            )

            vector_tool = QueryEngineTool.from_defaults(
                query_engine=vector_query_engine,
                description=(
                    f"Useful for retrieving specific context from the {filename} paper."
                ),
            )

            # Store tools in dictionary
            paper_to_tools_dict[filename] = [vector_tool, summary_tool]

        except Exception as process_error:
            print(f"Error processing {filename}: {process_error}")

    except requests.RequestException as download_error:
        print(f"Failed to download {filename}: {download_error}")

all_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]
# define an "object" index and retriever over these tools


obj_index = ObjectIndex.from_objects(
    all_tools,
    index_cls=VectorStoreIndex,
)
obj_retriever = obj_index.as_retriever(similarity_top_k=3)
agent_worker = FunctionCallingAgentWorker.from_tools(
    tool_retriever=obj_retriever,
    llm=Settings.llm,
    system_prompt=""" \
You are an agent designed to answer queries over a set of given papers.
Please always use the tools provided to answer a question. Do not rely on prior knowledge.\

""",
    verbose=True,
)
agent = AgentRunner(agent_worker)

response = agent.query(
    "Tell me about the evaluation dataset used "
    "in MetaGPT and compare it against SWE-Bench"
)
print(str(response))
