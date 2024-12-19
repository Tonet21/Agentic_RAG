# Multi-Document RAG Agent with LlamaIndex

This project demonstrates the creation of a **multi-document Retrieval-Augmented Generation (RAG) agent** using LlamaIndex. It includes several intermediate steps that serve as building blocks for the final agent. These steps illustrate the following concepts:

1. **Router**: How to create and implement a routing mechanism for queries.
2. **Function Calling**: Leveraging function-calling capabilities in the process.
3. **Reasoning Loop**: Developing a loop to enhance reasoning capabilities.

You will notice that the **Reasoning Loop** and **Multi-Document RAG agent** implementations appear twice in this project:

- Once using the **Mistral API**.
- Once using the **Gemini API**.

This duplication is intentional. LlamaIndex provides certain functionalities (as of this writing) for OpenAI, Anthropic, and Mistral APIs. By including Gemini API implementations, this project showcases how similar functionalities can be achieved with other APIs.

---

## Getting Started

### 1. Clone the Repository
Begin by cloning this repository to your local machine:
```bash
git clone <repository_url>
cd <repository_directory>
```

### 2. Install dependencies
Install the required Python packages by running:
```bash
pip install -r requirements.txt
```

### 3. Data Files: The "papers" Folder
The repository includes a folder named papers, which contains the documents used for retrieval in the RAG system. These files will serve as the knowledge base for the agent.

### 4. Python Scripts
The Python files implementing the intermediate steps and the final RAG agent are located in the root directory. You can run these scripts directly from the terminal.

For example:
```bash
python Reasoning_loop.py
```
Replace Reasoning_loop.py with the desired script you wish to execute.


## Code Formatting
This project adheres to consistent code formatting using the Python package black. Make sure to run it after making any code modifications to maintain a clean and uniform style:
```bash
black .
```

## Project Structure

```plaintext
├── papers/                   # Contains the documents used for RAG retrieval
├── requirements.txt          # Lists the dependencies for the project
├── Reasonig_loop.py # Script for Reasoning Loop using Mistral API
├── Gemini_reasoning_loop.py  # Script for Reasoning Loop using Gemini API
├── Multi_doc_agent.py # Multi-Document RAG agent with Mistral API
├── Gemini_multidoc_agent.py  # Multi-Document RAG agent with Gemini API
└── ... (other Python files and scripts)

