from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.prompts import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.lmstudio import LMStudio

Settings.llm = LMStudio(
    model_name="google/gemma-3-12b",
    base_url="http://localhost:1234/v1"
)

documents = SimpleDirectoryReader(input_dir="documents").load_data()

text_splitter = SentenceSplitter(
    chunk_size=250,
    chunk_overlap=50,
)

chunks = text_splitter.get_nodes_from_documents(documents)

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

index = VectorStoreIndex(nodes=chunks)

custom_prompt_template = PromptTemplate(
    """<start_of_turn>user
    Context information is below.
    ---------------------
    {context_str}
    ---------------------
    Do not invent or generate any fictional data to respond to this question. 
    Keep your answer brief.
    Given the context information and not prior knowledge, answer the query.
    Query: {query_str}
    <end_of_turn>
    <start_of_turn>model"""
)

query_engine = index.as_query_engine(
    response_mode="compact",
    text_qa_template=custom_prompt_template,
)

query = "What do critics say about the 2025 movie 'The Fantastic Four: First Steps'?"
response = query_engine.query(query)

print(f"Question: {query}")
print(f"Answer: {response}")
