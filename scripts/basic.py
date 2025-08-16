from llama_index.core import Settings
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.llms.lmstudio import LMStudio

Settings.llm = LMStudio(
    model_name="google/gemma-3-12b",
    base_url="http://localhost:1234/v1"
)

base_conditions = [
    "Do not invent or generate any fictional data to respond to this question.", 
    "Keep your answer brief."
]

chat_engine = SimpleChatEngine.from_defaults(
    system_prompt = " ".join(base_conditions)
)

question_samples = [
    "What is the capital of France?",
    "When did the Berlin Wall come down?",
    "What do critics say about the 2004 movie 'Mean Girls'?",
    "What do critics say about the 2025 movie 'The Fantastic Four: First Steps'?"
]
for question in question_samples:
    print(f'Question: {question}\nAnswer: {chat_engine.chat(question).response}\n--------')