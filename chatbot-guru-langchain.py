import getpass
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM


#os.environ["LANGCHAIN_TRACING_V2"] = "true"
#os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()

template = """Question: {question}

Answer: Let's think step by step."""

prompt = ChatPromptTemplate.from_template(template)

model = OllamaLLM(model="llama3")

chain = prompt | model

response = chain.invoke({"question": "What is LangChain?"})
print(response)