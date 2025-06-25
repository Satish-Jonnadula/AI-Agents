import os
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.7
)
responce= llm.invoke("Hello, world!")
print(responce.content)