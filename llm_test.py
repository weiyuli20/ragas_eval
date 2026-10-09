from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(
    model=os.environ["model_name"],
    api_key=os.environ["DASHSCOPE_API_KEY"],
    base_url = os.environ["base_url"]
)

print(llm.invoke("hi").content)