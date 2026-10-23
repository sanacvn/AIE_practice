# from langchain_core.runnables import RunnableLambda

# normalize_text = RunnableLambda(
#     lambda text: text.strip().lower()
# )

# analyze_text = RunnableLambda(
#     lambda text: len(text)
# )

# chain = normalize_text | analyze_texzt

# result = chain.run("   Hello, world!  ")
# print(result)
import os
import requests
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openrouter import ChatOpenRouter

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")


prompt = ChatPromptTemplate.from_messages([
    (
    "system",
    "you are an AI Engineer tutor."
    ),
    (
        "human",
        """ 
        Explain the concept :
        
        {input}
        """
    )
])

model = ChatOpenRouter(
    model="openrouter/free",
    temperature= 0.3,
    openrouter_api_key=API_KEY
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({
    "input" : "RAG"
})

print(result)

# url = "https://openrouter.ai/api/v1/chat/completions"

# headers = {
#     "Authorization": f"Bearer {API_KEY}",
#     "Content-Type": "application/json"
# }

# response = requests.post(
#     url,
#     headers=headers,
#     json={
#         "model": "openrouter/free",
#         "messages": [
#             {
#                 "role": "user",
#                 "content": "How many r's are in strawberry?"
#             }
#         ]
#     },
#     timeout=60
# )``

# response.raise_for_status()

# print(response.json()["choices"][0]["message"]["content"]) 