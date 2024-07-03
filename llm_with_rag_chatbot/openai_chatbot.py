# pip install docarray
import os
import re

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model=os.getenv("model"))


embeddings = OpenAIEmbeddings()

splits = []
knowledge_base = os.environ["KNOWLEDGE_BASE"]

files = os.listdir(knowledge_base)
for file in files:
    loader = TextLoader(os.path.join(knowledge_base, file))
    ko = loader.load()
    splits.extend(ko)

vectorstore2 = DocArrayInMemorySearch.from_documents(splits, embeddings)
template = """
Execute the function code in the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \

Context: {context}

Question: {question}
"""
# Do not include code or logic of the function in the responses. Instead, use your python interpreter tool to execute code functions and only use the final calculated value by the function to answer the questions. \

prompt = ChatPromptTemplate.from_template(template)
parser = StrOutputParser()
chain = (
    {"context": vectorstore2.as_retriever(), "question": RunnablePassthrough()}
    | prompt
    | model
    | parser
)
while True:
    text = input("Enter your query: --> ")
    response = chain.invoke(text)
    print(response)


# Example request: Can you calculate my life year gain if I stop using tobacco considering I am a 65 years old female that has been smoking for 10 years now and I still smoke and I smoke 5 cigarettes a day
