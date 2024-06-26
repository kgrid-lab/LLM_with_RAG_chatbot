from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory


try:
    if load_dotenv('.env') is False:
        raise TypeError
except TypeError:
    print('Unable to load .env file.')
    quit()

# Define llm parameters
llm = AzureChatOpenAI(
    deployment_name=os.environ['model'],
    openai_api_version=os.environ['API_VERSION'],
    openai_api_key=os.environ['OPENAI_API_KEY'],
    azure_endpoint=os.environ['openai_api_base'],
    openai_organization=os.environ['OPENAI_organization'],    
    )
knowledge_base=os.environ['KNOWLEDGE_BASE']

splits=[]
files = os.listdir(knowledge_base)
for file in files:   
    loader = TextLoader(os.path.join(knowledge_base,file))
    ko = loader.load()
    splits.extend(ko)
    
print(splits)

# Settings for embeddings
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.environ['openai_api_base'], 
    openai_api_version=os.environ['API_VERSION'],  
    openai_api_key=os.environ['OPENAI_API_KEY'],   
    openai_organization=os.environ['OPENAI_organization']   
)
print("Embedding documents...")
# Create the vectorstore
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

print("Documents embedded.")

# Retrieve and generate using the relevant snippets of the data
retriever = vectorstore.as_retriever(search_kwargs={"k": min(4, len(splits))}) # 4 is default k, ensures we aren't indexing greater than num elts available

# Contextualize question
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Answer question
qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
Do not simplify the formulas used to calculate the values. \
If you don't know the answer, just say that you don't know. \

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Manage chat history
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# User requests
while True:
  text = input('Enter your query (Example: How many undergrad students are at U of M?): --> ') # Example: How many undergrad students are at U of M?
  print(conversational_rag_chain.invoke(
      {"input": text},
      config={"configurable": {"session_id": "0"}},
  )["answer"])
