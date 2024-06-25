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
