# from langchain_openai import OpenAIEmbeddings
# # from langchain.vectorstores.chroma import Chroma
# from langchain_chroma import Chroma
# from langchain.chains import RetrievalQA
# # from langchain.chat_models import ChatOpenAI
# from langchain_openai import ChatOpenAI

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
# from langchain_classic.chains import RetrievalQA
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.tracers import ConsoleCallbackHandler
from redundant_filter_retriever import RedundantFilterRetriever
# import langchain

# langchain.debug = True


from dotenv import load_dotenv

load_dotenv()

chat = ChatOpenAI(   model="gpt-4o-mini",
    temperature=0,
    stream_usage=False)
embeddings = OpenAIEmbeddings()
db = Chroma(
  persist_directory="emb",
  embedding_function=embeddings
)

# retriever = db.as_retriever()
retriever = RedundantFilterRetriever(
  embeddings=embeddings,
  chroma=db
)

chain = RetrievalQA.from_chain_type(
  llm = chat,
  retriever=retriever,
  chain_type="stuff",
)

# result= chain.invoke(
#     {"query": "What is an interesting fact about the English language"}
# )

result = chain.invoke(
    {"query": "What is an interesting fact about the English language"},
    config={"callbacks": [ConsoleCallbackHandler()]}
)

print(result)