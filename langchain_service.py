#coding:utf-8

from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
# import gradio as gr
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import os, io, sys
# 使用pysqlite替换系统sqlite包, 解决系统包版本过低问题
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
sys.path.append(os.getcwd())
import config
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.environ["OPENAI_API_KEY"] = config.openai_key
data_directory_path = config.data_directory_path

class LangChainService:

    def __init__(self, dir):
        self.db_dir = data_directory_path + '/' + dir
        self._client = chromadb.PersistentClient(self.db_dir)
        self.db = Chroma(persist_directory=self.db_dir, embedding_function=OpenAIEmbeddings(), client=self._client)
    
    def build(self, file_name):
        loader = TextLoader(data_directory_path + '/' +file_name, encoding="gbk")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10)
        docs = text_splitter.split_documents(documents)
        texts = [doc.page_content for doc in docs]
        metadatas = [doc.metadata for doc in docs]
        return self.add(texts, metadatas)

        # embedding = OpenAIEmbeddings()
        # self.db.from_documents(documents=docs, embedding=embedding, persist_directory=self.db_dir)
        # self.db.persist()

    def get(self, ids: list = None, metadatas: dict = None):
        return self.db.get(ids, metadatas)

    def add(self, texts: list, metadatas: dict = None, ids: list = None):
        ids = self.db.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        self.db.persist()
        return ids

    def update(self, ids: list, texts: list, metadatas: dict = None):
        self.db._collection.update(ids=ids, documents=texts, metadatas=metadatas)
        self.db.persist()
    
    def delete(self, ids: list, metadatas: dict = None):
        self.db._collection.delete(ids, metadatas)
        self.db.persist()
    
    def flush(self):
        self.db.delete_collection()
        self.db.persist()

    def query(self, texts, maxDistance = 0.5):
        '''
        distance: 距离远近, 表现为相似度高低, 大于0.5极低 0.4-0.5低 0.3-0.4中 0.2-0.3高 0.1-0.2极高 0.0完全一致, 语言不同相似度会低一档
        '''
        content = ""
        result = self.db.similarity_search_with_score(texts, k=3)
        print(result)
        sys.stdout.flush()
        for item in result: 
            distance = item[1]
            if distance < maxDistance and len(content) < 1000 and item[0].page_content not in content: content += item[0].page_content + "\n"
        # retriever = self.db.as_retriever()
        # result = retriever.get_relevant_documents(texts)
        # content = result[0].page_content
        # print(result)
        return content

if __name__ == "__main__":
    server = LangChainService('test')
    # print(server.delete(['80bac0da-f909-11ed-af81-0242ac110003', '9145998e-f909-11ed-bdcd-0242ac110003']))
    # server.flush();exit()
    # print(server.add(['test7', 'test8'], [{'test': 7}, {'test': 8}]))
    # print(server.update(['bbee3e2c-f952-11ed-b17b-0242ac110003', 'bbee3f1c-f952-11ed-b17b-0242ac110003'], ['test3', 'test4'], [{'test': 3}, {'test': 4}]))
    # print(server.build('xx.txt'))
    # print(server.get())
    # q = 'test1'
    # print(server.get(["405c0d59-56ca-11ee-952a-3c6a48bcd374"]))
    # print(server.add([q]))
    
    