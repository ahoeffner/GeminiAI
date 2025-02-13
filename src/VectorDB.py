from langchain_chroma import Chroma
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


class VectorDB :
	def create(location:str, docs:list[Document]) :
		for doc in docs :
			print("Storing ",doc)
		#Chroma.from_documents(docs,VectorDB.embedding(),persist_directory=location)


	def splitText(docs:list[Document]) :
		splitter = RecursiveCharacterTextSplitter()
		return(splitter.split(docs))


	def embedding() :
		return(None)