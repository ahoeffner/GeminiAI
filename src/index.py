from VectorDB import VectorDB
from langchain_community.document_loaders import PyPDFDirectoryLoader


DOCUMENTS = "data"
LOCATION = "database"

def main() :
	print("Loading")
	loader = PyPDFDirectoryLoader(DOCUMENTS)
	docs = loader.load()
	VectorDB.create(LOCATION,docs)
	print("Done")

main()