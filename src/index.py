import os
import nltk

from VectorDB import VectorDB
from dotenv import load_dotenv


DOCUMENTS = "../bronze/"



def main() :
	load_dotenv()
	print("Starting")

	VectorDB.setup()
	VectorDB.connect()

	vdb:VectorDB = VectorDB()
	nltk.data.path = [os.environ.get("NLTK_PATH")]

	vdb.query("docker commands")
	#VectorDB.status()
	#VectorDB.loadPDF(DOCUMENTS+"Python.pdf")

	#for file in os.listdir(DOCUMENTS) :
	#	VectorDB.loadPDF(DOCUMENTS+file)

main()