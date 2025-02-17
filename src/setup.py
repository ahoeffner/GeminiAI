import os
import sys
import nltk
from VectorDB import VectorDB
from dotenv import load_dotenv
from google.cloud import aiplatform

load_dotenv()

def nltkModel() :
	path = os.environ.get("NLTK_PATH")
	print("Downloading nltk punkt model to "+path)

	nltk.download("punkt",path)
	nltk.download("punkt_tab",path)


def vectorDB() :
	PROJECT = os.environ.get("PROJECT_ID")
	GCREGION = os.environ.get("GOOGLE_CLOUD_REGION")
	print(f"Setup VectorDB for project {PROJECT} in {GCREGION}")
	
	aiplatform.init(project=PROJECT, location=GCREGION)
	VectorDB.create()


if (__name__ == "__main__") :
	if (len(sys.argv) != 2) :
		print("Use setup nltk | vectordb")
		sys.exit(-1)

	arg = sys.argv[1].lower()

	if (arg == "nltk") :
		nltkModel()

	if (arg == "vectordb") :
		vectorDB()
