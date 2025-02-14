import os

from VectorDB import VectorDB
from dotenv import load_dotenv
from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel


DOCUMENTS = "../documents/"

load_dotenv()


def main() :
	print("Loading")

	PROJECT = os.environ.get("PROJECT_ID")
	GCREGION = os.environ.get("GOOGLE_CLOUD_REGION")

	aiplatform.init(project=PROJECT, location=GCREGION)

	#VectorDB.create()
	VectorDB.connect()
	VectorDB.createCorpus()

	#for file in os.listdir(DOCUMENTS) :
		#print("1 file: "+file)
		#VectorDB.loadPDF(DOCUMENTS+file)

	print("Done")

main()