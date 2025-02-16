import os
import nltk

from VectorDB import VectorDB
from dotenv import load_dotenv
from google.cloud import aiplatform


DOCUMENTS = "../documents/"

load_dotenv()


def main() :
	print("Loading")

	PROJECT = os.environ.get("PROJECT_ID")
	GCREGION = os.environ.get("GOOGLE_CLOUD_REGION")

	nltk.data.path = [os.environ.get("NLTK_PATH")]
	aiplatform.init(project=PROJECT, location=GCREGION)

	VectorDB.connect()

	for file in os.listdir(DOCUMENTS) :
		print("file: "+file)
		VectorDB.loadPDF(DOCUMENTS+file)

	print("Done")

main()