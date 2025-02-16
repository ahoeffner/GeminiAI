import os
from dotenv import load_dotenv

from vertexai.preview import rag
from google.cloud import aiplatform
from nltk.tokenize import sent_tokenize
from langchain.schema.document import Document
from langchain_community.document_loaders import PyPDFLoader
from vertexai.language_models import TextEmbeddingModel, TextEmbedding


CHUNK = 3072
INDEX = os.environ.get("INDEX")
ENDPOINT = os.environ.get("ENDPOINT")


class VectorDB :
	index = None
	endpoint = None
	vectordb = None

	def create() :
		VectorDB.index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
			display_name="VectorDB",
			description="Vector database to hold document embeddings",
			dimensions=768,
			approximate_neighbors_count=10,
			leaf_node_embedding_count=500,
			leaf_nodes_to_search_percent=7,
			distance_measure_type="DOT_PRODUCT_DISTANCE", # Use matrix dot product distance
			feature_norm_type="UNIT_L2_NORM",
			index_update_method="STREAM_UPDATE",
		)

		VectorDB.endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
 	   	display_name="VectorDB", public_endpoint_enabled=True
		)

		VectorDB.endpoint.deploy_index(
    		index=VectorDB.index, deployed_index_id="VectorDB"
		)

		INDEX = VectorDB.index.resource_name
		ENDPOINT = VectorDB.endpoint.resource_name

		print("Created VectorDB "+INDEX+" "+ENDPOINT)


	def connect() :
		INDEX = os.environ.get("INDEX")
		ENDPOINT = os.environ.get("ENDPOINT")

		print(f"---------- {INDEX} ---------")
		VectorDB.index = aiplatform.MatchingEngineIndex(INDEX)
		VectorDB.endpoint = aiplatform.MatchingEngineIndexEndpoint(ENDPOINT)
		VectorDB.vectordb = rag.VertexVectorSearch(ENDPOINT,"VectorDB")


	def getEmbeddings(text:list[str]) :
		model = TextEmbeddingModel.from_pretrained("textembedding-gecko-multilingual@001")
		embedding = model.get_embeddings(text)
		return(embedding)


	def store(id:str, metadata, embedding:list[TextEmbedding]) :
		text = "Hello and thank you for all the fish";
		embeddings = VectorDB.getEmbeddings([text])

		index_datapoint = aiplatform.gapic.IndexDatapoint(
			deployed_index_id=INDEX,  # Important: Deployed index ID
			datapoint_id="xx",  # Unique ID within the deployed index
			embedding=embedding,
			metadata={"text": text},
		)

		return

		datapoint = aiplatform.gapic.IndexDatapoint(
			deployed_index_id=INDEX,
			datapoint_id=id,  # Unique ID within the deployed index
			embedding=embedding,
			metadata=metadata,
		)

		print(datapoint)

		#aiplatform.MatchingEngineIndex.upsert_datapoints(VectorDB.index, datapoints=[datapoint])
		#VectorDB.index.upsert_datapoints()
		#VectorDB.corpus = rag.create_corpus(display_name="Documents", vector_db=VectorDB.vectordb)
		#print(VectorDB.corpus)
		# projects/418382099622/locations/us-central1/ragCorpora/2305843009213693952
		#rag.delete_corpus("projects/418382099622/locations/us-central1/ragCorpora/2305843009213693952")
		#corpus = rag.create_corpus(display_name="Documents", description="")
		#print(corpus)


	def load(doc:Document) :
		print(f"Storing {doc.metadata.get("source")} bytes: {len(doc.page_content)}")

		path = doc.metadata.get("source")
		chunks = VectorDB.splitText(doc)

		count = len(chunks)
		doc.metadata["chunks"] = count

		for i in range(0,count)	:
			doc.metadata["chunk"] = i
			id = path + "[" + str(i) + "]"
			doc.page_content = VectorDB.concat(chunks[i])
			VectorDB.store(id,doc.metadata,VectorDB.getEmbeddings(chunks[i]))



	def loadPDF(path) :
		print("Loading PDF: "+path)
		loader = PyPDFLoader(path)
		parts = loader.load()

		text = ""
		for part in parts :
			text += part.page_content + "\n"

		doc = Document(page_content=text)
		doc.metadata = parts[0].metadata

		return(VectorDB.load(doc))


	def splitText(doc:Document) :
		chunks = []
		sentences = []
		parts = sent_tokenize(doc.page_content)

		count = len(parts)

		for s in range(0,count) : # Clean up
			parts[s] = "".join(parts[s].strip(" .;:?!"))
			if (len(parts[s]) > 0) :
				if (len(parts[s]) > 1 or len(parts[s][0]) > 1) :
					sentences.append(parts[s])

		csize = 0
		chunk = []
		count = len(sentences)

		for s in range(0,count) :
			if (csize + len(sentences[s]) > CHUNK) :
				chunks.append(chunk)
				csize = 0
				chunk = []

			ssize = len(sentences[s])
			chunk.append(sentences[s])
			csize += ssize

		return(chunks)


	def concat(chunks:list[str]) :
		text = ""
		count = len(chunks)

		for i in range(0,count) :
			if (i > 0) : text += ""
			text += chunks[i]

		return(text)