from vertexai.preview import rag
from google.cloud import aiplatform
from nltk.tokenize import sent_tokenize
from langchain.schema.document import Document
from langchain_community.document_loaders import PyPDFLoader
from vertexai.language_models import TextEmbeddingModel, TextEmbedding


CHUNK = 3072

class VectorDB :
	INDEX="projects/418382099622/locations/us-central1/indexes/6871855314623266816"
	ENDPOINT = "projects/418382099622/locations/us-central1/indexEndpoints/2872131079936933888"

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

		VectorDB.ENDPOINT = VectorDB.index.resource_name
		print("Created VectorDB "+VectorDB.index.resource_name+" "+VectorDB.endpoint.resource_name)


	def connect() :
		VectorDB.index = aiplatform.MatchingEngineIndex(VectorDB.INDEX)
		VectorDB.endpoint = aiplatform.MatchingEngineIndexEndpoint(VectorDB.ENDPOINT)
		VectorDB.vectordb = rag.VertexVectorSearch(VectorDB.ENDPOINT,"VectorDB")


	def getEmbeddings(text:list[str]) :
		model = TextEmbeddingModel.from_pretrained("textembedding-gecko-multilingual@001")
		embedding = model.get_embeddings(text)
		return(embedding)


	def store(id:str, metadata, embedding:list[TextEmbedding]) :
		datapoint = aiplatform.gapic.IndexDatapoint(
			deployed_index_id=VectorDB.INDEX,
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
		doc.metadata["alex"] = "ALEX"
		print(f"Storing {doc.metadata.get("alex")} {doc.metadata.get("source")} bytes: {len(doc.page_content)}")
		VectorDB.splitText(doc)


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
			chunk.append({"text": sentences[s], "size": ssize})
			csize += ssize

		return(chunks)


	def splitTextOLD(doc:Document) :
		chunks = []
		sentences = []
		parts = sent_tokenize(doc.page_content)

		count = len(parts)
		print(f"sentences: {len(parts)}")

		for s in range(0,count) :
			parts[s] = parts[s].strip(" .;:?!")
			if (len(parts[s]) > 3) : sentences.append(parts[s])

		chunk = ""
		count = len(sentences)

		for s in range(0,count) :
			if (len(chunk) + len(sentences[s]) > CHUNK) :
				chunks.append(chunk)

				half = int(CHUNK/2)
				chunk = chunk[half:]
				chunk = "".join(chunk.split(" ",1))

			chunk = chunk+sentences[s]+" "

		for s in range(0,1) :
			print(f"----------------{s}-----------------")
			print(chunks[s])
			print()
			print()
