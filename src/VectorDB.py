import os
from dotenv import load_dotenv

from nltk.tokenize import sent_tokenize
from langchain.schema.document import Document
from google.cloud import aiplatform, aiplatform_v1
from langchain_community.document_loaders import PyPDFLoader
from vertexai.language_models import TextEmbeddingModel, TextEmbedding


class VectorDB :
	CHUNK = 3072 		# Page size in characters
	INDEX = None 		# The Vertext Index ID
	ENDPOINT = None 	# The Vertext Index Endpoint ID

	index:aiplatform.MatchingEngineIndex = None
	endpoint:aiplatform.MatchingEngineIndexEndpoint = None


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

		VectorDB.INDEX = VectorDB.index.resource_name
		VectorDB.ENDPOINT = VectorDB.endpoint.resource_name

		print("Created VectorDB "+VectorDB.INDEX+" "+VectorDB.ENDPOINT)


	def connect() :
		VectorDB.INDEX = os.environ.get("INDEX")
		VectorDB.ENDPOINT = os.environ.get("ENDPOINT")
		VectorDB.index = aiplatform.MatchingEngineIndex(VectorDB.INDEX)
		VectorDB.endpoint = aiplatform.MatchingEngineIndexEndpoint(VectorDB.ENDPOINT)


	def query(text:str) :
		idx = "VectorDB"
		embeddings:list[TextEmbedding] = VectorDB.getEmbeddings([text])

		# Execute the request
		response = VectorDB.endpoint.find_neighbors(
			deployed_index_id=idx,
			queries=[embeddings[0].values]
		)

		for hits in response :
			for hit in hits :
				print(hit.id)


	def store(id:str, doc:Document, embeddings:list[TextEmbedding]) :
		datapoints = []

		for i in range(0,len(embeddings)) :
			dpid = id+"[" + str(i) + "]"

			datapoints.append(aiplatform_v1.IndexDatapoint(
		   	datapoint_id=dpid,  # Unique ID for this vector
    			feature_vector=embeddings[i].values,  # Must be a list of floats
    			restricts=[],  # Optional metadata filters
			))

		VectorDB.index.upsert_datapoints(datapoints=datapoints)



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
			VectorDB.store(id,doc,VectorDB.getEmbeddings(chunks[i]))



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
			if (csize + len(sentences[s]) > VectorDB.CHUNK) :
				chunks.append(chunk)
				csize = 0
				chunk = []
				print("Do overlap")

			ssize = len(sentences[s])
			chunk.append(sentences[s])
			csize += ssize

		return(chunks)


	def getEmbeddings(text:list[str]) :
		model = TextEmbeddingModel.from_pretrained("textembedding-gecko-multilingual@001")
		embedding = model.get_embeddings(text)
		return(embedding)


	def concat(chunks:list[str]) :
		text = ""
		count = len(chunks)

		for i in range(0,count) :
			if (i > 0) : text += ""
			text += chunks[i]

		return(text)