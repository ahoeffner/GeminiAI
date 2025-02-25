import os
from dotenv import load_dotenv

from collections import defaultdict
from nltk.tokenize import sent_tokenize
from langchain.schema.document import Document
from google.cloud import aiplatform, aiplatform_v1
from langchain_community.document_loaders import PyPDFLoader
from vertexai.language_models import TextEmbeddingModel, TextEmbedding


class VectorDB :
	CHUNK = 3072 			# Chuncked Page size in characters
	STORE = "../gold"		# Where to store the parsed content

	DBNAME = "VectorDB"	# Name of the vector database/index

	INDEX = None 			# The Vertext Index ID
	ENDPOINT = None 		# The Vertext Index Endpoint ID

	index:aiplatform.MatchingEngineIndex = None # The vector index
	endpoint:aiplatform.MatchingEngineIndexEndpoint = None # The vector index endpoint


	def setup() :
		PROJECT = os.environ.get("PROJECT_ID")
		GCREGION = os.environ.get("GOOGLE_CLOUD_REGION")
		aiplatform.init(project=PROJECT, location=GCREGION)


	def create() :
		print(f"Create index {VectorDB.DBNAME}")
		VectorDB.index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
			display_name=VectorDB.DBNAME,
			description="Vector database to hold document embeddings",
			dimensions=768,
			approximate_neighbors_count=10,
			leaf_node_embedding_count=500,
			leaf_nodes_to_search_percent=7,
			distance_measure_type="DOT_PRODUCT_DISTANCE", # Use matrix dot product distance
			feature_norm_type="UNIT_L2_NORM",
			index_update_method="STREAM_UPDATE",
		)

		print(f"Create endpoint {VectorDB.DBNAME}")
		VectorDB.endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
 	   	display_name=VectorDB.DBNAME, public_endpoint_enabled=True
		)

		print(f"Deploy endpoint {VectorDB.DBNAME}")
		VectorDB.endpoint.deploy_index(
    		index=VectorDB.index, deployed_index_id=VectorDB.DBNAME
		)

		VectorDB.INDEX = VectorDB.index.resource_name
		VectorDB.ENDPOINT = VectorDB.endpoint.resource_name

		print("Created Vector Database "+VectorDB.INDEX+" "+VectorDB.ENDPOINT)


	def connect() :
		VectorDB.INDEX = os.environ.get("INDEX")
		VectorDB.ENDPOINT = os.environ.get("ENDPOINT")
		VectorDB.index = aiplatform.MatchingEngineIndex(VectorDB.INDEX)
		VectorDB.endpoint = aiplatform.MatchingEngineIndexEndpoint(VectorDB.ENDPOINT)


	def query(text:str) :
		embeddings:list[TextEmbedding] = VectorDB.getEmbeddings([text])

		# Execute the request
		response = VectorDB.endpoint.find_neighbors(
			deployed_index_id=VectorDB.DBNAME,
			queries=[embeddings[0].values]
		)

		results = defaultdict(list)

		for hits in response :
			for hit in hits :
				parts = hit.id.rsplit('[',2)
				parts[1] = parts[1].rstrip(']')
				parts[2] = parts[2].rstrip(']')

				pages = results[parts[0]]
				pages.append({"part#": int(parts[1]), "sentence#": int(parts[2])})

		for doc in results :
			results[doc].sort(key=lambda entry: (entry["part#"], entry["sentence#"]))

		content = ""
		# For now, just get each page. Later check if last sentences hit, and include next
		for doc in results :
			parts = list()

			for page in results[doc] :
				parts.append(page["part#"])

			parts = list(set(parts))

			for i in range(0,len(parts)) :
				parts[i] = doc + "[" + str(parts[i]) + "]"

			for chunk in parts :
				content += VectorDB.read(chunk) + "\n\n"

			return(content)


	def store(id:str, embeddings:list[TextEmbedding]) :
		datapoints = []

		for i in range(0,len(embeddings)) :
			dpid = id+"[" + str(i) + "]"

			datapoints.append(aiplatform_v1.IndexDatapoint(
		   	datapoint_id=dpid,  # Unique ID for this vector
    			feature_vector=embeddings[i].values,  # Must be a list of floats
    			restricts=[],  # Optional metadata filters
			))

		VectorDB.index.upsert_datapoints(datapoints=datapoints)


	def delete(id:str, chunks:int) :
		ids = []

		for i in range(0,chunks) :
			ids.append(id+"[" + str(i) + "]")

		VectorDB.index.remove_datapoints(datapoint_ids=ids)



	def load(doc:Document) :
		print(f"Storing {doc.metadata.get("source")}")

		path = doc.metadata.get("source")
		file = os.path.split(path)[1]

		chunks = VectorDB.splitText(doc)

		index = ""
		count = len(chunks)
		doc.page_content = ""
		doc.metadata["chunks"] = count

		for i in range(0,count)	:
			if (i > 0) : index += "\n"
			index += str(i) + " " + str(len(chunks[i]))

		VectorDB.save(file+".inf",index)

		for i in range(0,count)	:
			doc.metadata["chunk"] = i
			id = file + "[" + str(i) + "]"
			VectorDB.save(id,VectorDB.concat(chunks[i]))
			VectorDB.store(id,VectorDB.getEmbeddings(chunks[i]))



	def loadPDF(path:str) :
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


	def save(id:str, text:str) :
		f = open(VectorDB.STORE+os.sep+id,"w")
		f.write(text)
		f.close()


	def read(id:str) :
		print("read "+id)
		f = open(VectorDB.STORE+os.sep+id,"r")
		lines = f.readlines()
		f.close()

		content = ""
		for i in range(0,len(lines)) :
			content += lines[i]

		return(content)
