from vertexai.preview import rag
from google.cloud import aiplatform
from langchain.schema.document import Document
from vertexai.language_models import TextEmbeddingModel
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class VectorDB :
	INDEX="projects/418382099622/locations/us-central1/indexes/6871855314623266816"
	ENDPOINT = "projects/418382099622/locations/us-central1/indexEndpoints/2872131079936933888"

	index = None
	endpoint = None
	corpus = None
	vectordb = None


	def create() :
		index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
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

		endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
 	   	display_name="VectorDB", public_endpoint_enabled=True
		)

		endpoint.deploy_index(
    		index=index, deployed_index_id="VectorDB"
		)

		VectorDB.ENDPOINT = index.resource_name

		#corpus = rag.create_corpus("VectorDB", index=index, endpoint=endpoint)

		print("Created VectorDB "+index.resource_name+" "+endpoint.resource_name)


	def connect() :
		VectorDB.index = aiplatform.MatchingEngineIndex(VectorDB.INDEX)
		VectorDB.endpoint = aiplatform.MatchingEngineIndexEndpoint(VectorDB.ENDPOINT)
		VectorDB.vectordb = rag.VertexVectorSearch(VectorDB.ENDPOINT,"VectorDB")


	def createCorpus() :
		text = "This is the text I want to embed."
		model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")
		embedding = model.get_embeddings([text])
		print(embedding)

		#datapoint = aiplatform.gapic.IndexDatapoint(
		#	deployed_index_id=VectorDB.INDEX,  # Important: Deployed index ID
		#	datapoint_id="file_name,chunk",  # Unique ID within the deployed index
		#	embedding=embedding,
		#	metadata={"text": "content"},
		#)

		#aiplatform.MatchingEngineIndex.upsert_datapoints(VectorDB.index, datapoints=[datapoint])
		#VectorDB.index.upsert_datapoints()
		#VectorDB.corpus = rag.create_corpus(display_name="Documents", vector_db=VectorDB.vectordb)
		#print(VectorDB.corpus)
		# projects/418382099622/locations/us-central1/ragCorpora/2305843009213693952
		#rag.delete_corpus("projects/418382099622/locations/us-central1/ragCorpora/2305843009213693952")
		#corpus = rag.create_corpus(display_name="Documents", description="")
		#print(corpus)


	def load(doc:Document) :
		print("Storing ",doc.metadata, len(doc.page_content))


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


	def splitText(docs:list[Document]) :
		splitter = RecursiveCharacterTextSplitter()
		return(splitter.split(docs))


	def embedding() :
		return(TextEmbeddingModel.from_pretrained("text-embedding-005"))
