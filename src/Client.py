import os

from google import genai
from VectorDB import VectorDB
from Functions import CallOuts
from google.cloud import aiplatform
from google.genai.types import GenerateContentConfig, Part


ROLE = '''
Your role is to answer questions from employees solely based on information from our knowledge base.
You will be provided with facts from this knowledge base during each session.
You can only use this information when answering questions.

Some information can be retrieved more efficiently by calling local functions.
Make sure always to check is any local functions can be applied
'''


class Client :
	def __init__(self, name):
		self.name = name

		MODEL = os.environ.get("MODEL")
		PROJECT = os.environ.get("PROJECT_ID")
		GCREGION = os.environ.get("GOOGLE_CLOUD_REGION")

		VectorDB.setup()
		VectorDB.connect()

		self.client = genai.Client(vertexai=True, project=PROJECT, location=GCREGION)

		self.chat = self.client.chats.create(
			model=MODEL,
			config=GenerateContentConfig(
			temperature=0,
			tools=[CallOuts],
			system_instruction=ROLE,
    		)
		)

	def prompt(self, question:str) -> str :
		# who is obama
		# list all docker commands
		# is that all docker commands
		# list the file Client.py from alex's mac

		# Local knowledge base query
		facts = VectorDB.query(question)

		question = facts + "\n\n" + question
		response = self.chat.send_message(question)

		if (response.function_calls) :
			for function_call in response.function_calls :
				name = function_call.name
				args = function_call.args

				if (name == "LocalFiles") :
					content = Client.readFile("src/"+args["file"])
					response = self.chat.send_message(Part.from_function_response(name=name,response={"content": content}))

		return(response.candidates[0].content.parts[0].text)


	def readFile(filename):
		try:
			with open(filename, 'r') as f:
				content = f.read()
				return(content)

		except FileNotFoundError:
			return f"Error: File '{filename}' not found."
		except Exception as e:
			return f"An error occurred while reading the file: {e}"


	def __str__(self):
		return f'Client: {self.name}'
