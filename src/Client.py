import os
from google import genai
from Functions import CallOuts
from google.genai.types import GenerateContentConfig, Part


RAG = "from the following context: {context}, answer the following question {message}"


class Client :
	def __init__(self, name):
		self.name = name

		MODEL = os.environ.get("MODEL")
		PROJECT = os.environ.get("PROJECT_ID")
		GCREGION = os.environ.get("GOOGLE_CLOUD_REGION")

		self.client = genai.Client(vertexai=True, project=PROJECT, location=GCREGION)

		self.chat = self.client.chats.create(
			model=MODEL,
			config=GenerateContentConfig(
			temperature=0,
			tools=[CallOuts],
			system_instruction="Use functions if available, otherwise just skip the tools and answer the question using a temperature of 1.0",
    		)
		)

	def prompt(self, message:str) -> str :
		# who is obama
		# list the file Client.py from alex's mac
		response = self.chat.send_message(message)

		if (response.function_calls) :
			for function_call in response.function_calls :
				name = function_call.name
				args = function_call.args

				if (name == "LocalFiles") :
					content = Client.readFile("src/"+args["file"])
					response = self.chat.send_message(Part.from_function_response(name=name,response={"content": content}))

		else :
			print()
			print("Using default response")
			response = self.chat.send_message(RAG.format(context="alex is a developer", message=message))

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
