from VectorDB import VectorDB
from LLMClient import LLMClient

from dotenv import load_dotenv


def prompt(client:LLMClient) :
	while(True) :
		try: text = input("Enter a query: ")
		except EOFError : break
		except KeyboardInterrupt : break

		if (text.__len__() == 0) :
			continue

		if(text.lower() == "exit") :
			break

		response = client.prompt(text)

		print()
		print(response)
		print()


def main() :
	load_dotenv()

	VectorDB.setup()
	VectorDB.connect()

	client = LLMClient("Test")

	prompt(client)


main()