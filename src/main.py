from Client import Client
from dotenv import load_dotenv

load_dotenv()

thread = None
client = Client("Test")


def prompt(client:Client) :
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
	prompt(client)


main()