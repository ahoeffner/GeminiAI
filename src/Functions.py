from google.genai.types import FunctionDeclaration, Tool, Retrieval

LocalFiles = FunctionDeclaration(
	name="LocalFiles",
	description="show a local file on alex's mac",
	parameters=
	{
		"type": "OBJECT",
		"properties":
		{
			"file": {"type": "STRING", "description": "the filename to display"}
		}
	}
)

CallOuts = Tool(function_declarations=[LocalFiles])