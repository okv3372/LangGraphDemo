from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Azure OpenAI model
azureOpenAiEndpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azureOpenAiApiKey = os.getenv("AZURE_OPENAI_API_KEY")
azureOpenAiDeploymentName = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
azureOpenAiApiVersion = os.getenv("AZURE_OPENAI_API_VERSION")

# Initialize the Azure OpenAI model
llm = AzureChatOpenAI(
    azure_deployment=azureOpenAiDeploymentName,
    azure_endpoint=azureOpenAiEndpoint,
    api_key=azureOpenAiApiKey,
    model_version=azureOpenAiDeploymentName,
    api_version=azureOpenAiApiVersion
)

print(llm.invoke("What is the capital of France?"))