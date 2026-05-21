from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

azureOpenAiEndpoint = os.getenv("AZURE_OPENAI_ENDPOINT") # Your Azure OpenAI endpoint, e.g., "https://your-resource-name.openai.azure.com/"
azureOpenAiApiKey = os.getenv("AZURE_OPENAI_API_KEY") # Your Azure OpenAI API key
azureOpenAiDeploymentName = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT") # The name of your Azure OpenAI deployment, e.g., "gpt-4-deployment"  
azureOpenAiApiVersion = os.getenv("AZURE_OPENAI_API_VERSION") # The API version for Azure OpenAI, e.g., "2024-06-01"

# Initialize the Azure OpenAI model
llm = AzureChatOpenAI(
    azure_deployment=azureOpenAiDeploymentName,
    azure_endpoint=azureOpenAiEndpoint,
    api_key=azureOpenAiApiKey,
    model_version=azureOpenAiDeploymentName,
    api_version=azureOpenAiApiVersion
)

# Define the system message for the chat assistant
# SystemMessage sets the context for the AI's responses
messages = [
    SystemMessage(
        content="You are a helpful chat assistant. Answer the user's questions and fact check yourself whenever necessary. You aim to give concise and accurate answers to the user."
    )
]

def main():
    while True:
        question = input("User Input: ")

        if question.lower() in ['exit', 'quit']:
            print("Exiting the chat.")
            break

        
        # Append the user's question to the message history
        messages.append(HumanMessage(question))

        # Invoke the LLM with the current message history
        response = llm.invoke(messages)

        #Response is a dict w/things like token count, filters, etc
        #reponse.content is the actual response from the model

        # Append the AI's response to the message history
        messages.append(AIMessage(response.content))

        print("LLM: " + response.content)

if __name__ == "__main__":
    main()