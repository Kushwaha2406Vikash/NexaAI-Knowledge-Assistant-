from langchain_google_genai import ChatGoogleGenerativeAI 
from dotenv import load_dotenv 
import getpass
import os 

load_dotenv() 



if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

def get_llm()->ChatGoogleGenerativeAI:

    return ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=1.0,  
    max_tokens=None,
    timeout=None,
    streaming=True
    
)