from langchain_huggingface import HuggingFaceEndpoint
from app.config.config import HF_TOKEN, HUGGINGFACE_REPO_ID

from langchain_groq import ChatGroq
from app.config.config import GROQ_API_KEY

from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)

def load_llm(model_name: str = "llama3-8b-8192", groq_api_key: str = GROQ_API_KEY):
    try:
        logger.info("Loading LLM from Groq...")
        
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=model_name,
            temperature=0.3,
            max_tokens=256,
        )
        
        logger.info("LLM loaded successfully from Groq...")
        return llm
    except Exception as e:
        error_message = CustomException("Failed to load LLM from Groq", e)
        logger.error(str(error_message))
        return None