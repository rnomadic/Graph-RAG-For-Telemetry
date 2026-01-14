import os
from langchain_openai import ChatOpenAI

# Kubernetes injects these Environment Variables
LLM_API_BASE = os.environ.get("OPENAI_API_BASE", "http://vllm-service.ai-support.svc.cluster.local:8000/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "meta-llama/Meta-Llama-3-70B-Instruct")

def get_llm():
    """
    Returns a LangChain LLM object connected to the self-hosted vLLM instance.
    """
    print(f"Connecting to vLLM at: {VLLM_URL}")
    
    llm = ChatOpenAI(
        model=MODEL_NAME,
        openai_api_base=VLLM_URL,
        openai_api_key="EMPTY", # vLLM ignores this unless configured otherwise
        temperature=0.1,
        max_tokens=512
    )
    
    return llm

# Example Usage
if __name__ == "__main__":
    llm = get_llm()
    response = llm.invoke("Diagnose a battery drain on XPS 13.")
    print(response.content)
