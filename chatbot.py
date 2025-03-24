from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks import CallbackManagerForLLMRun
from typing import Any, List, Optional
import requests
import json
import os

# Set your OpenRouter API key
OPENROUTER_API_KEY = "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"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

class OpenRouterLLM(LLM):
    """Custom LLM class for OpenRouter API"""
    
    @property
    def _llm_type(self) -> str:
        return "openrouter"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the OpenRouter API and return the response"""
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "Odoo AI Assistant ",
            "Content-Type": "application/json"
        }
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
            {"role": "user", "content": prompt}
        ]
        
        data = {
            "model": "mistralai/mistral-small-3.1-24b-instruct:free",
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 1000
        }
        
        try:
            response = requests.post(OPENROUTER_URL, headers=headers, json=data)
            response.raise_for_status()
            response_data = response.json()
            
            if 'choices' not in response_data or not response_data['choices']:
                raise ValueError(f"Invalid API response: {json.dumps(response_data, indent=2)}")
                
            return response_data["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")
        except (KeyError, json.JSONDecodeError, ValueError) as e:
            raise Exception(f"Error processing API response: {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error: {str(e)}")

def load_vector_db(persist_directory="./vector_db", 
                  embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
    """Load the vector database"""
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return vectordb

def create_chatbot(vectordb):
    """Create a chatbot with the vector database"""
    # Initialize the custom LLM
    llm = OpenRouterLLM()

    # Create a prompt template
    prompt_template = """You are a helpful assistant that answers questions based on the provided context. 
    Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}

    Question: {question}

    Please provide a detailed and helpful answer based on the context provided:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # Create the conversation chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": PROMPT}
    )
    
    return qa_chain

def chat_loop(qa_chain):
    """Run the chat loop"""
    chat_history = []
    
    print("Hi I am Odoo AI Assistant!. (Type 'quit' to exit)")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nGoodbye! Have a great day!")
                break
                
            if user_input:
                try:
                    # Get the response from the chain
                    result = qa_chain.invoke({"question": user_input, "chat_history": chat_history})
                    
                    # Extract the answer
                    answer = result["answer"]
                    
                    # Update chat history
                    chat_history.append((user_input, answer))
                    
                    # Print the response
                    print("\nAssistant:", answer)
                    print("\nSources:", "-" * 40)
                    for doc in result["source_documents"]:
                        print(f"- {doc.page_content[:150]}...")
                        
                except Exception as e:
                    print(f"\nError: Something went wrong - {str(e)}")
                    print("Please try asking your question in a different way.")
                    
        except KeyboardInterrupt:
            print("\n\nChat session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nUnexpected error: {str(e)}")
            print("Please try again.")

if __name__ == "__main__":
    try:
        # Load the vector database
        print("Loading vector database...")
        vectordb = load_vector_db()
        
        # Create the chatbot
        print("Initializing chatbot...")
        qa_chain = create_chatbot(vectordb)
        
        # Start the chat loop
        chat_loop(qa_chain)
    except Exception as e:
        print(f"Failed to initialize the chatbot: {str(e)}") 