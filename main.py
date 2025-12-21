from config import Config
from rag_system import RAGSystem

def main():
    # Initialize Config
    config = Config()
    
    # Initialize System
    rag = RAGSystem(config)
    
    # Example Loop
    print("\n--- RAG System Ready (Type 'exit' to quit) ---")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        answer = rag.ask(user_input)
        print(f"AI: {answer}")

if __name__ == "__main__":
    main()