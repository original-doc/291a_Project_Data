import os
import logging
import openai
import google.generativeai as genai
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class LLMGenerator:
    """
    Handles interactions with LLM providers (OpenAI, Google Gemini) to generate
    answers based on retrieved RAG context.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('llm', {})
        self.provider = self.config.get('provider', 'google') # Default to google based on your request
        self.model_name = self.config.get('model_name', 'gemini-2.5-flash')
        self.api_key = self.config.get('api_key_env_var')
        logger.info(f"Initialized API Key: {self.api_key}")
        # Setup API Key
        # Prioritize config var name, default to provider specific defaults
        # default_env_var = 'GEMINI_API_KEY' if self.provider == 'google' else 'OPENAI_API_KEY'
        # api_key_var = self.config.get('api_key_env_var', default_env_var)
        # self.api_key = os.getenv(api_key_var)
        
        # if not self.api_key:
        #     logger.warning(f"API key environment variable '{api_key_var}' not found. Generation will fail.")
        
        # Initialize Clients
        if self.provider == 'openai':
            self.client = openai.OpenAI(api_key=self.api_key)
            logger.info(f"Initialized OpenAI generator with model: {self.model_name}")
        elif self.provider == 'google':
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            logger.info(f"Initialized Google Gemini generator with model: {self.model_name}")
        else:
            logger.error(f"Unsupported LLM provider: {self.provider}")

    def _construct_prompt(self, query: str, retrieved_results: List[Dict[str, Any]]) -> str:
        """
        Constructs a prompt including metadata (filenames) for better citations.
        Expects 'retrieved_results' to be the list of dicts returned by the pipeline.
        """
        context_parts = []
        
        for i, result in enumerate(retrieved_results, 1):
            # Extract useful metadata
            meta = result.get('metadata', {})
            source = meta.get('file_path') or meta.get('source_file') or result.get('source', 'Unknown Source')
            raw_content = result.get('content', '')
            
            if isinstance(raw_content, list):
                # If content is a list, join it with newlines
                content = "\n".join(str(item) for item in raw_content).strip()
            else:
                # Otherwise, just convert to string and strip
                content = str(raw_content).strip()
            
            # Create a structured block for this chunk
            context_parts.append(f"Source {i} ({source}):\n{content}")
        
        context_str = "\n\n".join(context_parts)
        
        prompt = f"""You are a technical assistant for the PyTorch Lightning library. 
Use the provided Context below to answer the User Question.

Instructions:
1. Base your answer ONLY on the context provided.
2. If the context contains code snippets, strictly preserve the indentation and syntax.
3. If the answer is not in the context, politely state that you don't have enough information.
4. Refer to the 'Source' names provided in the context when possible.

Context:
{context_str}

User Question: 
{query}

Answer:"""
        return prompt

    def generate_answer(self, query: str, retrieved_results: List[Dict[str, Any]]) -> str:
        """Generates an answer using the configured LLM."""
        if not self.api_key:
            return "Error: API key not configured."

        prompt = self._construct_prompt(query, retrieved_results)

        try:
            if self.provider == 'openai':
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful RAG assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3 # Lower temperature for more factual answers
                )
                return response.choices[0].message.content.strip()

            elif self.provider == 'google':
                response = self.model.generate_content(prompt)
                return response.text
                
        except Exception as e:
            logger.error(f"LLM Generation failed: {e}")
            return f"Error producing answer: {str(e)}"
        
        return "Provider logic flow error."