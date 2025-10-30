import asyncio
import os
import json
import logging
from typing import Dict, List, Optional
from aiman.client._ai_man_client import AimanClient
from aiman.core.classes import PromptOptions, AIModel
from dataclasses import dataclass, asdict
from typing import Dict, Optional, List
from datetime import datetime

# Import ollama for local LLM support
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logging.warning("Ollama package not available. Local LLM support disabled.")

# Setup logging
log_path = "logs/model_switch.log"
os.makedirs(os.path.dirname(log_path), exist_ok=True)
logging.basicConfig(filename=log_path, level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class AsyncLLMTokenManager:
    """
    Manages multiple AI model agents asynchronously, handling model switching and conversation history.
    Supports both remote API models and local Ollama models.
    """
    def __init__(self, config_path: str = "src/einhorn/config.json"):
        """
        Initialize the manager with configuration.
        
        Args:
            config_path (str): Path to the configuration file
        """
        with open(config_path, "r") as f:
            config = json.load(f)

        self.client = AimanClient(
            host_url=config["api_url"],
            user_name=config["user_name"],
            password=config["password"]
        )

        self.models: List[AIModel] = self.client.get_models()
        self.agent_registry: Dict[str, Dict] = {}  # token -> {"model_tag_id": int, "messages": List[Dict]}
        self.active_model_tag_id: Optional[int] = None
        
        # Local LLM configuration: map negative IDs to Ollama models
        self.local_models: Dict[int, str] = {
            -1: "llama3.2:latest",
            -2: "llama3.1:8b",
            -3: "mistral-nemo:12b",
            -4: "gpt-oss:20b"
        }

    def _get_model_tag_id(self, model_id: int) -> int:
        """Get the model tag ID for a given model ID."""
        # Handle local model special case
        if model_id in self.local_models:
            return model_id
        
        for model in self.models:
            if model.id == model_id:
                return model.default_model_tag_id
        raise ValueError(f"No model found with id {model_id}")

    async def init_agent(self, model_id: int) -> str:
        """
        Initialize a new agent with the specified model ID.
        Supports local Ollama models via model_id = -1.
        
        Args:
            model_id (int): The ID of the model to use (-1 for local Ollama model)
            
        Returns:
            str: A unique token for the agent
        """
        # Check if local model is requested but Ollama is not available
        if model_id in self.local_models and not OLLAMA_AVAILABLE:
            raise RuntimeError("Ollama package not installed. Install with: pip install ollama")
        
        token = f"agent-{len(self.agent_registry) + 1}"
        model_tag_id = self._get_model_tag_id(model_id)
        self.agent_registry[token] = {
            "model_tag_id": model_tag_id,
            "messages": [],
            "is_local": model_id in self.local_models
        }
        
        if model_id in self.local_models:
            logging.info(f"[INIT] Agent {token} initialized with local Ollama model: {self.local_models[model_id]}")
        else:
            logging.info(f"[INIT] Agent {token} initialized with model_tag_id {model_tag_id}")
        
        return token

    async def ask_agent(self, token: str, prompt: str) -> str:
        """
        Send a prompt to an agent and get the response.
        Supports both remote API models and local Ollama models.
        
        Args:
            token (str): The agent's token
            prompt (str): The prompt to send
            
        Returns:
            str: The model's response
            
        Raises:
            ValueError: If token is invalid or prompt is None
            TypeError: If prompt is not a string
        """
        if token not in self.agent_registry:
            raise ValueError(f"Invalid token: {token}")
            
        if prompt is None:
            raise TypeError("Prompt cannot be None")
            
        if not isinstance(prompt, str):
            raise TypeError(f"Prompt must be a string, got {type(prompt)}")
            
        agent = self.agent_registry[token]
        model_tag_id = agent["model_tag_id"]
        is_local = agent.get("is_local", False)

        # Check if this is a local Ollama model
        if is_local:
            return await self._query_local_model(agent, prompt)
        else:
            return await self._query_remote_model(agent, model_tag_id, prompt)
    
    async def _query_local_model(self, agent: Dict, prompt: str) -> str:
        """
        Query a local Ollama model.
        
        Args:
            agent: The agent dictionary
            prompt: The prompt to send
            
        Returns:
            str: The model's response
        """
        # Update conversation history
        agent["messages"].append({"role": "user", "message": prompt})
        
        # Prepare messages for Ollama (convert to Ollama format)
        ollama_messages = [
            {"role": msg["role"], "content": msg["message"]} 
            for msg in agent["messages"]
        ]
        
        try:
            # Query Ollama asynchronously
            model_name = self.local_models.get(agent["model_tag_id"], self.local_models.get(-1))
            response = await asyncio.to_thread(
                ollama.chat,
                model=model_name,
                messages=ollama_messages,
                options={"temperature": 0}
            )
            
            result = response["message"]["content"]
            agent["messages"].append({"role": "assistant", "message": result})
            logging.info(f"[LOCAL] Successfully queried Ollama model: {model_name}")
            
            return result
            
        except Exception as e:
            logging.error(f"[LOCAL ERROR] Failed to query Ollama model: {e}")
            model_name = self.local_models.get(agent.get("model_tag_id"), "unknown")
            raise RuntimeError(f"Failed to query local Ollama model '{model_name}': {e}")
    
    async def _query_remote_model(self, agent: Dict, model_tag_id: int, prompt: str) -> str:
        """
        Query a remote API model.
        
        Args:
            agent: The agent dictionary
            model_tag_id: The model tag ID
            prompt: The prompt to send
            
        Returns:
            str: The model's response
        """
        # Handle model switching if needed
        if self.active_model_tag_id != model_tag_id:
            logging.info(f"[MODEL SWITCH] Switching from model {self.active_model_tag_id} to {model_tag_id}")
            self.active_model_tag_id = model_tag_id
            await asyncio.sleep(1)  # Let the server settle

        # Update conversation history
        agent["messages"].append({"role": "user", "message": prompt})
        conversation = "\n".join([f"{m['role']}: {m['message']}" for m in agent["messages"]])

        # Configure prompt options
        options = PromptOptions()
        options.keep_context = False
        options.temperature = 0

        # Try to get response with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.prompt(
                    model_tag_id=model_tag_id,
                    query=conversation,
                    prompt_options=options
                )
                result = response["responseText"]
                agent["messages"].append({"role": "assistant", "message": result})
                return result
            except Exception as e:
                logging.warning(f"[RETRY {attempt+1}] Failed querying model {model_tag_id}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1.5 + attempt)
                else:
                    raise RuntimeError(f"Failed to query model {model_tag_id} after {max_retries} retries: {e}")

    async def shutdown_agent(self, token: str) -> None:
        """
        Shut down a specific agent.
        
        Args:
            token (str): The agent's token
        """
        if token in self.agent_registry:
            logging.info(f"[SHUTDOWN] Agent {token} removed")
            del self.agent_registry[token]

    async def shutdown_all(self) -> None:
        """Shut down all active agents."""
        logging.info(f"[SHUTDOWN] Shutting down all agents: {list(self.agent_registry.keys())}")
        self.agent_registry.clear()
        self.active_model_tag_id = None
