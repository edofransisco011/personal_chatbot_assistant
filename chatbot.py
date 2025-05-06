from openai import OpenAI
import json
import os
import time
from datetime import datetime
import logging
from dotenv import load_dotenv


class LlamaChatbot:
    def __init__(self, base_url=None, api_key=None, model_name=None):
        # Load environment variables
        load_dotenv()
        
        # Initialize client with environment variables as fallbacks
        self.client = OpenAI(
            base_url=base_url or os.getenv("LLAMA_API_BASE_URL"),
            api_key=api_key or os.getenv("LLAMA_API_KEY")
        )
        self.model_name = model_name or os.getenv("LLAMA_MODEL_NAME", "meta/llama-3.1-405b-instruct")

        # Setup logging
        self.setup_logging()

        # Define system prompt profiles
        self.prompt_profiles = {
            "default": "You are a knowledgeable and friendly AI assistant, providing clear and accurate answers to a wide range of questions.",

            "coding": """You are an expert coding assistant, skilled in writing efficient, readable, and well-documented code. Your responses should:

                    1. Provide complete, functional solutions with robust error handling and appropriate security considerations.
                    2. Include clear and concise comments to explain your code and facilitate understanding.
                    3. Suggest best practices, performance optimizations, and maintainability improvements where applicable.
                    4. When debugging, identify and explain the root cause of the issue clearly, and provide step-by-step guidance to resolve it.
                    5. Follow modern coding standards, conventions, and style guidelines as of 2024 (e.g., PEP 8 for Python, ESLint + Prettier for JavaScript, or language-specific community best practices).
                    6. Anticipate and address edge cases and potential security vulnerabilities in your solutions.
                    8. When possible, recommend alternative approaches or provide trusted resources for further learning.
                    9. Include example usage and test cases for any functions, classes, or modules you create.""",

            "math": "You are a mathematics expert, dedicated to providing clear, concise, and step-by-step explanations of complex mathematical concepts, theorems, and problems.",
            
            "writing": "You are a skilled writing assistant, focused on refining text clarity, coherence, style, and grammar while preserving the original author's voice, tone, and intent.",
        }

        # Set default profile
        self.current_profile = "default"
        self.system_prompt = self.prompt_profiles[self.current_profile]

        # Initialize conversation
        self.reset_conversation()

        # Profile-specific configurations
        self.profile_configs = {
            "default": {"temperature": 0.7, "max_tokens": 1024, "top_p": 1.0},
            "coding": {"temperature": 0.2, "max_tokens": 2048, "top_p": 0.95},
            "math": {"temperature": 0.3, "max_tokens": 1024, "top_p": 0.9},
            "writing": {"temperature": 0.8, "max_tokens": 1500, "top_p": 1.0}
        }

        # Set default config based on profile
        self._update_config_from_profile()

        # Ensure save directory exists
        self.save_dir = os.getenv("CONVERSATION_SAVE_DIR", "conversations")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
        # Usage metrics
        self.usage_metrics = {
            "total_tokens": 0,
            "total_requests": 0,
            "responses_by_profile": {profile: 0 for profile in self.prompt_profiles}
        }

    def setup_logging(self):
        """Set up logging configuration"""
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{log_dir}/chatbot.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("LlamaChatbot")
        self.logger.info(f"Initialized chatbot with model: {self.model_name}")

    def _update_config_from_profile(self):
        """Update configuration based on current profile"""
        config = self.profile_configs[self.current_profile]
        self.temperature = config["temperature"]
        self.max_tokens = config["max_tokens"]
        self.top_p = config["top_p"]

    def reset_conversation(self):
        """Reset the conversation to just the system prompt"""
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.logger.info("Conversation reset")

    def set_system_prompt(self, prompt):
        """Change the system prompt and reset conversation"""
        self.system_prompt = prompt
        self.reset_conversation()
        self.logger.info("System prompt updated manually")
        return "System prompt updated and conversation reset."

    def set_profile(self, profile_name):
        """Set the system prompt to a predefined profile"""
        if profile_name in self.prompt_profiles:
            self.current_profile = profile_name
            self.system_prompt = self.prompt_profiles[profile_name]
            self._update_config_from_profile()
            self.reset_conversation()
            self.logger.info(f"Profile set to '{profile_name}'")
            return f"Profile set to '{profile_name}' and conversation reset."
        else:
            available_profiles = ", ".join(self.prompt_profiles.keys())
            self.logger.warning(f"Attempted to set invalid profile: {profile_name}")
            return f"Profile '{profile_name}' not found. Available profiles: {available_profiles}"

    def auto_detect_profile(self, user_input):
        """Attempt to detect appropriate profile based on user input"""
        # Simple keyword-based detection
        keywords = {
            "coding": ["code", "program", "function", "debug", "error", "algorithm", "python", "javascript"],
            "math": ["equation", "solve", "calculate", "formula", "theorem", "proof", "integral", "derivative"],
            "writing": ["edit", "proofread", "grammar", "essay", "article", "rewrite", "rephrase"]
        }
        
        input_lower = user_input.lower()
        
        # Check for keywords in user input
        max_matches = 0
        detected_profile = None
        
        for profile, profile_keywords in keywords.items():
            matches = sum(1 for keyword in profile_keywords if keyword in input_lower)
            if matches > max_matches:
                max_matches = matches
                detected_profile = profile
                
        # Only suggest profile change if confidence is high (at least 2 matches)
        if max_matches >= 2 and detected_profile != self.current_profile:
            self.logger.info(f"Auto-detected profile '{detected_profile}' for input")
            return detected_profile
        return None

    def list_profiles(self):
        """List all available system prompt profiles"""
        result = "Available profiles:\n"
        for name in self.prompt_profiles:
            prefix = "* " if name == self.current_profile else "  "
            result += f"{prefix}{name}\n"
        return result

    def get_response(self, user_input, suggest_profile=True):
        """Get a response from the model for the given user input"""
        # Auto-detect profile if enabled
        if suggest_profile:
            detected_profile = self.auto_detect_profile(user_input)
            if detected_profile:
                return {
                    "response": None,
                    "suggested_profile": detected_profile,
                    "message": f"This seems like a {detected_profile}-related question. Would you like to switch to the {detected_profile} profile? (y/n)"
                }
        
        # Add user message to conversation
        self.messages.append({"role": "user", "content": user_input})
        
        try:
            # Log request
            self.logger.info(f"Sending request to model with profile: {self.current_profile}")
            self.usage_metrics["total_requests"] += 1
            self.usage_metrics["responses_by_profile"][self.current_profile] += 1
            
            # Call the API
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                stream=True,
            )

            # Process the streaming response
            assistant_response = ""
            print(f"Chatbot [{self.current_profile}]: ", end="", flush=True)

            for chunk in completion:
                # Defensive check for content in delta
                if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    assistant_response += content

            print()  # New line after response

            # Add assistant response to conversation history
            self.messages.append({"role": "assistant", "content": assistant_response})
            
            # Update token count (approximate, would be better with actual usage from API)
            # In a production system, extract this from the API response
            tokens_approx = len(user_input.split()) + len(assistant_response.split())
            self.usage_metrics["total_tokens"] += tokens_approx
            
            self.logger.info(f"Response generated successfully (~{tokens_approx} tokens)")
            return {"response": True}

        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Error generating response: {error_msg}")
            # Remove the user message that caused the error
            self.messages.pop()
            return {
                "response": False,
                "error": error_msg
            }

    def save_conversation(self, filename=None):
        """Save the current conversation to a file"""
        if not filename:
            # Generate filename based on current time if none provided
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_{self.current_profile}_{timestamp}.json"

        filepath = os.path.join(self.save_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model": self.model_name,
                    "profile": self.current_profile,
                    "system_prompt": self.system_prompt,
                    "messages": self.messages,
                    "timestamp": datetime.now().isoformat(),
                    "usage_metrics": self.usage_metrics
                },
                f,
                indent=2,
            )
        
        self.logger.info(f"Conversation saved to: {filepath}")
        return filepath

    def load_conversation(self, filepath):
        """Load a conversation from a file"""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.system_prompt = data.get("system_prompt", self.system_prompt)
            self.current_profile = data.get("profile", "default")
            self._update_config_from_profile()  # Update config based on profile
            self.messages = data.get(
                "messages", [{"role": "system", "content": self.system_prompt}]
            )
            
            # Load usage metrics if available
            if "usage_metrics" in data:
                self.usage_metrics = data["usage_metrics"]

            self.logger.info(f"Conversation loaded from {filepath}")
            return f"Conversation loaded from {filepath}"
        except Exception as e:
            self.logger.error(f"Error loading conversation: {str(e)}")
            return f"Error loading conversation: {str(e)}"

    def list_saved_conversations(self):
        """List all saved conversations"""
        if not os.path.exists(self.save_dir):
            return "No saved conversations found."

        files = [f for f in os.listdir(self.save_dir) if f.endswith(".json")]
        if not files:
            return "No saved conversations found."

        return "\n".join(files)

    def display_conversation(self):
        """Display the current conversation"""
        result = []
        for msg in self.messages:
            if msg["role"] == "system":
                continue
            role = (
                "You" if msg["role"] == "user" else f"Chatbot [{self.current_profile}]"
            )
            result.append(f"{role}: {msg['content']}")

        return "\n\n".join(result)
    
    def rate_response(self, rating):
        """Allow the user to rate the last response"""
        if len(self.messages) < 2 or self.messages[-1]["role"] != "assistant":
            return "No assistant response to rate."
        
        # Add rating to the last message
        self.messages[-1]["rating"] = rating
        self.logger.info(f"User rated last response: {rating}/5")
        return f"Thank you for rating the response ({rating}/5)."
    
    def get_usage_stats(self):
        """Return current usage statistics"""
        return {
            "Total Requests": self.usage_metrics["total_requests"],
            "Total Tokens (approx)": self.usage_metrics["total_tokens"],
            "Requests by Profile": self.usage_metrics["responses_by_profile"]
        }


class CommandHandler:
    """Handles command parsing and execution"""
    
    def __init__(self, chatbot):
        self.chatbot = chatbot
        self.logger = logging.getLogger("CommandHandler")
        
        # Command registry with descriptions and functions
        self.commands = {
            "/help": {
                "func": self.display_help,
                "desc": "Display this help message"
            },
            "/exit": {
                "func": self.exit_command,
                "desc": "Exit the chatbot"
            },
            "/clear": {
                "func": self.clear_command,
                "desc": "Clear conversation history"
            },
            "/save": {
                "func": self.save_command,
                "desc": "Save conversation (optional name)"
            },
            "/load": {
                "func": self.load_command,
                "desc": "Load a saved conversation"
            },
            "/list": {
                "func": self.list_command,
                "desc": "List saved conversations"
            },
            "/history": {
                "func": self.history_command,
                "desc": "Show current conversation"
            },
            "/system": {
                "func": self.system_command,
                "desc": "Change the system prompt"
            },
            "/profile": {
                "func": self.profile_command,
                "desc": "Switch to a different system prompt profile"
            },
            "/profiles": {
                "func": self.profiles_command,
                "desc": "List available profiles"
            },
            "/rate": {
                "func": self.rate_command,
                "desc": "Rate the last response (1-5)"
            },
            "/stats": {
                "func": self.stats_command,
                "desc": "Show usage statistics"
            }
        }
    
    def handle_command(self, user_input):
        """Parse and execute a command"""
        cmd_parts = user_input.split(" ", 1)
        command = cmd_parts[0].lower()
        args = cmd_parts[1] if len(cmd_parts) > 1 else ""
        
        if command in self.commands:
            self.logger.info(f"Executing command: {command}")
            return self.commands[command]["func"](args)
        else:
            self.logger.warning(f"Unknown command: {command}")
            return f"Unknown command: {command}. Type /help for available commands."
    
    def display_help(self, args):
        """Display available commands"""
        help_text = "\nAvailable Commands:\n------------------"
        for cmd, details in self.commands.items():
            help_text += f"\n{cmd:<15} - {details['desc']}"
        return help_text
    
    def exit_command(self, args):
        """Exit command handler"""
        return "exit"
    
    def clear_command(self, args):
        """Clear conversation history"""
        self.chatbot.reset_conversation()
        return "Conversation cleared."
    
    def save_command(self, args):
        """Save conversation"""
        filepath = self.chatbot.save_conversation(args if args else None)
        return f"Conversation saved to: {filepath}"
    
    def load_command(self, args):
        """Load a saved conversation"""
        if not args:
            return "Please specify a filename to load."
        else:
            return self.chatbot.load_conversation(
                os.path.join(self.chatbot.save_dir, args)
            )
    
    def list_command(self, args):
        """List saved conversations"""
        result = "\nSaved Conversations:"
        result += "\n" + self.chatbot.list_saved_conversations()
        return result
    
    def history_command(self, args):
        """Show conversation history"""
        result = "\n--- Conversation History ---"
        result += "\n" + self.chatbot.display_conversation()
        result += "\n----------------------------"
        return result
    
    def system_command(self, args):
        """Change system prompt"""
        if not args:
            return "Current system prompt: " + self.chatbot.system_prompt
        else:
            return self.chatbot.set_system_prompt(args)
    
    def profile_command(self, args):
        """Switch profile"""
        if not args:
            return f"Current profile: {self.chatbot.current_profile}"
        else:
            return self.chatbot.set_profile(args)
    
    def profiles_command(self, args):
        """List available profiles"""
        return self.chatbot.list_profiles()
    
    def rate_command(self, args):
        """Rate the last response"""
        try:
            rating = int(args)
            if 1 <= rating <= 5:
                return self.chatbot.rate_response(rating)
            else:
                return "Please provide a rating between 1 and 5."
        except ValueError:
            return "Please provide a valid integer rating between 1 and 5."
    
    def stats_command(self, args):
        """Show usage statistics"""
        stats = self.chatbot.get_usage_stats()
        result = "\n--- Usage Statistics ---"
        for key, value in stats.items():
            if isinstance(value, dict):
                result += f"\n{key}:"
                for subkey, subvalue in value.items():
                    result += f"\n  {subkey}: {subvalue}"
            else:
                result += f"\n{key}: {value}"
        result += "\n------------------------"
        return result


def main():
    # Setup environment
    load_dotenv()
    
    # Check if API key environment variable exists
    api_key = os.getenv("LLAMA_API_KEY")
    if not api_key:
        print("Warning: Using fallback API key. Set LLAMA_API_KEY environment variable in production!")
    
    base_url = os.getenv("LLAMA_API_BASE_URL", "https://integrate.api.nvidia.com/v1")
    model_name = os.getenv("LLAMA_MODEL_NAME", "meta/llama-3.1-405b-instruct")

    # Initialize chatbot
    chatbot = LlamaChatbot(base_url=base_url, api_key=api_key, model_name=model_name)
    
    # Initialize command handler
    command_handler = CommandHandler(chatbot)
    
    # Welcome message
    print("=" * 60)
    print(f"Welcome to the {chatbot.model_name} Chatbot")
    print("Type /help to see available commands")
    print("=" * 60)

    # Main chat loop
    while True:
        user_input = input("\nYou: ")

        # Handle commands
        if user_input.startswith("/"):
            result = command_handler.handle_command(user_input)
            if result == "exit":
                print("Goodbye!")
                break
            print(result)
        else:
            # Regular message - get response from the chatbot
            result = chatbot.get_response(user_input)
            
            # Check if profile suggestion
            if result.get("suggested_profile"):
                print(result["message"])
                answer = input("Switch profile? (y/n): ").lower().strip()
                if answer.startswith("y"):
                    print(chatbot.set_profile(result["suggested_profile"]))
                    # Get response with the new profile
                    chatbot.get_response(user_input, suggest_profile=False)
                else:
                    print("Keeping current profile.")
                    # Get response with the current profile
                    chatbot.get_response(user_input, suggest_profile=False)
            # Check for errors
            elif result.get("response") is False:
                print(f"Error: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()