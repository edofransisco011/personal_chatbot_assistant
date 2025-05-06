# Personal Chatbot Assistant

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.8+-brightgreen.svg)

## Project Overview

The Personal Chatbot Assistant is a flexible, extensible Python application that provides a command-line interface for interacting with advanced large language models (LLMs). It currently supports OpenAI-compatible APIs, with a focus on the Llama family of models through NVIDIA's API integration.

### Key Features

- **Profile-Based Interactions**: Switch between specialized interaction modes (coding, math, writing, default)
- **Conversation Management**: Save, load, and review conversation history
- **Smart Context Handling**: Auto-detection of appropriate profiles based on user queries
- **Command Interface**: Rich set of commands for system control and configuration
- **Usage Analytics**: Track token usage, requests, and performance

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Key Components](#key-components)
- [Usage Examples](#usage-examples)
- [Command Reference](#command-reference)
- [Configuration](#configuration)
- [For Developers](#for-developers)
- [License](#license)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/personal_chatbot_assistant.git
cd personal_chatbot_assistant

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API credentials
```

### Requirements

- Python 3.8+
- OpenAI Python client
- python-dotenv

## Quick Start

1. Configure your `.env` file with the appropriate API credentials:

```
LLAMA_API_KEY=your_api_key_here
LLAMA_API_BASE_URL=https://integrate.api.nvidia.com/v1
LLAMA_MODEL_NAME=meta/llama-3.1-405b-instruct
CONVERSATION_SAVE_DIR=conversations
```

2. Run the chatbot:

```bash
python chatbot.py
```

3. Start chatting or use commands (type `/help` to see available commands)

## Architecture

The Personal Chatbot Assistant follows a modular design with two main classes:

1. **LlamaChatbot**: Core class that handles model interactions, conversation management, and profile configurations
2. **CommandHandler**: Processes and executes user commands, interfacing with the chatbot functionality

This separation of concerns creates a maintainable codebase that can be extended with new features and capabilities.

## Key Components

### LlamaChatbot Class

The primary class managing interactions with the language model API.

#### Features:

- **Multi-Profile Support**: Preconfigured expert profiles for coding, math, and writing tasks
- **Conversation Management**: Maintains, saves, and loads conversation history
- **Dynamic Configuration**: Adjusts parameters based on task context
- **Usage Tracking**: Monitors token usage and request metrics

### CommandHandler Class

Provides a command-line interface for controlling the chatbot.

#### Features:

- **Command Parsing**: Interprets user commands and routes to appropriate functions
- **Help System**: Built-in documentation for available commands
- **Conversation Controls**: Commands for saving, loading, and managing chats

## Usage Examples

### General Conversation

```
You: Tell me about quantum computing
Chatbot [default]: Quantum computing is a type of computation that harnesses quantum mechanical phenomena...
```

### Switching Profiles

```
You: /profile coding
Chatbot: Profile set to 'coding' and conversation reset.

You: Write a function to calculate Fibonacci numbers
Chatbot [coding]: Here's an efficient implementation of the Fibonacci sequence using dynamic programming...
```

### Saving Conversations

```
You: /save my_quantum_chat
Chatbot: Conversation saved to: conversations/my_quantum_chat.json
```

## Command Reference

| Command | Description | Example |
|---------|-------------|---------|
| `/help` | Display available commands | `/help` |
| `/exit` | Exit the chatbot | `/exit` |
| `/clear` | Clear conversation history | `/clear` |
| `/save [filename]` | Save conversation | `/save quantum_discussion` |
| `/load <filename>` | Load a saved conversation | `/load quantum_discussion.json` |
| `/list` | List saved conversations | `/list` |
| `/history` | Show current conversation | `/history` |
| `/system <prompt>` | Change the system prompt | `/system You are a helpful financial advisor` |
| `/profile <name>` | Switch profile | `/profile coding` |
| `/profiles` | List available profiles | `/profiles` |
| `/rate <1-5>` | Rate the last response | `/rate 5` |
| `/stats` | Show usage statistics | `/stats` |

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLAMA_API_KEY` | API key for authentication | None (Required) |
| `LLAMA_API_BASE_URL` | Base URL for API requests | https://integrate.api.nvidia.com/v1 |
| `LLAMA_MODEL_NAME` | Model to use for responses | meta/llama-3.1-405b-instruct |
| `CONVERSATION_SAVE_DIR` | Directory for saved conversations | conversations |

### Profile Configurations

Each profile has specific parameter settings optimized for different tasks:

| Profile | Temperature | Max Tokens | Top P | Use Case |
|---------|-------------|------------|-------|----------|
| default | 0.7 | 1024 | 1.0 | General questions |
| coding | 0.2 | 2048 | 0.95 | Programming assistance |
| math | 0.3 | 1024 | 0.9 | Mathematical problems |
| writing | 0.8 | 1500 | 1.0 | Content creation |

## For Developers

### Project Structure

```
personal_chatbot_assistant/
├── chatbot.py         # Main application code
├── requirements.txt   # Dependencies
├── .env.example       # Example environment configuration
├── conversations/     # Saved conversation files
└── logs/              # Application logs
```

### Extending the Chatbot

#### Adding New Profiles

To add a new expert profile:

1. Extend the `prompt_profiles` dictionary in the `LlamaChatbot` class.
2. Add corresponding configuration in the `profile_configs` dictionary.
3. Optionally, update the `auto_detect_profile` method to recognize the new profile type.

#### Adding New Commands

To add new commands:

1. Add a new entry to the `commands` dictionary in the `CommandHandler` class initialization.
2. Implement the corresponding command function following the pattern of existing commands.

### Logging

The application uses Python's built-in logging module with both file and console handlers. Logs are stored in the `logs/` directory.

## License

This project is licensed under the MIT License.