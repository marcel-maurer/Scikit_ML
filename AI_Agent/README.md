# Brunno Local AI Agent

Brunno is an experimental local AI/LLM project built around Python, llama.cpp and GGUF models. The project explores how local language and vision models can be initialized, served, configured and connected to simple agent tooling and persistent context.

## Current capabilities

- Local GGUF inference through `llama-cpp-python`
- llama.cpp server startup through Python subprocesses
- Configurable GPU layer offloading
- Automatic retry strategy when CUDA/VRAM or context creation fails
- Qwen text-model configuration
- Qwen-VL / multimodal server configuration with an mmproj adapter
- OpenAI-compatible local chat endpoint integration
- External personality and memory context loading
- Simple memory-tag extraction
- Helper-tool and JSON loading
- Linux desktop integration for opening local endpoints

## Structure

```text
AI_Agent/
├── AgentInits/
│   ├── __init__.py
│   ├── AgentConfigs.py
│   ├── Agent_Presets.py
│   ├── Agent_tools.py
│   ├── external_subprocess_run.py
│   └── init_model.py
└── README.md
```

## Configuration

Large model files and personal context files are intentionally **not** included in this public repository.

The portfolio version supports environment-based local paths:

```bash
export BRUNNO_HOME="$HOME/Brunno"
export LLAMA_CPP_HOME="$HOME/llama.cpp"
```

An optional Odysseus admin password can be supplied through the environment rather than committed to source control:

```bash
export ODYSSEUS_ADMIN_PASSWORD="..."
```

Expected local model/context files are configured in `AgentConfigs.py`. GGUF weights, private user information, personality/memory data and credentials should remain local.

## Development status

This is an active experimental project rather than a finished production framework. Some modules preserve parts of the original development structure while the project is being refactored toward cleaner configuration, agent/tool separation and reusable local-model infrastructure.

The project is included in this portfolio because it demonstrates practical work beyond classical machine learning: local LLM inference, GPU/VRAM constraints, multimodal model serving, process management and application-level AI integration.
