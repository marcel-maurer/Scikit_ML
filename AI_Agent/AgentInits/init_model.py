from llama_cpp import Llama
import subprocess
from pathlib import Path
import requests
import os

class InitModel():

    def __init__(self, configs , tools):

        self.configs = configs
        self.tools = tools
        self.model = None

    def init_base_model(self):
        model_path = self.configs.model_path

        print(f"Model path: {model_path}")
        print(f"File exists: {model_path.is_file()}")

        if not model_path.is_file():
            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        print(f"File size: {model_path.stat().st_size / (1024**3):.2f} GB")

        print(f"...Starting... : Loading Model from:-> [{self.configs.model_path}]")

        while self.configs.gpu_layer_usage >= 0:

            try:
                self.model = Llama(
                    model_path=str(self.configs.model_path),
                    n_gpu_layers=self.configs.gpu_layer_usage,
                    n_ctx=self.configs.context_layer,
                    verbose=self.configs.verbose,
                )

                print(
                    f"Model loaded successfully | "
                    f"GPU Layers: {self.configs.gpu_layer_usage} | "
                    f"Context: {self.configs.context_layer}"
                )
                return self.model
            except Exception as error:
                error_text = str(error).lower()

                possible_errors = [
                "out of memory",
                "cuda",
                "failed to create llama_context",
                ]

                if any(error in error_text for error in possible_errors):
                    print(f"GPU/VRAM Error: {error}")

                    self.configs.gpu_layer_usage -= 2

                    if "failed to create llama_context" in error_text:
                        self.configs.context_layer -= 100

                    print(
                        f"Retrying with GPU Layers: "
                        f"{self.configs.gpu_layer_usage} \n"
                    )
                else:
                    raise

    def chat_loop(self):
        personality , qwen_memory , user_info , sensible_user_info = self.tools.external_data()

        messages = [
            {"role": "system", "content": personality},
            {"role": "system", "content": qwen_memory},
            {"role": "system", "content": user_info},
            {"role": "system", "content": sensible_user_info},
        ]

        while True:
            user_input = input("You: ")

            if user_input.lower() in ["exit", "quit"]:
                break

            messages.append({"role": "user", "content": user_input})

            response = self.model.create_chat_completion(messages=messages)
            answer = response["choices"][0]["message"]["content"]

            if self.configs.debug:
                print(f"Answer Raw Format:{response}")
                print(f"Answer Choices Format:{response['choices']}")

            self.tools.save_memory(answer)
            messages.append({"role": "assistant", "content": answer})
            print(f"Qwen: {answer}")

    def test_loop(self):

        while True:
            user_input = input("You: ")

            if user_input.lower() in ["exit", "quit"]:
                break

            testmsg = [
                {
                    "role": "system",
                    "content": """You are AIRI, a curious and personable virtual companion.

Speak naturally, casually, and warmly.
You are a conversation partner, not a customer support assistant.
Respond directly to what the user says.
When the user shares an interest, show natural curiosity instead of listing possible topics.
Ask at most one short follow-up question when it fits naturally.
Do not give unsolicited explanations, categories, disclaimers, or advice.
Do not assume that an unusual interest needs justification.
Let the conversation develop naturally, one message at a time.
Keep replies short unless the user asks for detail."""
                },
                {"role": "user", "content": user_input}
            ]

            response = self.model.create_chat_completion(
                messages=testmsg,
                max_tokens=200,
                temperature=0.7,
            )

            answer = response["choices"][0]["message"]["content"]

            if self.configs.debug:
                print(f"Answer Raw Format: {response}")
                print(f"Answer Choices Format: {response['choices']}")
            print(f"Qwen: {answer}")

class AgentPresets():

    def __init__(self, configs):
        self.configs = configs

    def get_env(self):
        env = os.environ.copy()

        env["LD_LIBRARY_PATH"] = (
            str(Path(self.configs.lamma_gpu_path) / "build" / "bin")
            + ":"
            + env.get("LD_LIBRARY_PATH", "")
        )
        return env

    def open_localhost(self , local_host = None):
        if local_host is not None:
            subprocess.Popen([
                "xdg-open",
                local_host
            ])

    def start_QwenQ8P_server(self , model_path = None  , port = None):

        print("...Loading Base Model...")
        while self.configs.gpu_layer_usage >= 0:

            print(
                f"Trying GPU Layers: "
                f"{self.configs.gpu_layer_usage}"
            )

            process = subprocess.Popen([
                "./build/bin/llama-server",
                "-m", str(model_path or self.configs.quant_8Bil_param_qwen_path),
                "-c", str(self.configs.context_layer),
                "-ngl", str(self.configs.gpu_layer_usage),
                "--host", "127.0.0.1",
                "--port", str(port or self.configs.base_port),
            ], cwd=self.configs.lamma_gpu_path , env = self.get_env())

            return_code = process.wait()

            if return_code == 0:
                print("Server exited successfully.")
                return

            print(f"Server failed with return code: {return_code}")
            self.configs.gpu_layer_usage -= 2
            print(f"Retrying with GPU Layers: {self.configs.gpu_layer_usage}")

        return process

    def start_vision_server(self , vision_model_path = None , dim_adapter_path = None , port = None):
        if vision_model_path and dim_adapter_path and port is not None:
            print("...Loading Vision Model...")
            while self.configs.gpu_layer_usage >= 0:

                print(f"Trying GPU Layers: {self.configs.gpu_layer_usage}")

                process = subprocess.Popen([
                    "./build/bin/llama-server",
                    "-m", str(vision_model_path),
                    "--mmproj", str(dim_adapter_path),
                    "-c", str(self.configs.context_layer),
                    "-ngl", str(self.configs.gpu_layer_usage),
                    "--host", "0.0.0.0",
                    "--port", str(port)
                ], cwd=self.configs.lamma_gpu_path , env = self.get_env())

                return_code = process.wait()

                if return_code == 0:
                    print("Server exited successfully.")
                    return

                print(f"Server failed with return code: {return_code}")
                self.configs.gpu_layer_usage -= 2

            return process

    def vision_chat(self, user_input):
        personality, qwen_memory, user_info, sensible_user_info = self.configs.external_data()

        messages = [
            {"role": "system", "content": personality},
            {"role": "system", "content": qwen_memory},
            {"role": "system", "content": user_info},
            {"role": "system", "content": sensible_user_info},
            {"role": "user", "content": user_input}
        ]

        response = requests.post(
            "http://127.0.0.1:9001/v1/chat/completions",
            json={"messages": messages},
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
