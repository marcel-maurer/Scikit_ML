import os
import torch as T
from pathlib import Path

class AgentConfigs():

    def __init__(self):

        self.device: T.device = T.device(
            "cuda:0" if T.cuda.is_available() else "cpu"
        )

        self.brunno_home = Path(os.getenv("BRUNNO_HOME", Path.home() / "Brunno"))
        self.main_dir = self.brunno_home / "Qwen"

        self.qwen3_8B_Q4_K_M_model = self.main_dir / "Qwen3-8B-Q4_K_M.gguf"

        self.Stheno_model_path = self.brunno_home / "uncModels" / "L3-8B-Stheno-v3.1.Q4_K_M.gguf"

        self.tool_path = self.main_dir / "tools.json"

        self.vision_model_path = self.main_dir / "VisionModel" / "Qwen3VL-4B-Instruct-Q4_K_M.gguf"

        self.dim_adapter_path = self.main_dir / "VisionModel" / "mmproj-Qwen3VL-4B-Instruct-F16.gguf"

        self.search_query_keys = {
            "txt": ".txt",
            "json": ".json",
        }

        self.personality_path = self.main_dir / "personalityAiri2.txt"

        self.qwen_memory_save = self.main_dir / "QWEN_DATA_INFO.json"

        self.custom_user_info = self.main_dir / "CustomUserInfo.json"

        self.sensible_custom_user_info = self.main_dir / "CustomUserSensibleINfos.json"

        self.lamma_gpu_path = Path(os.getenv("LLAMA_CPP_HOME", Path.home() / "llama.cpp"))
        
        self.linkinfos = [

            # Qwen3-VL-4B
            "https://huggingface.co/mradermacher/Qwen3-VL-4B-Instruct-Uncensored-abliterated-GGUF",
            "https://huggingface.co/dummy9996/Qwen3-VL-4B-Instruct-Uncensored-abliterated-gguf",

            # Qwen3-VL-8B
            "https://huggingface.co/HauhauCS/Qwen3VL-8B-Uncensored-HauhauCS-Aggressive",
            "https://huggingface.co/HauhauCS/Qwen3VL-8B-Uncensored-HauhauCS-Balanced",
            "https://huggingface.co/tripolskypetr/Qwen3VL-Uncensored-Aggressive-GGUF",

            # LFM2.5-VL-3B
            "https://huggingface.co/SC117/LFM2.5-VL-3B-Uncensored-GGUF",

            # Gemma 4 26B
            "https://huggingface.co/Jommarn/UNSEEN_Gemma_4_26B_NSFW-GGUF",

            # Qwen3.6-27B
            "https://huggingface.co/marafx2025/Qwen3.6-27B-Abliterated-Heretic-Uncensored-GGUF",
        ]

        self.firefox_path = "/usr/bin/firefox"
        self.oddyseuspwrd = os.getenv("ODYSSEUS_ADMIN_PASSWORD") 
        self.base_model_presets = None
        self.other_presets = None


        self.vision_port = "9001"
        self.base_port = "8080"
        self.localhost_string = f"http://localhost:"

        self.start_base_model = False
        self.debug = False
        self.start_llam_server = True 
        self.use_vision_model = False
        self.open_dir = False
        self.download_new_model = None
        self.temp_dir = None

        self.quant_8Bil_param_qwen_path = self.qwen3_8B_Q4_K_M_model
        self.stheno_model_path = self.Stheno_model_path
        self.vision_model_path = self.vision_model_path

        self.context_layer  = 4048 #4048 2048 1048 512
        self.gpu_layer_usage = 35
        self.verbose = False
        self.init_model = None
        self.agent_tools = None

       # self.tool_functions = {
        #    "iterate_dirs": self.qwen_tools.iterate_dirs,
        #    "load_text_data": self.qwen_tools.load_text_data
        #}

        self.infos()

    def infos(self):
        for i in vars(self).items():
            print(i)

    def load_text_data(self, path=None):
        if path is not None:
            with open(path, mode="r", encoding="utf-8") as content_file:
                text_file = content_file.read()
                return text_file
            
    def external_data(self):

        personality = self.load_text_data(path=self.personality_path)
        qwen_memory = self.load_text_data(path=self.qwen_memory_save)
        user_info = self.load_text_data(path=self.custom_user_info)
        sensible_user_info = self.load_text_data(path=self.sensible_custom_user_info)

      #  for file in files:

           #  for file_type, extension in search_query_keys.items():

               #  if file.suffix == extension:
                   #  pass
                    #print(
                       # f"Datei: {file.name} | "
                       # f"Typ: {file_type} | "
                       # f"Endung: {extension}"
                   # )

        return personality , qwen_memory , user_info , sensible_user_info

    def main(self):

        print("=== Brunno Agent ===")
        print()

        # --------------------------------------------------
        # Base LLM direkt mit llama-cpp laden
        # --------------------------------------------------

        if self.start_base_model:
            print("-> Starting Base Model")
            self.init_model.init_base_model()
            self.init_model.chat_loop()

            #self.init_model.test_loop()
        # --------------------------------------------------
        # Base LLM als llama-server starten
        # --------------------------------------------------

        if self.start_llam_server:

            print("-> Starting llama-server")
            print("[0] Start Server")
            print("[1] Start Vision Server")

            answer = input("Choose Option: ")

            if answer == "0":
                self.presets.start_QwenQ8P_server(model_path = self.quant_8Bil_param_qwen_path  , port = self.base_port)
                self.presets.open_localhost(local_host = f"{self.localhost_string}:{self.base_port}" )
        
            if answer == "1":
                self.presets.start_vision_server(vision_model_path = self.vision_model_path , 
                                                 dim_adapter_path = self.dim_adapter_path,
                                                 port = self.vision_port)
                self.presets.open_localhost(local_host = f"{self.localhost_string}:{self.vision_port}" )
            else:
                print("Invalid option.")

        # --------------------------------------------------
        # Neues Modell herunterladen
        # --------------------------------------------------

        if self.download_new_model is not None:
            print("-> Downloading Model")

            self.init_model.download(
                self.download_new_model
            )

        # --------------------------------------------------
        # Verzeichnis öffnen
        # --------------------------------------------------

        if self.open_dir:
            print("-> Opening Model Directory")

            self.presets.open_localhost()

        print("=== Configuration finished ===")
