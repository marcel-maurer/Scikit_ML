import json
import subprocess
import os

class AgentTools():

    def __init__(self, configs):

        self.configs = configs

    def load_helper_tools(self):

        with open(self.configs.tool_path, mode="r", encoding="utf-8") as tools_file:
            return json.load(tools_file)
        
    def iterate_dirs(self , iterate_path = None):
        if iterate_path is not None:
            for dirpath, dirnames, filenames in os.walk():
                print(f"dirpath:{dirpath}")
                print(f"dirnames:{dirnames}")
                print(f"filenames:{filenames}")

    def load_text_data(self, path=None):

        if path is not None:

            with open(path, mode="r", encoding="utf-8") as content_file:

                text_file = content_file.read()

                return text_file
            
    def write_json(self , save_path = None , arg = None , json_name : str = None):
        if save_path is None: 
            save_path = self.main_dir

            if json_name is None:
                json_name = "QWEN_DATA_INFO.json"

            fullpath = os.path.join(save_path, json_name)
            with open(fullpath, mode="w", encoding="utf-8") as new_json_file:
                json.dump(arg, new_json_file, indent=4, ensure_ascii=False)

    def save_memory(self, answer):

        if "<|MEMORY" not in answer:
            return

        start = answer.find("<|MEMORY")
        end = answer.find("|>", start)

        if end == -1:
            return

        memory_text = answer[start + len("<|MEMORY"):end]

        print("Memory gefunden:")
        print(memory_text)
        self.write_json(arg= memory_text)

    def open_directory(self , path:str = None):
        if path is not None:
            subprocess.Popen([
                "xdg-open",
                str(path)
            ])
            
    def load_text_files(self , path = None):

        if path is not None:

            with open(path, mode="r", encoding="utf-8") as content_file:

                text_file = content_file.read()

                return text_file
            
    def external_data(self):
        personality = self.load_text_files(path=self.configs.personality_path)
        qwen_memory = self.load_text_files(path=self.configs.qwen_memory_save)
        user_info = self.load_text_files(path=self.configs.custom_user_info)
        sensible_user_info = self.load_text_files(path=self.configs.sensible_custom_user_info)

        return personality, qwen_memory, user_info, sensible_user_info
