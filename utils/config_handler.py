import yaml
from utils.path_tool import get_abs_path


"""
加载yml文件
"""
def log_rag_config(
    config_path:str=get_abs_path("config/rag.yml"),
    encoding="utf-8",
):
    with open(config_path,"r",encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def log_prompts_config(
        config_path:str=get_abs_path("config/prompts.yml"),
        encoding="utf-8",
):
    with open(config_path,"r",encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def log_chroma_config(
        config_path=get_abs_path("config/chroma.yml"),
        encoding="utf-8",
):
    with open(config_path,"r",encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

rag_config=log_rag_config()
prompts_config=log_prompts_config()
chroma_config=log_chroma_config()
