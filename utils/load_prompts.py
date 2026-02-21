import os
from utils.config_handler import prompts_config
from utils.path_tool import get_abs_path
from utils.logger_handler import logger

def _load_file_content(filepath: str) -> str:
    """读取文件内容的通用函数"""
    abs_path = get_abs_path(filepath)
    try:
        if not os.path.exists(abs_path):
            logger.error(f"Prompt文件不存在: {abs_path}")
            return ""
        with open(abs_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"加载Prompt文件失败 {abs_path}: {e}")
        return ""

def load_version_prompts():
    """加载视觉估算Prompt"""
    # 优先从配置读取，如果没有配置则使用默认路径
    path = prompts_config.get('version_path', 'prompts/version.txt')
    return _load_file_content(path)

def load_kcal_prompts():
    """加载热量计算Prompt"""
    path = prompts_config.get('kcal_prompt_path', 'prompts/kcal.txt')
    return _load_file_content(path)

def load_estimation_prompts():
    """加载文本估算Prompt"""
    # 假设你在 prompts.yml 里加了 estimation_prompt_path，如果没有就用默认的
    path = prompts_config.get('estimation_prompt_path', 'prompts/estimation.txt')
    return _load_file_content(path)

if __name__ == '__main__':
    # 测试加载
    print("Version Prompt:", len(load_version_prompts()))
    print("Kcal Prompt:", len(load_kcal_prompts()))
    print("Estimation Prompt:", len(load_estimation_prompts()))