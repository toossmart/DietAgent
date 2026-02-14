import hashlib
import os.path
from langchain_core.documents import Document
from utils.logger_handler import logger
from langchain_community.document_loaders import PyPDFLoader, TextLoader, JSONLoader


def get_file_md5_hex(filepath: str):
    # 1. 检查文件是否存在
    if not os.path.exists(filepath):
        logger.error(f"[md5计算]文件{filepath}不存在")
        return None

    # 2. 检查是否为文件
    if not os.path.isfile(filepath):
        logger.error(f"[md5计算]路径{filepath}不是文件")
        return None

    md5_obj = hashlib.md5()
    chunk_size = 4096

    try:
        with open(filepath, 'rb') as f:
            # 循环读取文件内容
            while chunk := f.read(chunk_size):
                md5_obj.update(chunk)

        return md5_obj.hexdigest()

    except Exception as e:
        logger.error(f"计算文件{filepath}md5失败，{str(e)}")
        return None


def list_dir_with_allowed_type(path: str, allowed_types: tuple[str]):
    files = []
    if not os.path.isdir(path):
        logger.error(f"[list_dir_with_allowed_type]{path}不是文件夹")
        return ()  # 修正：返回空元组，而不是返回后缀名类型

    for f in os.listdir(path):
        if f.endswith(allowed_types):
            files.append(os.path.join(path, f))

    return tuple(files)


def pdf_loader(filepath: str, password=None) -> list[Document]:
    # 这里的 password 参数如果不传默认是 None，PyPDFLoader 可以处理
    return PyPDFLoader(filepath, password=password).load()


def txt_loader(filepath: str) -> list[Document]:
    # 强制指定 utf-8 编码
    return TextLoader(filepath, encoding="utf-8").load()

def json_loader(filepath: str) -> list[Document]:
    return  JSONLoader(filepath,jq_schema='.[]|tostring').load()