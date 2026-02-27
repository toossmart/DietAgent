from abc import ABC, abstractmethod
from typing import Optional

from langchain_community.embeddings import DashScopeEmbeddings

from utils.config_handler import rag_config
from langchain_community.chat_models import ChatTongyi
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self)->Optional[Embeddings|BaseChatModel]:
        pass

class ChatModelFactory(BaseModelFactory):
    def generator(self,chat_model_name)->Optional[Embeddings|BaseChatModel]:
        return ChatTongyi(model=chat_model_name)

class VersionModelFactory(BaseModelFactory):
    def generator(self,version_model_name)->Optional[Embeddings|BaseChatModel]:
        return ChatTongyi(model=version_model_name)

class EmbeddingsFactory(BaseModelFactory):
    def generator(self,embedding_model_name)->Optional[Embeddings|BaseChatModel]:
        return DashScopeEmbeddings(model=embedding_model_name)



chat_model_factory_version =VersionModelFactory().generator(rag_config['chat_model_factory_version'])

chat_model_factory_kcal=ChatModelFactory().generator(rag_config['chat_model_factory_kcal'])

embeddings_factory = EmbeddingsFactory().generator(rag_config['embedding_model_name'])