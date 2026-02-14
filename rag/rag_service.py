
from model.factory import chat_model_factory, embeddings_factory
from langchain_core.prompts import PromptTemplate
from vector_store import VectorStoreService
from utils.load_prompts import load_kcal_prompts,load_version_prompts
class NutritionRAGService:
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.retriever = self.vector_store.get_retriever()
        self.model=chat_model_factory()
        self.prompt_version_text=load_version_prompts()
        self.prompt_kcal_text = load_kcal_prompts()
        # TODO
        self.prompt_template=PromptTemplate.from_template(self.prompt_version_text)

        self.chain=self.__init_chain()


    def __init_chain(self):
        pass


