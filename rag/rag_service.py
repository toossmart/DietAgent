from typing import List, final

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field
from model.factory import chat_model_factory_kcal,chat_model_factory_version, embeddings_factory
from langchain_core.prompts import PromptTemplate
from rag.vector_store import VectorStoreService
from utils.load_prompts import load_kcal_prompts, load_version_prompts, load_estimation_prompts


class DishItem(BaseModel):
    name: str = Field(description="菜品名称")
    weight_g: int = Field(description="估算或实际克重")
    is_estimated: bool = Field(description="克重是否为估算值")
class DishList(BaseModel):
    items: List[DishItem]
class AnalysisResult(BaseModel):
    items: List[DishItem]
    total_calories: int= Field(description="总卡路里(整数)")
    advice: str= Field(description="健康饮食建议")


import logging
logger = logging.getLogger(__name__)

class NutritionRAGService:
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.retriever = self.vector_store.get_retriever()


        self.version_model=chat_model_factory_version
        self.kcal_model=chat_model_factory_kcal

        #加载prompt,以能加入链的格式
        self.prompt_version_text=load_version_prompts()
        self.prompt_estimation_text=load_estimation_prompts()
        self.prompt_kcal_text = load_kcal_prompts()


        #初始化解析器
        self.estimation_parser = JsonOutputParser(pydantic_object=DishList)
        self.calculation_parser = JsonOutputParser(pydantic_object=AnalysisResult)

        self.__init_chains()


    def __init_chains(self):
        def build_vision_input(input_dict):
            """
            动态构建视觉模型的输入消息：
            SystemMessage = 加载的Prompt文本 + Parser格式指令
            HumanMessage = 图片URL + 引导文本
            """
            image_data = input_dict.get("image")

            # A. 获取解析器的格式指令 (这是保证输出能被 parse 的关键)
            # 它会生成类似: "The output should be formatted as a JSON instance that conforms to the JSON schema..."
            format_instructions = self.estimation_parser.get_format_instructions()

            system_content = (
                f"{self.prompt_version_text}\n\n"
                f"【IMPORTANT OUTPUT INSTRUCTIONS】\n"
                f"{format_instructions}\n"
                f"Ensure the output is pure JSON without Markdown code blocks."
            )

            # C. 返回消息列表
            return [
                SystemMessage(content=system_content),
                HumanMessage(content=[
                    # 文本用 type: text
                    {"type": "text", "text": "请分析这张图片中的食物，识别菜品名称并估算热量/克重。"},
                    # 图片用 type: image，并将 url 直接传给 image 字段
                    {"type": "image", "image": image_data}
                ])
            ]
        #图片估算链
        self.chain_version=(RunnableLambda(build_vision_input)|self.version_model|self.estimation_parser)

        #估算链
        estimation_prompt_template = PromptTemplate.from_template(self.prompt_estimation_text)
        self.chain_estimation = (estimation_prompt_template | self.kcal_model | self.estimation_parser)

        #卡路里估算链
        kcal_prompt_template = PromptTemplate.from_template(self.prompt_kcal_text)
        self.chain_kcal=(kcal_prompt_template|self.kcal_model|self.calculation_parser)


    #将用户提问根据json格式划分解析rag，只解析name字段,list[dict]由后端传回
    def retrieve_context(self,items:list[dict]):
        context_parts=[]
        for item in items:
            docs=self.retriever.invoke(item['name'])
            if docs:
                context_parts.append(f"[{item['name']}]参考数据: {docs[0].page_content}")

        return "\n".join(context_parts)


    def analyze(self,user_input=None,image_data=None):
        logger.info("=== analyze 函数被调用了 ===")

        if  image_data:
            print(">> 检测到图片，启用视觉估算模式...")
            estimated_data = self.chain_version.invoke({
                "image": image_data
            })
        elif user_input:
            print(f">> [Step 1] 启用文本估算模式: {user_input}")

            estimated_data = self.chain_estimation.invoke({
                "input": user_input,
                "format_instructions": self.estimation_parser.get_format_instructions()
            })
        else:
            return {"error": "未提供图片或文本输入"}
        if not estimated_data or not isinstance(estimated_data, dict):
            # 增加日志打印，方便排查模型到底返回了什么
            logger.error(f"Step 1 返回格式异常: {estimated_data}")
            return {"error": "食物识别失败，请尝试更清晰的描述"}

        items_list = estimated_data.get('items', [])
        if not items_list:
            return {"error": "未能识别出任何菜品，请重新输入"}

        # Step 2: RAG 检索
        rag_context = self.retrieve_context(items_list)

        # Step 3: 传参时确保 user_data 的结构是完整的 JSON 字符串或符合 Prompt 预期
        final_result = self.chain_kcal.invoke({
            "user_data": estimated_data,  # 确保这里是一个包含 items 的字典
            "context": rag_context if rag_context else "未找到参考数据，请基于常识估算。",
            "format_instructions": self.calculation_parser.get_format_instructions()
        })
        return final_result

if __name__ == '__main__':
    service = NutritionRAGService()

    # 测试 1: 纯文本输入
    print("\n" + "=" * 20 + " 测试文本模式 " + "=" * 20)
    input_text = "我中午吃了一份宫保鸡丁，还有一碗米饭"
    result = service.analyze(user_input=input_text)
    print("最终结果:", result)
    #
    # print("\n" + "="*20 + " 测试图片模式 " + "="*20)
    # img_url = "https://tse1.mm.bing.net/th/id/OIP.09GZTHe-lwPCKp7hWa4gRAHaE6?rs=1&pid=ImgDetMain&o=7&rm=3"
    # result_img = service.analyze(image_data=img_url)
    # print("图片结果:", result_img)