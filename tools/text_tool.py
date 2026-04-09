import json

from langchain_core.tools import tool

from rag.rag_service import NutritionRAGService

service = NutritionRAGService()


from pydantic import BaseModel, Field

class TextEstimationInput(BaseModel):
    user_input: str = Field(
        ...,
        description="用户提供的食物名称或饮食记录纯文字描述，例如：'我中午吃了一碗米饭和宫保鸡丁'"
    )


@tool("text_calorie_estimation", args_schema=TextEstimationInput)
def text_estimation_tool(user_input: str) -> str:
    """
    当用户仅提供文字描述的菜品，需要识别菜品名称并估算其卡路里时，请调用此工具。

    注意：
    1. 只有在用户输入纯文本时使用。
    2. 如果用户提供了图片链接或要求分析图片，请【绝对不要】调用此工具，应改用视觉工具。
    """
    try:
        # 这里直接调用你原本封装好的 analyze 逻辑
        # 注意：这里去掉了外层的 if-else 路由，直接进入文本处理链
        print(f">> [Tool 被唤醒] 正在执行文本估算: {user_input}")

        estimated_data = service.chain_estimation.invoke({
            "input": user_input,
            "format_instructions": service.estimation_parser.get_format_instructions()
        })

        if not estimated_data or 'items' not in estimated_data:
            return "未能从您的描述中识别出具体的菜品，请尝试换一种说法（例如包含具体的菜名）。"


        items_list = estimated_data.get('items', [])
        rag_context = service.retrieve_context(items_list)

        final_result = service.chain_kcal.invoke({
            "user_data": estimated_data,
            "context": rag_context if rag_context else "未找到参考数据，请基于常识估算。",
            "format_instructions": service.calculation_parser.get_format_instructions()
        })


        return json.dumps(final_result, ensure_ascii=False)

    except Exception as e:
        # 工具内部的异常必须被捕获，并作为文本返回给大模型，让大模型知道调用失败了
        return f"抱歉，在处理您的饮食建议时遇到技术故障：{str(e)}。您可以尝试直接输入菜品名称。"