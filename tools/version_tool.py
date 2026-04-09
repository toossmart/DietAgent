from pydantic import BaseModel, Field
from langchain_core.tools import tool
import json

from rag.rag_service import NutritionRAGService

service = NutritionRAGService()

class ImageEstimationInput(BaseModel):
    image_url: str = Field(
        ...,
        description="用户上传图片的 URL 或 Base64 数据字符串。当用户发送图片时，必须提取此参数。"
    )


@tool("image_calorie_estimation", args_schema=ImageEstimationInput)
def version_estimation_tool(image_url: str)->str:
    """
        当用户上传了食物图片，或者提供图片链接要求分析热量和营养时，调用此工具。
        工具会识别图片内容，调用data中的数据进行高精度估算。
    """
    try:
        # 直接调用 service 中封装好的原子方法
        estimated_data = service.chain_version.invoke({
            "image": image_url
        })
        if not estimated_data or 'items' not in estimated_data:
            return "未能从您的图片中中识别出具体的菜品，请尝试换重新上传。"
        items_list = estimated_data.get('items', [])
        rag_context = service.retrieve_context(items_list)

        final_result = service.chain_version.invoke({
            "user_data": estimated_data,
            "context": rag_context if rag_context else "未找到参考数据，请基于常识估算。",
            "format_instructions": service.calculation_parser.get_format_instructions()
        })

        return json.dumps(final_result, ensure_ascii=False)
    except Exception as e:

        return f"视觉识别过程中出现错误：{str(e)}。请确保图片清晰且格式正确。"
