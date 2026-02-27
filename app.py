import base64
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

# 从 rag_service 导入已有的 Pydantic 模型和 service 实例
from rag.rag_service import NutritionRAGService, logger, DishItem

app = FastAPI(title="营养分析 RAG 服务", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

service = NutritionRAGService()


# ---------- 请求/响应模型 ----------

class AnalyzeRequest(BaseModel):
    text: Optional[str] = None
    image_url: Optional[str] = None


# 明确响应结构，确保 FastAPI 正确解析嵌套的 items 列表
class AnalyzeResponse(BaseModel):
    items: List[DishItem]
    total_calories: int
    advice: str


# ---------- 工具函数 ----------

def image_file_to_data_url(file: UploadFile) -> str:
    contents = file.file.read()
    base64_str = base64.b64encode(contents).decode("utf-8")
    mime_type = file.content_type or "image/jpeg"
    return f"data:{mime_type};base64,{base64_str}"


# ---------- 路由端点 ----------

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_json(request: AnalyzeRequest):
    logger.info(f"接收到请求: text={request.text}")
    try:
        # 调用核心分析逻辑
        result = service.analyze(user_input=request.text, image_data=request.image_url)

        # 检查业务逻辑错误（如识别失败）
        if isinstance(result, dict) and "error" in result:
            logger.warning(f"分析失败: {result['error']}")
            raise HTTPException(status_code=400, detail=result["error"])

        return result
    except Exception as e:
        logger.error(f"分析异常: {str(e)}")
        raise HTTPException(status_code=500, detail="服务内部错误，请稍后重试")


@app.post("/analyze_with_image", response_model=AnalyzeResponse)
async def analyze_with_image(
        text: Optional[str] = Form(None),
        image: UploadFile = File(None)
):
    try:
        image_data = image_file_to_data_url(image) if image else None
        result = service.analyze(user_input=text, image_data=image_data)

        if isinstance(result, dict) and "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except Exception as e:
        logger.error(f"图片上传分析异常: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)