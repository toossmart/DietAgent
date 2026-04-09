# 这里的“初始化”是指将模型、工具、Prompt 三者绑定的过程
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate

from model.factory import chat_model_factory_kcal
from rag.rag_service import NutritionRAGService

from tools.text_tool import text_estimation_tool  # 你包装好的工具
from tools.version_tool import version_estimation_tool
from utils.load_prompts import load_agent_prompts
class MasterAgentManager:
    def __init__(self):
        # 1. 核心大脑
        self.llm = chat_model_factory_kcal
        # 2. 注入工具箱 (这里包含了你的 RAG 服务)
        self.tools = [text_estimation_tool,version_estimation_tool]

        self.prompt = PromptTemplate.from_template(load_agent_prompts())
        # 组装
        self.agent_runnable=create_react_agent(llm=self.llm,tools=self.tools,prompt=self.prompt)

        self.agent_executor = AgentExecutor(
            agent=self.agent_runnable,
            tools=self.tools,
            verbose=True,  # 打印详细执行过程
            handle_parsing_errors=True,
            max_iterations=5  # 防止无限循环
        )

    def run(self, user_input: str) -> str:
        """对用户输入执行 Agent 调用"""
        response = self.agent_executor.invoke({"input": user_input})
        return response["output"]


# 单例化
master_agent = MasterAgentManager()