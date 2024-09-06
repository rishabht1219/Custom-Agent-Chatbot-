from langchain.schema.messages  import (
    SystemMessage
)
from langchain.tools import BaseTool, Tool
from prompts.agents_prompt import agents_prompt
from langchain.agents.agent import AgentExecutor
from src.Tools.ToolsController import ToolsController
from langchain.agents import AgentType, initialize_agent
from langchain_community.chat_models import ChatYandexGPT

class CustomAgent:
    def __init__(self):
        self.__turbo_llm = ChatYandexGPT()
        self.__tool_names = ["University_tool"]
        self.__tools = None
        if self.__tool_names and all(
                [isinstance(item, BaseTool) or isinstance(item, Tool) for item in self.__tool_names]):
            self.__tools = self.__tool_names

        if self.__tool_names and all([isinstance(item, str) for item in self.__tool_names]):
            self.__tools = ToolsController().get_tools(
                tools=self.__tool_names)
        self.__prompt = agents_prompt
        self.agent = self.create_agent()

    def create_agent(self) -> AgentExecutor:
        agent_kwargs = {
            "system_message": SystemMessage(content=self.__prompt)
        }
        conversational_agent = initialize_agent(
            tools=self.__tools,
            agent=AgentType.OPENAI_FUNCTIONS,
            llm=self.__turbo_llm,
            max_iterations=3,
            agent_kwargs=agent_kwargs,
            system_message=SystemMessage(content=self.__prompt),
            handle_parsing_errors=True
        )
        return conversational_agent
    
    def chat(self,query:str)->str:
        return self.agent.invoke(query)

# if __name__=="__main__":
#     agent = CustomAgent()
#     print(agent.chat(""))