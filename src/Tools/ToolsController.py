from typing import List, Dict
from langchain.tools import BaseTool
from src.Tools.dummy_tool import UniversityTool

class ToolsController:
    def __init__(self):
        self.__tools_map = {
            UniversityTool.get_name(): UniversityTool,
        }

    @staticmethod
    def get_tool_type_to_cls_dict() -> Dict:
        tool_type_to_cls_dict = dict()
        tool_type_to_cls_dict["UniversityTool"] = UniversityTool
        return tool_type_to_cls_dict

    def get_tools(self, *, tools: List[str], **kwargs) -> List[BaseTool]:
        all_args = dict()
        all_args.update(kwargs)

        all_tools = []

        for tool_name in tools:
            if tool_name not in self.__tools_map:
                raise ValueError(f"invalid tool given : {tool_name}")
            reqd_args = self.__tools_map[tool_name].get_required_args(all_args)
            tool_obj = self.__tools_map[tool_name].create_tool(**reqd_args)
            all_tools.append(tool_obj)
        return all_tools

# if __name__=='__main__':
#     controlller = ToolsController()
#     print(controlller.get_tools())
