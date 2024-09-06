import json
import asyncio
from typing import Type,Optional
from pydantic import BaseModel,Field,PrivateAttr

from langchain.tools import BaseTool
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

import os
from langchain.tools import BaseTool
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatYandexGPT
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.yandex import YandexGPTEmbeddings

os.environ["YC_API_KEY"] = "Your Yandex GPT API Key"
os.environ["YC_FOLDER_ID"] = "Your Yandex FOLDER ID"

embeddings = YandexGPTEmbeddings()

model = ChatYandexGPT()

class SearchInput(BaseModel):
    query: str = Field(description="This is the exact user query given.")
    
class UniversityTool(BaseTool):
    name: str = "University_tool"
    description: str = "Tool to provide information about the document to master program applicants/ or write your custom tool "
    args_schema: type = SearchInput

    def __init__(self, **kwargs):
        super().__init__()
    
    class Config:
        """Configuration for this pydantic object."""

        extra = 'allow'
        arbitrary_types_allowed = True
    
    @classmethod
    def create_tool(cls, **kwargs) -> BaseTool:
        return cls(**kwargs)

    @staticmethod
    def get_name():
        return "University_tool"

    @staticmethod
    def get_description():
        return "Tool to provide information about the document to master program applicants "

    @staticmethod
    def get_args_schema():
        return SearchInput
    
    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        loader = TextLoader(r"Your custom question answer text file for context to Large language Model(RAG) in text format ", encoding='utf-8')
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
        split_docs = text_splitter.split_documents(documents)
        embeddings = YandexGPTEmbeddings()        
        db = FAISS.from_documents(split_docs, embeddings)
        retriever = db.as_retriever()
        res = retriever.invoke(query)
        context = " ".join([doc.page_content for doc in res])
        template = """Answer the question based on context and your intelligence ,Please do not provide information about document or context in you response:
        {context} 
        Question: {question} for general salutation question give general salutation"""
        prompt = ChatPromptTemplate.from_template(template)
        model = ChatYandexGPT()
        chain = (
            {"context": retriever, "question": RunnablePassthrough()} 
            | prompt 
            | model 
            | StrOutputParser()
            
        )
        return chain.invoke(query)

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        raise NotImplementedError
    
    @staticmethod
    def get_required_args(kwargs: dict) -> dict:
        return {}