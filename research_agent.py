from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langgraph.graph.message import AnyMessage
from dotenv import load_dotenv
load_dotenv()
import os
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph.message import add_messages, MessagesState
from typing import Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.tools import DuckDuckGoSearchRun
import operator
from langchain_google_genai import ChatGoogleGenerativeAI


model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
)