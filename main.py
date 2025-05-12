
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

class State(TypedDict):
    question: str
    answer: str
    context:Annotated[list, operator.add]
    
graph = StateGraph(State)

# Nodes
def duckduckgo_search(state):
    """Node for websearch"""
    search = DuckDuckGoSearchRun()
    question = state['question']
    search_result = search.invoke(question)
    return {'context': [search_result]}

def wiki_search(state):
    """Node for wikipedia search"""
    question = state['question']
    documents = WikipediaLoader(question, load_max_docs=5).load()
    search_result = []
    for doc in documents:
        search_result.append(doc.metadata['summary'])
    
    return {'context':  search_result}

def model_call(state):
    context = '\n'.join(state['context'])
    prompt = (f"context: {context}\n Answer the questions below using the context above, if the answer does not present in the context say it, Here is the questions {state['question']}")
    return {'answer': model.invoke(prompt)}

graph.add_node('wiki', wiki_search)
graph.add_node('duck',duckduckgo_search)
graph.add_node('model', model_call)

graph.add_edge(START, 'wiki')
graph.add_edge(START, 'duck')
graph.add_edge('wiki', 'model')
graph.add_edge('duck','model')
graph.add_edge('model',END)
graph = graph.compile()

if __name__  == '__main__':
    user_query = input('Enter your prompt here: ')
    print('model genarating...')
    graph_result = graph.invoke({'question': user_query})
    ai_result = graph_result['answer'].content
    print(ai_result)