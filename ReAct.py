from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, ToolCall
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import tools_condition, ToolNode

from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
    max_tokens=None,
)


# Define tool
def multiply(a: int, b:int) -> int:
    """Multiply two numbers
    
    Args:
        a: first int
        b: second int
    """
    
    return a * b

def add(a: int, b:int) -> int:
    """Multiply two numbers
    
    Args:
        a: first int
        b: second int
    """
    
    return a + b

def divide(a: int, b:int) -> float:
    """Multiply two numbers
    
    Args:
        a: first int
        b: second int
    """
    
    return a / b

# bind tools to model
tools = [add, multiply, divide]
llm_with_tools = llm.bind_tools(tools)

llm_with_tools.invoke('add 5 + 20 and multiply 2 * 3').tool_calls

# Define state
class State(MessagesState):
    pass

# Node1
def assistent(state: State) -> State:
    message = state['messages']
    llm_response = llm_with_tools.invoke(message)
    return {'messages': [llm_response]}

llm_tools_json = llm_with_tools.kwargs['tools']
tools_by_name = {tool_dict['function']['name']: tool_func for tool_dict, tool_func  in zip(llm_tools_json, tools)}

# Tool Execution node
def tool_execution(state: State):
    for tool_call in state['messages'][-1].tool_calls:
        tool = tools_by_name[tool_call['name']]
        tool_args_dict = tool_call['args']
        tool_result = tool(**tool_args_dict)
    return {'messages' :[ToolMessage(content=str(tool_result), name=tool_call['name'], tool_call_id=tool_call['id'])]}

def tool_routing_condition(state: State):
    if state['messages'][-1].content:
        return 'end'
    else:
        return "continue"
    
    
graph = StateGraph(State)

graph.add_node('llm', assistent)
graph.add_node('tools', tool_execution)
graph.add_edge(START, 'llm')
graph.add_edge('tools', 'llm')
graph.add_conditional_edges('llm',
                            path=tool_routing_condition,
                            path_map={'continue': 'tools', 'end': END} )


graph = graph.compile()

output = graph.invoke({'messages':[HumanMessage(content='what is 2 * 5')]})

for m in output['messages']:
    print(m.pretty_print())
