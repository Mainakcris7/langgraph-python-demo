from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
# from IPython.display import display, Image

class AgentState(TypedDict):
    num1: int
    num2: int
    operation1: str
    num3: int
    num4: int
    operation2: str
    result1: int
    result2: int

def add_node1(state: AgentState):
    """
    Adds two numbers 1
    """
    print("Add node 1: ", state)
    state['result1'] = state['num1'] + state['num2']    
    return state

def subtract_node1(state: AgentState):
    """
    Subtracts two numbers 2
    """
    print("Subtract node 1: ", state)
    state['result1'] = state['num1'] - state['num2']
    
def add_node2(state: AgentState):
    """
    Adds two numbers 2
    """
    print("Add node 2: ", state)
    state['result2'] = state['num3'] + state['num4']    
    return state

def subtract_node2(state: AgentState):
    """
    Subtracts two numbers 2
    """
    print("Subtract node 2: ", state)
    state['result2'] = state['num3'] - state['num4']
    return state
    
def decide_edge1(state: AgentState):
    if(state['operation1'] == '+'):
        return "add_op_1"
    else:
        return "subtract_op_1"
    
def decide_edge2(state: AgentState):
    if(state['operation2'] == '+'):
        return "add_op_2"
    else:
        return "subtract_op_2"
    
graph = StateGraph(AgentState)

graph.add_node("router1", lambda x: x)
graph.add_node("add_node1", add_node1)
graph.add_node("subtract_node1", subtract_node1)
graph.add_node("router2", lambda x: x)
graph.add_node("add_node2", add_node2)
graph.add_node("subtract_node2", subtract_node2)

graph.add_edge(START, "router1")
graph.add_conditional_edges(
    "router1",
    decide_edge1,
    {
        "add_op_1": "add_node1",
        "subtract_op_1": "subtract_node1",
    }
)
graph.add_edge("add_node1", "router2")
graph.add_edge("subtract_node1", "router2")
graph.add_conditional_edges(
    "router2",
    decide_edge2,
    {
        "add_op_2": "add_node2",
        "subtract_op_2": "subtract_node2",
    }
)
graph.add_edge("add_node2", END)
graph.add_edge("subtract_node2", END)

app = graph.compile()

initial_state = AgentState(
    num1 = 10,
    num2 = 20,
    operation1 = '+',
    num3 = 80,
    num4 = 70,
    operation2 = '-'
)

result = app.invoke(initial_state)
print(result)