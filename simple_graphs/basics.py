from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    name: str
    age: str
    skills: List[str]
    summary: str
    
# First node
def first_node(state: AgentState):
    """
    Formats the name of the user
    """
    print("Processing name...")
    state['summary'] = f"{state['name']}, welcome to the system!"
    return state

# Second node
def second_node(state: AgentState):
    """
    Formats the age of the user
    """
    print("Processing age...")
    state['summary'] = state['summary'] + f"\nYou are {state['age']} years old!"
    return state

# Third node
def third_node(state: AgentState):
    """
    Formats the user skills
    """
    print("Processing skills...")
    state['summary'] = state['summary'] + f"\nYou have skills in: {", ".join(state['skills'])}"
    return state

graph = StateGraph(AgentState)

graph.add_node("first_node", first_node)
graph.add_node("second_node", second_node)
graph.add_node("third_node", third_node)

graph.add_edge(START, "first_node")
graph.add_edge("first_node", "second_node")
graph.add_edge("second_node", "third_node")
graph.add_edge("third_node", END)

# graph.set_entry_point("first_node")
# graph.set_finish_point("third_node")

app = graph.compile()

initial_state = AgentState(name="Mainak", age=22, skills=["Java", "Python", "GenAI"])

result = app.invoke(initial_state)
print(result['summary'])