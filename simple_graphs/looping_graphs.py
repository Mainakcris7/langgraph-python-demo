from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from random import randint

class AgentState(TypedDict):
    player_name: str
    guesses: List[int]
    lower_bound: int
    upper_bound: int
    target: int
    hint: str
    result: str

def setup_node(state: AgentState):
    """Sets up the state"""
    state['guesses'] = []
    state['hint'] = ''
    state['target'] = randint(state['lower_bound'], state['upper_bound'])
    return state

def guess_node(state: AgentState):
    """Guesses a number"""
    if state['hint'] == '':
        state['guesses'].append(randint(state['lower_bound'], state['upper_bound']))
    elif state['hint'] == 'higher':
        state['guesses'].append(randint(state['guesses'][-1] + 1, state['upper_bound']))
    else:
        state['guesses'].append(randint(state['lower_bound'], state['guesses'][-1] - 1))
    
    return state

def hint_node(state: AgentState):
    if state['guesses'][-1] > state['target']:
        state['hint'] = 'lower'
    elif state['guesses'][-1] < state['target']:
        state['hint'] = 'higher'
    return state

def decide_next_node(state: AgentState):
    if state['guesses'][-1] == state['target']:
        return "exit"
    if len(state['guesses']) < 7:
        return "guess_again"
    else:
        return "exit"

def evaluate_node(state: AgentState):
    if(state['guesses'][-1] == state['target']):
        state['result'] = f"Congrats {state['player_name']}, you have guessed it in {len(state['guesses'])} attempts"
    else:
        state['result'] = f"Sorry {state['player_name']}, you have failed to guess the number"
    
    return state
        
graph = StateGraph(AgentState)

graph.add_node('setup', setup_node)
graph.add_node('guess', guess_node)
graph.add_node('hint', hint_node)
graph.add_node('evaluate', evaluate_node)

graph.add_edge(START, 'setup')
graph.add_edge('setup', 'guess')
graph.add_edge('guess', 'hint')
graph.add_conditional_edges(
    'hint',
    decide_next_node,
    {
        'guess_again': 'guess',
        'exit': 'evaluate'
    }
)
graph.add_edge('evaluate', END)

init_state = AgentState(
    player_name = 'Mainak',
    lower_bound=1,
    upper_bound=20
)

app = graph.compile()

result = app.invoke(init_state)
print(result)