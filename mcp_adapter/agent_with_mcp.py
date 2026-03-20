import json
import asyncio
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Sequence
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model='gemini-3.1-flash-lite-preview',
    temperature=0.7
)

# MCP Servers
mcp_servers = MultiServerMCPClient({
    "context7": {
        "transport": "http",
        "url": "https://mcp.context7.com/mcp"
    },
    "calculator": {
        "transport": 'stdio',
        "command": 'uvx',
        "args": ['mcp-server-calculator']
    }
})


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# TOOLS


@tool(
    name_or_callable='get_random_number',
    description='Useful to get a random number between the given range [low, high]'
)
def get_random_number(low=0, high=10) -> int:
    '''
    Returns a random number between low and high (inclusive).
    Parameters:
    - low (int): The lower bound of the range.
    - high (int): The upper bound of the range.
    Returns:
    - int: A random number between low and high.
    '''
    import random
    return random.randint(low, high)


async def build_graph():
    global llm
    mcp_tools = await mcp_servers.get_tools()
    tools = [get_random_number, *mcp_tools]

    print(f"\n{'='*50}")
    print(f"  🛠️  AVAILABLE TOOLS ({len(tools)} total)")
    print(f"{'='*50}")

    llm = llm.bind_tools(tools)

    async def llm_node(state: AgentState) -> AgentState:
        res = await llm.ainvoke(state['messages'])
        return {'messages': [res]}

    def should_call_tools(state: AgentState):
        last_message = state['messages'][-1]
        if last_message.tool_calls:
            print(
                f'Tool calling: \n{json.dumps(
                    last_message.tool_calls, indent=4
                )}')
            return 'tool_call'
        else:
            return 'end'

    tools_node = ToolNode(tools=tools)

    graph = StateGraph(AgentState)

    graph.add_node('llm_node', llm_node)
    graph.add_node('tools_node', tools_node)

    graph.add_edge(START, 'llm_node')
    graph.add_conditional_edges(
        'llm_node',
        should_call_tools,
        {
            'tool_call': 'tools_node',
            'end': END
        }
    )
    graph.add_edge('tools_node', 'llm_node')

    app = graph.compile()

    return app


async def main():
    app = await build_graph()

    chat_history: list[BaseMessage] = []
    system_prompt = SystemMessage(
        """
        ### INSTRUCTIONS
        You are a helpful AI assistant. Answer the user queries with proper knowledge, use tool calls if required from the available tools. DO NOT HALUCINATE. Answer the questions based on real facts only!
        """
    )
    chat_history.append(system_prompt)

    while True:
        user_input = input('User: ')

        if (user_input.strip() in ('exit', 'bye')):
            print('AI: Bye!')
            break

        user_prompt = HumanMessage(user_input)
        chat_history.append(user_prompt)

        response = await app.ainvoke({'messages': chat_history})

        last_message = response['messages'][-1]

        # Code to print the message
        if isinstance(last_message.content, list):
            if last_message.content[0]['text']:
                print(f"AI: {last_message.content[0]['text']}")
            elif last_message.content[0]['message']:
                print(f"AI: {last_message.content[0]['message']}")
        else:
            print(f"AI: {last_message.content}")

        chat_history.append(last_message)

if __name__ == '__main__':
    asyncio.run(main())
