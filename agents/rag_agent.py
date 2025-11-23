import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage
from langchain_core.messages import SystemMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated, Sequence
from langgraph.prebuilt import ToolNode

load_dotenv()

pdf_file_path = os.path.join(
    "data", "TechNova_Internal_Projects_Documentation_Detailed.pdf")

if os.path.exists(pdf_file_path):
    print("PDF file exists.")
else:
    print("PDF file does not exist.")

llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
embedding = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001", google_api_key=os.environ['GOOGLE_API_KEY'])

pdf_loader = PyPDFLoader(pdf_file_path)
docs = pdf_loader.load()

print(f"Total {len(docs)} pages loaded.")

vector_db_path = os.path.join("db", "faiss_db")

if os.path.exists(vector_db_path):
    print("Vector store already exists!")
else:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)

    for i, chunk in enumerate(chunks, 1):
        chunk.metadata['doc_index'] = i

    print("Document chunk information:")
    print(f"Total chunks: {len(chunks)}")
    print(f"First chunk length: {len(chunks[0].page_content)}")
    print(f"First chunk:\n{chunks[0].page_content}")

    print("\nStoring the vector store in local...")
    store = FAISS.from_documents(
        documents=chunks,
        embedding=embedding
    )
    store.save_local(vector_db_path)

    print("Vector store created successfully!")


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


vector_store = FAISS.load_local(
    folder_path=vector_db_path,
    embeddings=embedding,
    allow_dangerous_deserialization=True
)

retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.1}
)


@tool(name_or_callable="Internal_Doc_Info_Retriever", description="Useful when retrieving informations about the internal projects, codebases from the company documentation")
def internal_info_retriever(query: str) -> str:
    docs = retriever.invoke(query)

    if not docs:
        return "Sorry, no informations about your query can be retrieved!"

    knowledge_base = [
        f"Page: {doc.metadata['page_label']} Document: {doc.metadata['doc_index']}:\n{doc.page_content}" for i, doc in enumerate(docs)]

    return "\n\n".join(knowledge_base)


@tool(name_or_callable="Current_Date", description="Useful when knowing about the current date")
def get_current_date() -> str:
    import datetime
    date = datetime.datetime.now()

    return date.strftime("%d/%m/%y")


llm = llm.bind_tools([internal_info_retriever, get_current_date])


def llm_node(state: AgentState) -> AgentState:
    all_messages = state['messages']
    result = llm.invoke(all_messages)

    return {'messages': [result]}


def should_call_tools(state: AgentState):
    if state['messages'][-1].tool_calls:
        return "tool_call"
    return "end"


graph = StateGraph(AgentState)
graph.add_node("llm_node", llm_node)
graph.add_node("tool_node", ToolNode(
    tools=[internal_info_retriever, get_current_date]))


graph.add_conditional_edges(
    "llm_node",
    should_call_tools,
    {
        "tool_call": "tool_node",
        "end": END
    }
)

graph.add_edge(START, "llm_node")
graph.add_edge("tool_node", "llm_node")

app = graph.compile()

chat_history = []
system_prompt = """
You are a helpful AI assistant. You are given a set of tools (internal project info retreiver, get current date), your task is to efficiently address and answer the user queries. DO NOT HALUCINATE. If you don't know the answer, simply say Sorry I couldn't find the answer of your query, do not make things by your own!

If you are using the 'Internal Project Info retriever' tool, add the metadata (document number) from where you got the context in response content in the given format - 
<Your Response>
<Resource: Page x Document a, Page y Document b, ...>
"""
chat_history.append(SystemMessage(content=system_prompt))

while True:

    user_input = input("User: ")
    if (user_input.strip() == 'exit'):
        break

    chat_history.append(HumanMessage(content=user_input))

    result = app.invoke(
        {'messages': chat_history}
    )

    last_message = result['messages'][-1]

    if isinstance(last_message.content, list):
        if last_message.content[0]['text']:
            print(f"AI: {last_message.content[0]['text']}")
        elif last_message.content[0]['message']:
            print(f"AI: {last_message.content[0]['message']}")
    else:
        print(f"AI: {last_message.content}")

    chat_history.append(last_message)

# print(chat_history)
