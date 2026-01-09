# import all the important modules
import os
from dotenv import load_dotenv
import requests
from typing import TypedDict, List, Annotated
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage,HumanMessage,SystemMessage,AIMessage,ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START ,END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_community.tools import DuckDuckGoSearchRun
import sqlite3
from langsmith import traceable

# Load environment variables from .env file

load_dotenv()

#Change langsmith project name
os.environ["LANGCHAIN_PROJECT"] = "Financial_Agent"

# Initialize LLMs

llm1 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

llm2 = ChatOpenAI(model_name="gpt-4", temperature=0.5)


# Define our tools

#searching tool
search_tool = DuckDuckGoSearchRun()

#stock sentiment new tool
@tool
def get_stock_sentiment_news(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey="
    r = requests.get(url)
    return r.json()


# creating tool list
tool_list = [search_tool, get_stock_sentiment_news]
# Make LLM aware of tools
llm_with_tools = llm1.bind_tools(tool_list)

# Define state for the graph
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# graph nodes

# Also we want to observe node functions in langsmith
@traceable
def data_loader(state: AgentState) -> dict:
    """LLM node that may answer or request a tool call."""
    messages = state['messages']
    summary_prompt = [
    SystemMessage(content="""
    Your work is to get data. LLM with access to two tools: 
    1) DuckDuckGo – use this for general queries. 
    2) Finance Sentiment News Tool – use this to get stock/company sentiment news. 

    Rules:
    - If the user asks about general query, always call DuckDuckGo.
    - If the user asks for market news or stock sentiment, call the Get Stock Sentiment News Tool.
    """)]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

@traceable
def summarizer(state: AgentState) -> dict:
    """Final LLM node that summarizes the conversation."""
    messages = state['messages']
    summary_prompt = [
    SystemMessage(content="""
    You are a financial market and general query expert. 
    Summarize the conversation briefly. 
    Additionally, analyze any company stock sentiment news mentioned. 
    Provide the main information in 5 concise lines, demonstrating excellent knowledge of the market.
    """),
    *messages
]
    response = llm2.invoke(summary_prompt)
    return {"messages": [response]}

tool_node = ToolNode(tool_list)  # Executes tool calls

# graph structure
graph = StateGraph(AgentState)

# define nodes
graph.add_node("summarizer", summarizer)
graph.add_node("data_loader", data_loader)
graph.add_node("tools", tool_node)

#define edges
graph.add_edge(START, "data_loader")

# If the LLM asked for a tool, go to ToolNode; else finish
graph.add_conditional_edges("data_loader", tools_condition)

graph.add_edge("tools", "summarizer")   

#adding sqlite connection
conn = sqlite3.connect(database="agent_chat_db", check_same_thread=False)
#adding Sqlite checkpointing 
checkpointer = SqliteSaver(conn=conn)

#compiling the graph
chatbot = graph.compile()

#Now we can have a chat loop
print("Financial AI Agent is ready to chat! Type 'bye', 'exit', or 'stop' to end the conversation.")
while True:
    # Get user input
    user_input = input("User: ")

    # Check if the user wants to exit
    if user_input.lower() in ["bye", "exit", "stop"]:
        print("AI: Goodbye!")
        break

    # Invoke the chatbot
    out = chatbot.invoke(
        {"messages": [HumanMessage(content=user_input)]}, 
        checkpointer=checkpointer
    )

    # AI response
    ai_message = out["messages"][-1].content

    # Print AI response

    print(f"AI: {ai_message}")
