from typing import TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END


class State(TypedDict):
    input: str
    output: str


llm = ChatOpenAI(model="gpt-4o-mini")  # any supported model works


def call_model(state: State) -> State:
    msg = llm.invoke(state["input"])
    return {"output": msg.content}


builder = StateGraph(State)
builder.add_node("call_model", call_model)
builder.set_entry_point("call_model")
builder.add_edge("call_model", END)

graph = builder.compile()
