from models.StateSchema import State
from typing import Literal 
from langgraph.graph import StateGraph, START, END 
from nodes.Decision_node import decide_retrieval 
from nodes.Generate_node import generate_direct 
from nodes.Generate_from_context import generate_from_context 
from nodes.Relevance_node import is_relevant 
from nodes.Reterival_node import retrieve 
from nodes.Web_node import web_search_node,rewrite_query_node
from langsmith import traceable 



def no_relevant_docs(state: State):
    return {"answer": "No relevant document found.", "context": ""}


def route_after_decide(state: State) -> Literal["generate_direct", "retrieve"]:
    if state["need_retrieval"]:
        return "retrieve"
    return "generate_direct" 

def route_after_relevance(state: State) -> Literal["generate_from_context", "rewrite_query"]:
    if state.get("relevant_docs") and len(state["relevant_docs"]) > 0:
        return "generate_from_context"
    return "rewrite_query"




@traceable(name="graph_node")
def create_app() -> any:

    g = StateGraph(State)

    g.add_node("decide_retrieval", decide_retrieval)
    g.add_node("generate_direct", generate_direct)
    g.add_node("retrieve", retrieve)

    g.add_node("is_relevant", is_relevant)
    g.add_node("generate_from_context", generate_from_context)


    g.add_node("rewrite_query", rewrite_query_node)
    g.add_node("web_search", web_search_node)

    g.add_edge(START, "decide_retrieval")

    g.add_conditional_edges(
        "decide_retrieval",
        route_after_decide,
        {
            "generate_direct": "generate_direct",
            "retrieve": "retrieve",
         },
    )

    g.add_edge("generate_direct", END)

    # vector retrieval → relevance
    g.add_edge("retrieve", "is_relevant")

    # relevance router: if relevant → generate, else → rewrite_query
    g.add_conditional_edges(
        "is_relevant",
        route_after_relevance,
        {
            "generate_from_context": "generate_from_context",
            "rewrite_query": "rewrite_query",
        },
    )

    # web fallback path
    g.add_edge("rewrite_query", "web_search")
    g.add_edge("web_search", "is_relevant")  # 🔁 circle back

    # final
    g.add_edge("generate_from_context", END)

    return g.compile()
