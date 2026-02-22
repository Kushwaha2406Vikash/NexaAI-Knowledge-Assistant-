from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate 
from models.StateSchema import State 
from langchain_core.documents import Document 
from langchain_community.tools.tavily_search import TavilySearchResults
from models.llm import get_llm 
from langsmith import traceable 

llm =get_llm()

class WebQuery(BaseModel):
    query: str

rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user question into a web search query composed of keywords.\n"
            "Rules:\n"
            "- Keep it short (6–14 words).\n"
            "- If the question implies recency, add (last 30 days).\n"
            "- Do NOT answer the question.\n"
            "- Return JSON with a single key: query",
        ),
        ("human", "Question: {question}"),
    ]
)

rewrite_chain = rewrite_prompt | llm.with_structured_output(WebQuery)

@traceable(name="Rewrite Query Node")
def rewrite_query_node(state: State):
    out = rewrite_chain.invoke({"question": state["question"]})
    return {"web_query": out.query}

#tavily = TavilySearchResults(max_results=5)

@traceable(name="Web Node")
def web_search_node(state: State):
    q = state.get("web_query") or state["question"]
    tavily = TavilySearchResults(max_results=5)
    results = tavily.invoke({"query": q})

    docs = []
    for r in results or []:
        title = r.get("title", "")
        url = r.get("url", "")
        content = r.get("content", "") or r.get("snippet", "")
        text = f"TITLE: {title}\nURL: {url}\nCONTENT:\n{content}"
        docs.append(
            Document(
                page_content=text,
                metadata={"source": "web", "url": url, "title": title},
            )
        )

    return {"docs": docs}