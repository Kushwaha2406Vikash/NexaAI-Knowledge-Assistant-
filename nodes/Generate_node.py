from pydantic import BaseModel,Field
from langchain_core.prompts import ChatPromptTemplate 
from models.StateSchema import State 
from models.llm import get_llm 
from langsmith import traceable

llm = get_llm() 

direct_generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the question using only your general knowledge.\n"
            "Do NOT assume access to external documents.\n"
            "If you are unsure or the answer requires specific sources, say:\n"
            "'I don't know based on my general knowledge.'"
        ),
        ("human", "{question}"),
    ]
)

@traceable(name="Generator Node")
def generate_direct(state: State):
    out = llm.invoke(
        direct_generation_prompt.format_messages(
            question=state["question"]
        )
    )
    return {
        "answer": out.content
    }