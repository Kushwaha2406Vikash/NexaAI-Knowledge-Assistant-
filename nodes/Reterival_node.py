from models.StateSchema import State 
from ingestions.document_retriever import get_company_retriever 
from langsmith import traceable



@traceable(name="Retrieve Node")
def retrieve(state: State):
    retriever= get_company_retriever()
    return {"docs": retriever.invoke(state["question"])}