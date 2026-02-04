from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

def get_basic_rag_prompt():
    return ChatPromptTemplate.from_messages(
        [
            SystemMessage("""
            You are science expert. You must find answer to user's query in context.
            Context is placed between xml tag <context>.
            Each document from context is placed between xml tag <document>.
            Your answers must be short.
            If you cannot find answer in context, answer "no information".
            """),
            HumanMessagePromptTemplate.from_template("""<query>{query}</query><context>{context}</context>"""),
        ]
    )