from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate


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
            HumanMessagePromptTemplate.from_template("""<query>{query}</query><context>{context}</context>If you cannot find answer in context, answer exactly <output>'no information'</output> without any additional text"""),
        ]
    )


def get_query_cat_prompt():
    return ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template("""
            You are science expert. You must categorize user query with science topics.
            Query is placed between xml tag <query>.
            You are allowed to choose zero, one or several categories. It is normal to return several categories.
            Categories titles are placed between xml tag <category>.
            Return output (placed between <output> and </output> tags) which follows matching categories. Dont return category name.

            Categories:
            - <category>Astrophysics</category><output>astro-ph</output>
            - <category>Condensed Matter</category>cond-mat</output>
            - <category>High Energy Physics - Experiment</category><output>gr-qc</output>
            - <category>High Energy Physics - Lattice</category><output>hep-ex</output>
            - <category>High Energy Physics - Phenomenology</category><output>hep-lat</output>
            - <category>High Energy Physics - Theory</category><output>hep-ph</output>
            - <category>Mathematical Physics</category><output>math-ph</output>
            - <category>Nonlinear Sciences</category><output>nlin</output>
            - <category>Nuclear Experiment</category><output>nucl-ex</output>
            - <category>Nuclear Theory</category><output>nucl-th</output>
            - <category>Physics</category><output>physics</output>
            - <category>Quantum Physics</category><output>quant-ph</output>
            - <category>Mathematics</category><output>math</output>
            - <category>Computing Research Repository</category><output>cs</output>
            - <category>Quantitative Biology</category><output>q-bio</output>
            - <category>Quantitative Finance</category><output>q-fin</output>
            - <category>Statistics</category><output>stat</output>
            - <category>Electrical Engineering and Systems Science</category><output>eess</output>
            - <category>Economics</category></output>econ</output>

            {format_instruction}
            """),
            HumanMessagePromptTemplate.from_template("""Which categories fit this query mostly? <query>{query}</query>""")
        ]
    )