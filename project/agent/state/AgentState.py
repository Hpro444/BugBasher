from langchain_core.messages import BaseMessage
from typing import Sequence, TypedDict, Annotated
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    Represents the state of an agent within a LangGraph workflow.

    Attributes:
        messages (Annotated[Sequence[BaseMessage], add_messages]):
            A list of messages exchanged during the conversation.
            This sequence is automatically managed by LangGraph to
            accumulate messages (system, human, AI, or tool) as the
            agent interacts with its environment.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
