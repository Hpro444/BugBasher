from typing import List, Optional
import json

from project.sandbox import run_in_sandbox_tool
from langchain_core.tools.structured import StructuredTool
from project.agent.state import AgentState

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama.chat_models import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from project.config import Configuration


class BugBashAgent:
    """
    BugBashAgent is an intelligent debugging assistant designed to automatically fix
    buggy Python code using a language model and a testing tool.

    The agent:
      - Receives Python code as input.
      - Uses an LLM (ChatOllama) with tool access to fix and verify code.
      - Executes fixes in an isolated sandbox (via `run_in_sandbox_tool`) to confirm correctness.
      - Returns only working, validated Python code with no explanations or comments.

    """

    def __init__(self, model: Optional[str] = None, ollama_url: Optional[str] = None, tools: Optional[List[StructuredTool]] = None, max_tool_calls: int = 5):
        """
         Initializes the BugBashAgent with a language model, tools, and configuration.

         Parameters
         ----------
         model : Optional[str]
             The model name for the LLM. Defaults to Configuration().ollama_model.
         ollama_url : Optional[str]
             The base URL for the Ollama API. Defaults to Configuration().ollama_url.
         tools : Optional[List[StructuredTool]]
             A list of tools the agent can use. Defaults to [run_in_sandbox_tool].
         max_tool_calls : int
            Maximum number of tool calls allowed per debugging session
            before the agent stops automatically.
         """
        model = model or Configuration().ollama_model
        ollama_url = ollama_url or Configuration().ollama_url

        self._tools = tools or [run_in_sandbox_tool]
        self._chat_model_with_tools = ChatOllama(model=model, base_url=ollama_url).bind_tools(self._tools)
        self._state: AgentState = AgentState(messages=[])
        self._system_prompt = SystemMessage(content="""
        You are CodeFixer, an AI Python debugging assistant. Your only job is to fix buggy Python code.
        You have access to a tool: run_in_sandbox(code: str, timeout: int = 5) -> str. This tool executes Python code in an isolated temporary directory by wrapping it inside a pytest test function. It returns 'Success' if the code runs without errors, or the pytest-formatted error output if it fails. Always use this tool to verify your fixes before returning code.
        Rules:
        1. Only return working, fixed Python code.
        2. Do not include explanations, comments, or extra text.
        3. Preserve the original intent and functionality of the code.
        4. Focus solely on syntax, runtime, and logical errors.
        5. Assume all code is Python 3.
        Your output must be executable Python code only.
        """)

        self._max_tool_calls = max_tool_calls
        self._tool_calls_made = 0

        # Create graph
        graph = StateGraph(AgentState)

        graph.add_node('model_call', self._model_call)
        graph.add_node('tools', ToolNode(tools=self._tools))  # Automatically handles state

        graph.add_node('tool_code_fixer_node', self._tool_code_fixer_node)
        graph.add_edge('model_call', 'tool_code_fixer_node')
        graph.add_conditional_edges('tool_code_fixer_node', self._should_continue, {'end': END, 'continue': 'tools'})
        graph.add_edge('tools', 'model_call')

        graph.set_entry_point('model_call')

        self._agent = graph.compile()

    @staticmethod
    def extract_tool_call_from_content(content: str) -> Optional[dict]:
        """
        Checks if the content contains a </think> tag followed by a JSON tool call.
        If yes, returns the parsed tool call dict; else None.
        """
        if "</think>" not in content:
            return None

        # Everything after </think>
        after_think = content.split("</think>", 1)[1].strip()

        # Try to parse JSON if it contains "name" field
        if '"name"' in after_think:
            try:
                tool_call = json.loads(after_think)
                # Make sure it has required keys
                if isinstance(tool_call, dict) and "name" in tool_call and "arguments" in tool_call:
                    return tool_call
            except json.JSONDecodeError:
                return None
        return None

    @staticmethod
    def remove_after_think(text: str) -> str:
        """
        Removes everything after the first occurrence of </think> in the given text.

        Parameters
        ----------
        text : str
            The input string possibly containing a </think> tag.

        Returns
        -------
        str
            The text up to and including </think>. If the tag is not found, returns the original text.
        """
        if "</think>" in text:
            return text.split("</think>", 1)[0] + "</think>"
        return text

    def _tool_code_fixer_node(self, state: AgentState) -> AgentState:
        if not state['messages']:
            return state

        last_msg = state['messages'][-1]
        tool_call = self.extract_tool_call_from_content(last_msg.content)

        if tool_call:
            # Modify in-place
            last_msg.tool_call = tool_call
            last_msg.content = self.remove_after_think(last_msg.content)
            state['messages'][-1] = last_msg  # optional, already in-place

        return state

    def _model_call(self, state: AgentState) -> AgentState:
        """
               Executes a single model step within the LangGraph.

               Parameters
               ----------
               state : AgentState
                   The current agent state containing message history.

               Returns
               -------
               AgentState
                   Updated state containing the model's response message.
               """
        response = self._chat_model_with_tools.invoke([self._system_prompt] + state['messages'])
        return {"messages": [response]}

    def _should_continue(self, state: AgentState) -> str:
        """
              Determines whether the agent should continue the reasoning loop or stop.

              Parameters
              ----------
              state : AgentState
                  The current agent state with message history.

              Returns
              -------
              str
                  'continue' if there are pending tool calls, otherwise 'end'.
              """
        last_msg = state['messages'][-1]
        # If the last message has no tool calls, stop
        if (self._tool_calls_made >= self._max_tool_calls) or not getattr(last_msg, 'tool_calls', None):
            return 'end'
        else:
            self._tool_calls_made += 1
            return 'continue'

    def invoke(self, query: str):
        """
               Runs the debugging process for a given buggy Python code snippet.

               The agent:
                 - Receives the user's query as a message.
                 - Passes it through the LLM and sandbox tool pipeline.
                 - Returns the fixed, verified Python code.

               Parameters
               ----------
               query : str
                   The buggy Python code to be fixed.

               Returns
               -------
               str
                   The corrected, executable Python code.
               """
        self._tool_calls_made = 0
        self._state['messages'].append(HumanMessage(query))
        self._state = self._agent.invoke(self._state)
        return self._state['messages'][-1].content

    def reset_state(self):
        """
                Resets the agent's conversation and internal state.

                This is useful between independent debugging sessions to clear context.
                """
        self._state['messages'] = []
