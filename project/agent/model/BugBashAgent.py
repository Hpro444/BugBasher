from typing import List, Optional
from project.cleaners import LLOutputCleaner
from project.sandbox import run_in_sandbox_tool
from langchain_core.tools.structured import BaseTool
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

    def __init__(self, model: Optional[str] = None, ollama_url: Optional[str] = None, tools: Optional[List[BaseTool]] = None, max_tool_calls: int = 5, use_persistent_memory: bool = False):
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
         use_persistent_memory : bool, default=False
            If `True`, enables the agent to persist context or conversation state
            between sessions, allowing for incremental debugging and multi-step reasoning.
         """
        self._use_persistent_memory = use_persistent_memory
        self._model = model or Configuration().ollama_model
        ollama_url = ollama_url or Configuration().ollama_url
        self._tools = tools or [run_in_sandbox_tool]
        self._chat_model_with_tools = ChatOllama(model=self._model, base_url=ollama_url).bind_tools(self._tools)
        self._state: AgentState = AgentState(messages=[], last_executed_code='')
        self._system_prompt = SystemMessage(content="""
        You are CodeFixer, an autonomous Python debugging and repair assistant.
        Your sole purpose is to fix Python code so it runs correctly and produces the desired output.

        You have access to one tool:
        run_in_sandbox(code: str, timeout: int = 5) -> str
        - Executes Python code safely (Docker if available, otherwise a virtual environment).
        - Returns:
            - Full program output if execution succeeds.
            - 'Success (no output)' if code runs correctly but has no output.
            - Full traceback or detailed error if execution fails.
            - 'Error: Execution timed out' if it exceeds the time limit.

        ==============================
        CORE INSTRUCTIONS
        ==============================
        1. Only output valid, executable Python code.
        2. Preserve the original logic and intent while fixing all errors.
        3. Always run the fixed code using run_in_sandbox.
        4. Compare the tool's output to the expected output:
           - If the output matches exactly, return **only the fixed code**.
           - If the output is incorrect or errors occur, fix the code so it will produce the correct output.
        5. If the code defines only functions/classes, automatically wrap it minimally so it produces output for verification (e.g., call the main function and print the result).
        6. Do not include explanations, comments, temporary print statements, or testing harnesses.
        7. Do not include "if __name__ == '__main__':" unless necessary to produce output.
        8. Python version is 3.12 with standard libraries only.

        ==============================
        STRICT OUTPUT RULES
        ==============================
        - Return only the fixed, working Python code.
        - Never return the tool output or any additional text.
        - Ensure the returned code is directly executable and produces the correct output in run_in_sandbox.
        """)
        self._max_tool_calls = max_tool_calls
        self._tool_calls_made = 0
        self._cleaner = LLOutputCleaner()
        self._agent = None
        self._compile_graph()

    def _compile_graph(self):
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

    def _tool_code_fixer_node(self, state: AgentState) -> AgentState:
        if not state['messages']:
            return state

        last_msg = state['messages'][-1]
        tool_call = self._cleaner.extract_tool_call_from_content(last_msg.content)

        if tool_call:
            # Modify in-place
            last_msg.tool_call = tool_call
            state['messages'][-1] = last_msg  # optional, already in-place

        if state['messages'][-1].tool_calls:
            state['last_executed_code'] = state['messages'][-1].tool_calls[0]['args']['code']

        return state

    def _update_system_prompt(self, state: AgentState):
        """
        Updates the system prompt with context about the last executed code attempt.

        This helps the model understand what code it last tried to run and improve
        its next fix attempt accordingly.
        """
        last_code = state.get("last_executed_code", "").strip()

        if last_code:
            self._system_prompt = SystemMessage(content=f"""
            You are CodeFixer, an autonomous Python debugging assistant.

            Rules :
            - Fix Python code so it runs and produces the expected output.
            - Always run the fixed code with run_in_sandbox(code: str, timeout: int = 5) and carefully check the tool response.
            - Focus on the output part — it determines success.
            - If the original code only defines functions/classes, call them (e.g., run a simple main) so the sandbox can verify output.
            - When the tool shows the output matches the expected result, return ONLY the fixed Python code (no explanations, no extra text).
            - If it fails or errors, keep fixing and re-running until the sandbox output is correct.
            - Use Python 3.12 standard library.

            Last attempt at fixing:
            ```python
            {last_code}
           ==============================
        STRICT OUTPUT RULES
        ==============================
        - Return only the fixed, working Python code.
        - Never return the tool output or any additional text.
        - Ensure the returned code is directly executable and produces the correct output in run_in_sandbox.
            """)

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
        self._update_system_prompt(state)
        response = self._chat_model_with_tools.invoke([self._system_prompt] + state['messages'])
        return {"messages": [response], "last_executed_code": state['last_executed_code']}

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
        output = self._state['messages'][-1].content
        if self._use_persistent_memory:
            self._reset_state()
        return output

    def _reset_state(self):
        """
                Resets the agent's conversation and internal state.

                This is useful between independent debugging sessions to clear context.
                """
        self._state['messages'] = []


    def get_model(self):
        return self._model