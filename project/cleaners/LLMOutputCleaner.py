from typing import Optional
import json
import re


class LLOutputCleaner:

    @staticmethod
    def extract_tool_call_from_content(content: str) -> Optional[dict]:
        """
        Extracts a tool call JSON object after </think>.
        Handles cases where the JSON is preceded by text or explanations.
        Returns the parsed tool call as a dict, or None if not found/invalid.
        """
        if "</think>" not in content:
            return None

        after_think = content.split("</think>", 1)[1].strip()

        # Try to extract a JSON object (find the first '{' and last '}')
        json_match = re.search(r"\{.*\}", after_think, re.DOTALL)
        if not json_match:
            return None

        json_text = json_match.group(0)

        # Preprocess: escape control characters in JSON string (like raw newlines in Python code)
        def escape_newlines_in_json_string(match):
            inner = match.group(0)
            escaped = inner.replace("\n", "\\n").replace("\r", "\\r")
            return escaped

        json_text = re.sub(r'\"(.*?)\"', escape_newlines_in_json_string, json_text, flags=re.DOTALL)

        try:
            tool_call = json.loads(json_text)
        except json.JSONDecodeError as e:
            print("JSON decode error:", e)
            return None

        # Normalize to expected format and add hardcoded id
        if isinstance(tool_call, dict):
            if "name" in tool_call and "arguments" in tool_call:
                if isinstance(tool_call["arguments"], dict):
                    tool_call["args"] = tool_call["arguments"]
                tool_call["id"] = "1905tool_call_id"  # <-- added hardcoded id
                tool_call['name'] = 'run_in_sandbox_tool'
                return tool_call

            if "code" in tool_call:
                return {
                    "id": "1905tool_call_id",  # <-- added hardcoded id
                    "name": "run_in_sandbox_tool",
                    "args": {"code": tool_call["code"], "timeout": 5}
                }

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

    @staticmethod
    def shrink_think(text: str) -> str:
        if "<think>" in text and "</think>" in text:
            before = text.split("<think>", 1)[0] + "<think>"
            think_content = text.split("<think>", 1)[1].split("</think>", 1)[0]
            after = text.split("</think>", 1)[1]  # preserve everything after </think>
            # truncate think content
            think_content = think_content[:1000] + ("..." if len(think_content) > 1000 else "")
            return before + think_content + "</think>" + after
        return text

    @staticmethod
    def _remove_thinking_tags(llm_output: str) -> str:
        """
        Removes <think>...</think> sections from the LLM output.
        Returns the cleaned string even if the tags are missing or incomplete.
        """
        if not isinstance(llm_output, str):
            return ""

        if "</think>" in llm_output:
            parts = llm_output.split("</think>", 1)
            return parts[1].strip()

        if "<think>" in llm_output:
            return llm_output.split("<think>", 1)[0].strip()

        return llm_output.strip()

    @staticmethod
    def _extract_code(llm_output: str) -> str:
        match = re.search(r"```(?:python)?\s*(.*?)\s*```", llm_output, re.DOTALL)
        if match:
            return match.group(1).strip()
        return llm_output.strip()

    def get_code_from_llm_output(self, llm_output: str) -> str:
        llm_output = self._remove_thinking_tags(llm_output)
        llm_code = self._extract_code(llm_output)
        return llm_code
