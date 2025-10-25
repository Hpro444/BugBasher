from typing import Optional
import json


class LLOutputCleaner:

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
