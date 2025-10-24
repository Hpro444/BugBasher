import configparser
import os


class Configuration:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Configuration, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True

        config_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.filepath = os.path.join(config_dir, "config.ini")

        self.config_parser = configparser.ConfigParser()

        # Default values
        self.ollama_url = "http://192.168.1.32:11434"
        self.ollama_model = "qwen3:0.6b"

        self.load()

    def load(self):
        """Load config.ini or create it with defaults if missing."""
        if os.path.exists(self.filepath):
            self.config_parser.read(self.filepath)

            self.ollama_url = self.config_parser.get(
                "Ollama", "ollama_url", fallback=self.ollama_url
            )
            self.ollama_model = self.config_parser.get(
                "Ollama", "ollama_model", fallback=self.ollama_model
            )
        else:
            self.save()

    def save(self):
        """Save current configuration values to config.ini."""
        self.config_parser["Ollama"] = {
            "ollama_url": self.ollama_url,
            "ollama_model": self.ollama_model,
        }

        with open(self.filepath, "w") as configfile:
            self.config_parser.write(configfile)
