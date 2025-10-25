import streamlit as st
from project.agent.model import BugBashAgent
from project.cleaners import LLOutputCleaner
from project.eval.evaluator import BugFixEvaluator


class PythonBugFixerGUI:
    def __init__(self):
        self.pages = {
            "Fix Code": self.fix_code_page,
            "Evaluate Model": self.eval_page
        }
        # Initialize reusable components
        self.cleaner = LLOutputCleaner()

    def run(self):
        st.sidebar.title("🐞 Python Bug Fixer AI")
        page = st.sidebar.radio("Select a page:", list(self.pages.keys()))
        self.pages[page]()

    # ------------------- PAGE 1: FIX CODE -------------------
    def fix_code_page(self):
        st.title("Fix Python Code 🛠️")

        # Let user select or enter a model name
        model_name = st.text_input("Model name to use for fixing:", "qwen3:8b")

        buggy_code = st.text_area("Paste your buggy Python code here:", height=200)

        if st.button("Fix Code"):
            if not buggy_code.strip():
                st.warning("Please paste some code first!")
                return

            with st.spinner(f"Fixing code using model '{model_name}'... 🔧"):
                try:
                    # Initialize agent with chosen model
                    agent = BugBashAgent(model=model_name)
                    fixed_code = agent.invoke(buggy_code)
                    fixed_code = self.cleaner.get_code_from_llm_output(fixed_code)
                except Exception as e:
                    st.error(f"Error during fixing: {e}")
                    return

            st.subheader("✅ Fixed Code")
            st.code(fixed_code, language="python")

            # Store for evaluation or re-use later
            st.session_state["last_fixed_code"] = fixed_code
            st.session_state["last_model_used"] = model_name

    # ------------------- PAGE 2: EVALUATE MODEL -------------------
    def eval_page(self):
        st.title("Evaluate Model 🧠")

        model_name = st.text_input("Enter model name to evaluate:", "qwen3:8b")
        num_tests = st.number_input("Number of tests (0 = full dataset):", min_value=0, value=0)

        if st.button("Run Evaluation"):
            with st.spinner(f"Loading model '{model_name}'..."):
                agent = BugBashAgent(model=model_name)
                evaluator = BugFixEvaluator(agent=agent)

            progress_bar = st.progress(0)
            progress_text = st.empty()
            status_box = st.empty()

            result = evaluator.evaluate_for_gui(
                progress_bar=progress_bar,
                progress_text=progress_text,
                status_box=status_box,
                num_tests=num_tests or None
            )

            st.subheader("📊 Results")
            st.markdown(f"**Model:** `{result.model_name}`")
            st.markdown(f"**Passed:** {result.passed} / {result.total}")
            st.markdown(f"**Pass Rate:** {result.score_in_percentage}")
            st.markdown(f"**Score (pass@1):** {result.score_pass1:.3f}")
