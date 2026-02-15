# RAGtune Clean Runner Makefile

.PHONY: run-terrier run-pyterrier run-langchain run-active-learning run clean

# This "macro" ensures that PYTHONVERBOSE is disabled even if set in your environment.
# It suppresses the "# destroy" and "# cleanup" interpreter shutdown messages.
PYTHON_CLEAN = PYTHONVERBOSE=0 venv/bin/python

# Aliases for PyTerrier demo
run-terrier: run-pyterrier
run-pyterrier:
	@echo "--- Running PyTerrier BRIGHT Demo ---"
	@$(PYTHON_CLEAN) examples/demo_pyterrier_bright.py

run-scaled-terrier:
	@echo "--- Running Scaled PyTerrier Demo (ir_datasets) ---"
	@$(PYTHON_CLEAN) examples/demo_pyterrier_ird.py

run-langchain:
	@echo "--- Running LangChain Demo ---"
	@$(PYTHON_CLEAN) examples/demo_langchain_retriever.py

run-active-learning:
	@echo "--- Running Active Learning Demo ---"
	@$(PYTHON_CLEAN) examples/demo_active_learning.py

# Generic target: make run SCRIPT=examples/my_script.py
run:
	@if [ -z "$(SCRIPT)" ]; then echo "Usage: make run SCRIPT=path/to/script.py"; exit 1; fi
	@$(PYTHON_CLEAN) $(SCRIPT)

clean:
	rm -f *.log
