# StampedeZero — Makefile
# =======================
# Usage:
#   make run           → start Streamlit app
#   make demo          → run standalone webcam tracker (no Streamlit)
#   make test          → run all tests
#   make test-tracker  → run VisionTracker unit tests only
#   make test-alert    → run alert engine integration tests
#   make install       → install all dependencies
#   make clean         → remove __pycache__ and .pyc files

.PHONY: run demo test test-tracker test-alert install clean

run:
	streamlit run app.py

demo:
	python demo.py

test: test-tracker test-alert

test-tracker:
	python -m pytest test_tracker.py -v

test-alert:
	cd alert_engine && python -m pytest test_integration.py -v

install:
	pip install -r requirements.txt

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "*.pyo" -delete 2>/dev/null || true

lint:
	python -m flake8 crowd_tracker.py config.py app.py alert_engine/ heatmap_engine/ --max-line-length=100
