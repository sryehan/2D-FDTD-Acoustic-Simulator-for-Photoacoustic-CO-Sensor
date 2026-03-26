# Install testing dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/ --cov=src --cov-report=html