# Setup Guide for Flat Syntax Project

## Environment Created ✓

A conda environment named `flat_syntax` has been created with:
- **Python**: 3.11.14
- **Core dependencies**: requests, click, pyyaml

## Activation

To activate the environment:

```bash
conda activate flat_syntax
```

## Installed Packages

```
certifi            2025.10.5
charset-normalizer 3.4.4
click              8.3.0
idna               3.11
pip                25.2
PyYAML             6.0.3
requests           2.32.5
setuptools         80.9.0
urllib3            2.5.0
wheel              0.45.1
```

## Environment Files

Two files are available for environment management:

1. **requirements.txt** - Pip dependencies only
   ```bash
   conda activate flat_syntax
   pip install -r requirements.txt
   ```

2. **environment.yml** - Full conda environment specification
   ```bash
   # To recreate environment on another machine:
   conda env create -f environment.yml
   ```

## External Services Required

### 1. Trankit Parser Service (Docker)

**Status**: Should already be running

**Default URL**: `http://localhost:5000`

**Test connectivity**:
```bash
curl http://localhost:5000/health
# or
curl http://localhost:5000
```

If not running, start your Docker container with Trankit parser.

### 2. Ollama Service (Local LLM)

**Status**: Should already be running with llama3.1:8b

**Default URL**: `http://localhost:11434`

**Test connectivity**:
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Check if llama3.1:8b is available
ollama list | grep llama3.1:8b
```

**If not installed**:
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull llama3.1:8b model
ollama pull llama3.1:8b

# Start Ollama (if not running as service)
ollama serve
```

## Verification Checklist

Before starting development, verify:

- [ ] Conda environment `flat_syntax` is activated
- [ ] Python version is 3.11.14: `python --version`
- [ ] All pip packages installed: `pip list`
- [ ] Trankit service accessible: `curl http://localhost:5000`
- [ ] Ollama service accessible: `curl http://localhost:11434/api/tags`
- [ ] Model llama3.1:8b available: `ollama list`

## Project Structure

Current structure:
```
flat_syntax/
├── discussion.md           # Theoretical background
├── implementation.md       # Technical specification
├── QUICKSTART.md          # Quick reference guide
├── README.md              # Project overview
├── SETUP.md               # This file
├── requirements.txt       # Pip dependencies
└── environment.yml        # Conda environment spec
```

To be created during implementation:
```
flat_syntax/
├── flat_syntax/           # Main package directory
│   ├── __init__.py
│   ├── trankit_client.py
│   ├── converter.py
│   ├── boundaries.py
│   ├── ce_labelers.py
│   ├── formatters.py
│   ├── cli.py
│   ├── llm_client.py
│   ├── refinement.py
│   ├── disambiguators.py
│   └── config.py
├── tests/                 # Test suite
│   ├── test_data/
│   ├── gold_annotations/
│   └── test_pipeline.py
├── examples/              # Examples and few-shot data
│   ├── few_shot_examples.json
│   └── sample_annotations.md
└── setup.py              # Package installation
```

## Next Steps

1. **Verify services are running**
   ```bash
   # Test Trankit
   curl http://localhost:5000

   # Test Ollama
   curl http://localhost:11434/api/tags
   ```

2. **Create package structure**
   ```bash
   mkdir -p flat_syntax tests examples
   touch flat_syntax/__init__.py
   ```

3. **Start implementation** (Phase 1: MVP)
   - Begin with `trankit_client.py`
   - Document Trankit API format
   - Create test sentences

## Troubleshooting

### Environment activation fails
```bash
# List all conda environments
conda env list

# If flat_syntax doesn't exist, recreate from environment.yml
conda env create -f environment.yml
```

### Missing dependencies
```bash
conda activate flat_syntax
pip install -r requirements.txt
```

### Trankit service not responding
- Check if Docker container is running: `docker ps`
- Check container logs: `docker logs <container_name>`
- Restart container if needed

### Ollama service not responding
```bash
# Check if running
ps aux | grep ollama

# Start service
ollama serve

# Or restart if running as systemd service
sudo systemctl restart ollama
```

## Development Workflow

```bash
# 1. Activate environment
conda activate flat_syntax

# 2. Make changes to code
vim flat_syntax/trankit_client.py

# 3. Test changes
python -m flat_syntax.trankit_client

# 4. Run tests (when available)
pytest tests/

# 5. Deactivate when done
conda deactivate
```

## Adding New Dependencies

```bash
# Activate environment
conda activate flat_syntax

# Install new package
pip install <package_name>

# Update requirements.txt
pip freeze > requirements.txt

# Update environment.yml
conda env export > environment.yml
```

## Ready to Start!

Your environment is set up and ready. Proceed to implementation following the plan in `QUICKSTART.md`.
