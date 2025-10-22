# Flat Syntax Annotation Pipeline

Automatic annotation tool for William Croft's "Flat Syntax" scheme using:
- **Trankit parser** (Docker service) for Universal Dependencies parsing
- **Ollama** (local LLM with Llama 3.1:8B) for few-shot disambiguation
- **No training required** - uses rule-based conversion + few-shot learning

## Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Quick overview and getting started guide
- **[implementation.md](implementation.md)** - Complete technical specification and API design
- **[discussion.md](discussion.md)** - Theoretical background and Croft's Flat Syntax theory

## Key Features

✓ **Local-first**: Everything runs on localhost (private & free)
✓ **No training**: Uses few-shot learning instead of fine-tuning
✓ **Hybrid approach**: Rule-based (70-80%) + LLM disambiguation (20-30%)
✓ **Multiple outputs**: JSON, CSV (CLDF), human-readable tables

## Quick Example

```bash
# Annotate a sentence
flat-syntax annotate "The dog chased the cat."

# Output:
# Analyzed Text:  The dog + chased + the cat .
# Phrasal CEs:    Mod Head  Head     Mod Head Head
# Clausal CEs:    Arg       Pred     Arg
# Sentential CEs: Main
```

## Implementation Status

- [x] **Phase 1a: Trankit Service Integration** - COMPLETED ✓
  - [x] TrankitClient implementation
  - [x] Configuration module
  - [x] Comprehensive test suite (6/6 tests passing)
  - [x] English and Portuguese parsing validated
  - [x] MWE and contracted form handling
- [x] **Phase 1b: Rule-based converter** - COMPLETED ✓
  - [x] Boundary detection (phrase, clause, sentence, interruption)
  - [x] CE labeling (3 tiers: phrasal, clausal, sentential)
  - [x] Output formatters (JSON, CSV/CLDF, Table)
  - [x] Complete UD → Flat Syntax conversion pipeline
- [ ] **Phase 2: CLI and Integration** - NEXT
  - [ ] Command-line interface
  - [ ] Pipeline integration
  - [ ] Batch processing
- [ ] Phase 3: Ollama LLM integration
- [ ] Phase 4: Testing and refinement

## Prerequisites

- ✓ Ollama with llama3.1:8b - running on `http://localhost:11434`
- ✓ Trankit parser service (Docker) - running on `http://localhost:8406`

## Quick Start

```bash
# Activate conda environment
conda activate flat_syntax

# Test the Trankit client
python -m flat_syntax.trankit_client

# Run tests
pytest tests/ -v
```

## Documentation

- **[PROGRESS.md](PROGRESS.md)** - Current implementation status and completed tasks
- **[QUICKSTART.md](QUICKSTART.md)** - Detailed implementation plan and roadmap
- **[SETUP.md](SETUP.md)** - Environment setup guide
- **[TRANKIT_API.md](TRANKIT_API.md)** - Trankit service API documentation

## Current Features

✓ **Trankit Parser Integration**
- Parse English and Portuguese sentences
- Universal Dependencies output
- Multiword expression detection
- Contracted form handling
- Batch processing support
- Health monitoring and retry logic

✓ **Flat Syntax Converter**
- Rule-based UD → Flat Syntax conversion
- Boundary detection (phrase +, clause #, sentence .)
- Three-tier CE labeling (phrasal, clausal, sentential)
- MWE handling (joined with ^)
- Interruption detection (center-embedding)
- Multiple output formats (Table, JSON, CSV/CLDF)

## Example Usage

```python
from flat_syntax import TrankitClient, FlatSyntaxConverter, TableFormatter

# Initialize
client = TrankitClient(language="english")
converter = FlatSyntaxConverter(language="english")
formatter = TableFormatter()

# Parse and convert
sentence = "The dog chased the cat."
ud_parse = client.parse(sentence)
annotation = converter.convert(ud_parse, sentence)

# Display
print(formatter.format(annotation))
```

**Output:**
```
Analyzed Text:  The dog + chased the cat # . .
Phrasal CEs:    Mod Head  Head Mod Head  Head
Clausal CEs:    Arg   Pred    FPM
Sentential CEs: Main       Main
```

See [PROGRESS.md](PROGRESS.md) for detailed status and more examples.
