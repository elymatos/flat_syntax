# Flat Syntax - Quick Start Guide

## What is This?

A tool to automatically annotate text with William Croft's "Flat Syntax" scheme:
- Non-recursive, three-level syntactic annotation
- Uses Universal Dependencies as foundation
- Enhanced with local LLM (Ollama) for disambiguation
- No training required - uses few-shot learning

## Prerequisites

You already have:
- ✓ Trankit parser (Docker) - running on localhost
- ✓ Ollama with llama3.1:8b - running on localhost

## Architecture

```
Input: "The dog chased the cat."
    ↓
[Trankit Parser] → Universal Dependencies parse
    ↓
[Rule-Based Converter] → Preliminary flat syntax
    ↓
[Ollama LLM] → Disambiguate ambiguous cases (Arg vs FPM, Main vs Adv, etc.)
    ↓
Output:
Analyzed Text:  The dog + chased + the cat .
Phrasal CEs:    Mod Head  Head     Mod Head Head
Clausal CEs:    Arg       Pred     Arg
Sentential CEs: Main
```

## Key Points

### No Training Needed
- Uses **few-shot learning** (not fine-tuning)
- Provide 3-5 examples in prompts → LLM learns the pattern
- Llama 3.1:8B is sufficient for linguistic reasoning

### Everything Local
- Trankit: `http://localhost:5000`
- Ollama: `http://localhost:11434`
- No external API calls
- Free and private

### Hybrid Strategy
- Rules handle ~70-80% automatically
- LLM only for ambiguous cases
- Fast and accurate

## Installation Plan

```bash
# 1. Clone/create project
git clone <repo> flat_syntax
cd flat_syntax

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify services
curl http://localhost:5000/health     # Trankit
curl http://localhost:11434/api/tags  # Ollama

# 4. Test
python -m flat_syntax annotate "The dog ran."
```

## Implementation Phases

### Phase 1: MVP (Week 1-3)
**Goal**: Rule-based converter working
- Trankit client
- UD → Flat Syntax converter
- Output formatters (JSON, CSV, table)
- CLI interface

**Test**: 20 simple sentences, no LLM

### Phase 2: Ollama (Week 4)
**Goal**: Add LLM disambiguation
- OllamaClient
- Few-shot prompts for:
  - Arg vs FPM (participant vs modifier)
  - Main vs Adv (asserted vs background)
  - Pred vs CPP (main verb vs auxiliary)
- 20-50 gold examples

**Test**: 50 diverse sentences, compare improvement

### Phase 3: Refinement (Week 5)
**Goal**: Improve accuracy
- Error analysis
- Refine rules and prompts
- Handle edge cases
- Documentation

**Test**: 100+ sentences

## Example Usage (Planned)

```bash
# Simple annotation
flat-syntax annotate "The tree fell."

# Output:
# Analyzed Text:  The tree + fell .
# Phrasal CEs:    Mod Head   Head  Head
# Clausal CEs:    Arg        Pred
# Sentential CEs: Main

# Batch processing
flat-syntax annotate-file sentences.txt -o output.json

# Without LLM (rule-based only)
flat-syntax annotate "Some text." --no-llm

# With custom config
flat-syntax annotate "Some text." --config config.yaml
```

## Python API (Planned)

```python
from flat_syntax import FlatSyntaxPipeline, PipelineConfig

# Initialize
config = PipelineConfig(
    trankit_url="http://localhost:5000",
    use_llm_refinement=True
)
pipeline = FlatSyntaxPipeline(config)

# Annotate
annotation = pipeline.annotate("The dog chased the cat.")

# Output
print(annotation.to_table())
```

## Testing Strategy

### Step 1: Baseline (No LLM)
```python
config.use_llm_refinement = False
# Measure: boundary accuracy, label accuracy
```

### Step 2: With Ollama
```python
config.use_llm_refinement = True
config.llm_config.model = "llama3.1:8b"
# Measure: improvement, processing time
```

### Step 3: Compare Models (Optional)
```python
for model in ["llama3.1:8b", "llama3.1:70b", "mistral"]:
    config.llm_config.model = model
    # Compare accuracy vs speed
```

## Ambiguities LLM Will Handle

1. **Arg vs FPM** (most common)
   - "She walked **to school**" → Arg (destination/participant)
   - "The book **on the table**" → FPM (location/modifier)

2. **Main vs Adv** (pragmatic)
   - "I ate **and** left" → both Main (asserted events)
   - "**After I ate**, I left" → Adv + Main (background + assertion)

3. **Pred vs CPP** (complex predicates)
   - "She **might** have **been** running" → CPP CPP CPP Pred
   - "also wrecked" → CPP Pred

4. **Center-embedding** (interruption)
   - "The tree {that fell} died" → detect `{}`

## Files to Create

```
flat_syntax/
├── __init__.py
├── trankit_client.py       # Phase 1
├── converter.py            # Phase 1
├── boundaries.py           # Phase 1
├── ce_labelers.py          # Phase 1
├── formatters.py           # Phase 1
├── cli.py                  # Phase 1
├── llm_client.py           # Phase 2
├── refinement.py           # Phase 2
├── disambiguators.py       # Phase 2
└── config.py               # Phase 1

tests/
├── test_data/
│   ├── simple.txt
│   ├── ambiguous_obliques.txt
│   ├── coordination.txt
│   └── complex_predicates.txt
├── gold_annotations/
│   └── *.json
└── test_pipeline.py

examples/
├── few_shot_examples.json  # For LLM prompts
└── sample_annotations.md
```

## Next Steps

1. **Confirm Trankit API format**
   - What's the exact request/response structure?

2. **Create test sentences**
   - 20 simple sentences to start

3. **Start Phase 1 implementation**
   - Begin with `trankit_client.py`

4. **Manual gold annotations**
   - Annotate 20 test sentences by hand for evaluation

## Quick Reference: Flat Syntax Basics

### Boundary Markers
- `+` = phrase boundary (between words in clause)
- `#` = clause boundary (between clauses in sentence)
- `.` = sentence boundary
- `{}` = interruption/center-embedding
- `^` = multiword expression (e.g., on^top^of)

### CE Labels (3 Tiers)

**Phrasal** (word-level):
- Head, Mod, Adm, Adp, Lnk, Clf, Idx, Conj

**Clausal** (phrase-level):
- Pred, Arg, CPP, Gen, FPM, Conj

**Sentential** (clause-level):
- Main, Adv, Rel, Comp, Dtch, Int

## Resources

- Full documentation: `implementation.md`
- Theoretical background: `discussion.md`
- Croft's original paper: [link when available]
