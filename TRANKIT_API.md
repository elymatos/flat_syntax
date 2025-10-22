# Trankit API Documentation

## Service Endpoint

**URL**: `http://localhost:8406/api/v1/parse`

**Method**: `POST`

**Content-Type**: `application/json`

## Request Format

```json
{
    "text": "Roberto marcou o gol contra no jogo",
    "language": "portuguese",
    "mwe_enabled": true
}
```

### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `text` | string | Yes | Input sentence or text to parse |
| `language` | string | Yes | Language code (e.g., "english", "portuguese") |
| `mwe_enabled` | boolean | Yes | Enable multiword expression detection |

## Response Format

### Top-Level Structure

```json
{
    "sentences": [...],      // Array of sentence objects
    "mwe_count": 1,          // Number of MWEs detected
    "mwes": [...],           // Array of MWE objects (if mwe_enabled=true)
    "language": "portuguese",
    "processing_time": 0.415
}
```

### Sentence Object

Each sentence contains:

```json
{
    "id": 1,                 // Sentence ID (1-indexed)
    "text": "...",           // Original sentence text
    "dspan": [0, 35],        // Document character span
    "tokens": [...]          // Array of token objects
}
```

### Token Object (Standard)

Standard token structure:

```json
{
    "id": 1,                           // Token ID (1-indexed within sentence)
    "text": "Roberto",                 // Surface form
    "upos": "PROPN",                   // Universal POS tag
    "xpos": "NNP",                     // Language-specific POS (English only)
    "feats": "Gender=Masc|Number=Sing", // Morphological features
    "head": 2,                         // ID of head token (0 = root)
    "deprel": "nsubj",                 // Dependency relation
    "dspan": [0, 7],                   // Document character span
    "span": [0, 7],                    // Sentence character span
    "lemma": "Roberto",                // Lemma/dictionary form
    "ner": "O"                         // Named entity tag (English only)
}
```

### Token Object (Part of MWE)

Tokens that are part of a multiword expression have additional fields:

```json
{
    "id": 4,
    "text": "gol",
    "upos": "NOUN",
    "feats": "Gender=Masc|Number=Sing",
    "head": 2,
    "deprel": "obj",
    "dspan": [17, 20],
    "span": [17, 20],
    "lemma": "gol",

    // MWE-specific fields
    "mwe_span": [3, 5],                // Token IDs in MWE (3-5 exclusive = tokens 3,4)
    "mwe_lemma": "gol contra",         // Full MWE lemma
    "mwe_pos": "NOUN",                 // MWE part of speech
    "mwe_type": "fixed",               // MWE type (fixed, flat, compound, etc.)
    "mwe_head": 3,                     // ID of MWE head token
    "mwe_position": 0                  // Position within MWE (0-indexed)
}
```

### Token Object (Contracted Forms)

Some languages have contracted forms (e.g., Portuguese "no" = "em" + "o"):

```json
{
    "id": [6, 7],                      // Array of IDs for expanded tokens
    "text": "no",                      // Surface form (contracted)
    "span": [28, 30],
    "dspan": [28, 30],
    "expanded": [                      // Array of expanded tokens
        {
            "id": 6,
            "text": "em",              // First part: preposition
            "upos": "ADP",
            "head": 8,
            "deprel": "case",
            "lemma": "em"
        },
        {
            "id": 7,
            "text": "o",               // Second part: article
            "upos": "DET",
            "feats": "Definite=Def|Gender=Masc|Number=Sing|PronType=Art",
            "head": 8,
            "deprel": "det",
            "lemma": "o"
        }
    ]
}
```

### MWE Object

Summary information about detected multiword expressions:

```json
{
    "span": [3, 5],                    // Token IDs (3-5 exclusive = tokens 3,4)
    "text": "gol contra",              // Surface form
    "lemma": "gol contra",             // Lemma form
    "pos": "NOUN",                     // Part of speech
    "type": "fixed",                   // MWE type
    "tokens": ["gol", "contra"]        // Individual token texts
}
```

## MWE Types

Based on Universal Dependencies standards:

| Type | Description | Example |
|------|-------------|---------|
| `fixed` | Fixed multiword expressions | "gol contra" (own goal), "on top of" |
| `flat` | Flat multiword expressions (names) | "New York", "São Paulo" |
| `compound` | Compound words | "ice cream", "mother-in-law" |

## Language-Specific Features

### English
- Includes `xpos` (Penn Treebank POS tags)
- Includes `ner` (Named Entity Recognition tags)
- No contracted forms (expanded field)

### Portuguese
- No `xpos` field
- No `ner` field
- Has contracted forms (e.g., "no" = "em" + "o")
- MWEs detected (e.g., "gol contra")

## Example Responses

### Portuguese Example

**Input:**
```json
{
    "text": "Roberto marcou o gol contra no jogo",
    "language": "portuguese",
    "mwe_enabled": true
}
```

**Key Features:**
- Sentence: "Roberto marcou o gol contra no jogo"
- Translation: "Roberto scored the own goal in the game"
- MWE detected: "gol contra" (own goal) - tokens 4-5
- Contracted form: "no" → "em" + "o" (in + the)

**Dependency Structure:**
```
Roberto (1) --nsubj--> marcou (2, ROOT)
o (3) --det--> gol (4) --obj--> marcou (2)
contra (5) --case--> gol (4)
em (6) --case--> jogo (8)
o (7) --det--> jogo (8)
jogo (8) --obl--> marcou (2)
```

### English Example

**Input:**
```json
{
    "text": "The dog chased the cat.",
    "language": "english",
    "mwe_enabled": true
}
```

**Key Features:**
- Simple sentence
- No MWEs detected
- NER tags included (all "O" = outside entity)
- XPOS tags included (Penn Treebank)

**Dependency Structure:**
```
The (1) --det--> dog (2) --nsubj--> chased (3, ROOT)
the (4) --det--> cat (5) --obj--> chased (3)
. (6) --punct--> chased (3)
```

## Handling Special Cases in Code

### 1. Contracted Forms

```python
def process_token(token):
    """Process a token, handling contracted forms."""
    if isinstance(token['id'], list):
        # This is a contracted form with expanded tokens
        return process_contracted_token(token)
    else:
        # Standard token
        return process_standard_token(token)
```

### 2. MWE Detection

```python
def detect_mwes(sentence_data):
    """Extract MWEs from sentence."""
    mwes = []
    for token in sentence_data['tokens']:
        if 'mwe_span' in token and token.get('mwe_position') == 0:
            # This token is the head of an MWE
            mwes.append({
                'span': token['mwe_span'],
                'lemma': token['mwe_lemma'],
                'pos': token['mwe_pos'],
                'type': token['mwe_type']
            })
    return mwes

    # Or use the top-level 'mwes' field directly
    return sentence_data.get('mwes', [])
```

### 3. Token ID Handling

```python
def get_token_id(token):
    """Get token ID, handling both single and contracted forms."""
    token_id = token['id']
    if isinstance(token_id, list):
        # Contracted form: use first ID
        return token_id[0]
    return token_id
```

## Field Summary Table

| Field | Type | Present In | Description |
|-------|------|------------|-------------|
| `id` | int or list | All tokens | Token ID (list for contracted) |
| `text` | string | All tokens | Surface form |
| `upos` | string | All tokens | Universal POS tag |
| `xpos` | string | English only | Language-specific POS |
| `feats` | string | Most tokens | Morphological features |
| `head` | int | All tokens | Head token ID |
| `deprel` | string | All tokens | Dependency relation |
| `span` | [int, int] | All tokens | Character span in sentence |
| `dspan` | [int, int] | All tokens | Character span in document |
| `lemma` | string | All tokens | Dictionary form |
| `ner` | string | English only | Named entity tag |
| `expanded` | array | Contracted only | Expanded token list |
| `mwe_span` | [int, int] | MWE tokens | Token IDs in MWE |
| `mwe_lemma` | string | MWE tokens | Full MWE lemma |
| `mwe_pos` | string | MWE tokens | MWE part of speech |
| `mwe_type` | string | MWE tokens | MWE type |
| `mwe_head` | int | MWE tokens | MWE head token ID |
| `mwe_position` | int | MWE tokens | Position in MWE |

## Integration Notes

### For Flat Syntax Converter

1. **MWE Handling**: The `mwes` array at the top level provides clean MWE information. Use this to join tokens with `^` in flat syntax (e.g., "gol^contra").

2. **Contracted Forms**: When processing tokens, check if `id` is a list. If so, use the `expanded` array for dependency analysis.

3. **Token Iteration**: Be careful with token IDs when contracted forms are present. IDs may not be consecutive (e.g., 1, 2, 3, 4, 5, [6,7], 8).

4. **Dependency Relations**: The `head` and `deprel` fields provide the dependency tree structure needed for boundary detection and CE label assignment.

5. **Morphological Features**: Parse the `feats` field (pipe-separated key=value pairs) for detailed morphological information needed for CE labeling.

## Testing Checklist

- [ ] Test with simple English sentence
- [ ] Test with simple Portuguese sentence
- [ ] Test with MWE in English (e.g., "on top of")
- [ ] Test with MWE in Portuguese (e.g., "gol contra")
- [ ] Test with contracted form in Portuguese (e.g., "no", "da", "do")
- [ ] Test with complex sentence with multiple clauses
- [ ] Test with relative clause (center-embedding)
- [ ] Test error handling (invalid language, empty text, etc.)
