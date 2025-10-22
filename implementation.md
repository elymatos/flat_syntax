# Flat Syntax Implementation Plan

## Overview

This document describes the practical implementation strategy for William Croft's "Flat Syntax" annotation scheme, adapted to use:
- **Trankit parser** as a Docker service for UD parsing
- **Ollama** (local LLM) for few-shot refinement using Llama 3.1:8B

## Architecture

```
Input Text
    ↓
[1] Trankit Parser Service (Docker) → Universal Dependencies Parse
    ↓
[2] UD-to-Flat Syntax Converter (Rule-based) → Preliminary Flat Syntax
    ↓
[3] Ollama LLM Refinement (Few-shot with Llama 3.1:8B) → Disambiguate ambiguous cases
    ↓
Output: Final Flat Syntax Annotation (CLDF/JSON/Table format)
```

## Key Design Decisions

### 1. **No Model Training Required** ✓
- Uses **few-shot learning** instead of fine-tuning
- Pre-existing Llama 3.1:8B model via Ollama (no training/fine-tuning needed)
- 3-5 examples provided in prompts to guide disambiguation
- Faster to implement, flexible, and cost-effective

### 2. **Local-First Architecture** ✓
- **Trankit service**: Docker container (localhost) - Already running
- **Ollama LLM**: Local inference (localhost) - Already running with llama3.1:8b
- **No external APIs**: Everything runs on your machine
- **Private & free**: No data leaves your system, no API costs

### 3. **Hybrid Approach** ✓
- **Rule-based converter**: Handles ~70-80% of cases mechanically
- **LLM refinement**: Only for ambiguous semantic/pragmatic decisions
- **Efficient**: LLM only called when necessary

### 4. **Phased Testing Strategy** ✓
- **Phase 1**: Rule-based baseline (measure accuracy without LLM)
- **Phase 2**: Add Ollama few-shot (measure improvement)
- **Phase 3**: Compare models (llama3.1:8b vs 70b vs mistral)
- **Phase 4**: Optional cloud comparison (GPT-4, Claude)

---

## PHASE 1: Trankit Parser Service Integration

### Service Architecture

**Docker Container**: Already running Trankit parser
- **Input**: HTTP/REST API endpoint accepting sentence text
- **Output**: Universal Dependencies parse in CoNLL-U format
- **Language Support**: Multilingual (Trankit supports 100+ languages)

### 1.1 Service Interface Design

```python
# Client module for Trankit service
class TrankitClient:
    """
    Client for communicating with Trankit parser Docker service.
    """

    def __init__(self, service_url: str, language: str = 'english'):
        """
        Initialize Trankit client.

        Args:
            service_url: URL of Docker service (e.g., 'http://localhost:5000')
            language: Language code for parsing
        """
        self.service_url = service_url
        self.language = language
        self.session = requests.Session()

    def parse(self, text: str) -> Dict:
        """
        Parse text and return UD parse.

        Args:
            text: Input sentence or text

        Returns:
            Dictionary containing:
            - tokens: List of token dictionaries
            - sentences: Sentence boundaries
            - dependencies: UD dependency tree
        """
        payload = {
            'text': text,
            'language': self.language
        }

        response = self.session.post(
            f'{self.service_url}/parse',
            json=payload,
            timeout=30
        )

        if response.status_code != 200:
            raise TrankitServiceError(
                f"Parser returned status {response.status_code}: {response.text}"
            )

        return response.json()

    def parse_to_conllu(self, text: str) -> str:
        """
        Parse text and return CoNLL-U format string.

        Args:
            text: Input sentence or text

        Returns:
            CoNLL-U format string
        """
        result = self.parse(text)
        return self._format_conllu(result)

    def _format_conllu(self, parse_result: Dict) -> str:
        """Convert parser output to CoNLL-U format."""
        # Implementation depends on Trankit service output format
        pass
```

### 1.2 Expected Input/Output Format

#### Input to Service:
```json
{
  "text": "The two brothers might lose the game and exit the competition.",
  "language": "english"
}
```

#### Output from Service (Expected Structure):
```json
{
  "sentences": [
    {
      "tokens": [
        {
          "id": 1,
          "text": "The",
          "lemma": "the",
          "upos": "DET",
          "xpos": "DT",
          "feats": "Definite=Def|PronType=Art",
          "head": 3,
          "deprel": "det",
          "start_char": 0,
          "end_char": 3
        },
        {
          "id": 2,
          "text": "two",
          "lemma": "two",
          "upos": "NUM",
          "xpos": "CD",
          "feats": "NumType=Card",
          "head": 3,
          "deprel": "nummod",
          "start_char": 4,
          "end_char": 7
        },
        // ... more tokens
      ]
    }
  ]
}
```

### 1.3 Error Handling

```python
class TrankitServiceError(Exception):
    """Exception raised when Trankit service fails."""
    pass

def parse_with_retry(client: TrankitClient, text: str, max_retries: int = 3):
    """
    Parse text with retry logic for service failures.
    """
    for attempt in range(max_retries):
        try:
            return client.parse(text)
        except (requests.Timeout, requests.ConnectionError) as e:
            if attempt == max_retries - 1:
                raise TrankitServiceError(
                    f"Failed to reach parser service after {max_retries} attempts"
                ) from e
            time.sleep(2 ** attempt)  # Exponential backoff
```

### 1.4 Service Configuration

```python
# config.py
from dataclasses import dataclass

@dataclass
class ServiceConfig:
    """Configuration for Trankit parser service."""

    # Service endpoint
    url: str = "http://localhost:5000"

    # Default language
    language: str = "english"

    # Timeout settings
    timeout: int = 30

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0

    # Batch processing
    batch_size: int = 100

    @classmethod
    def from_env(cls):
        """Load configuration from environment variables."""
        import os
        return cls(
            url=os.getenv('TRANKIT_SERVICE_URL', cls.url),
            language=os.getenv('TRANKIT_LANGUAGE', cls.language),
            timeout=int(os.getenv('TRANKIT_TIMEOUT', cls.timeout)),
        )
```

---

## PHASE 2: UD-to-Flat Syntax Converter

### 2.1 Core Converter Module

```python
class FlatSyntaxConverter:
    """
    Converts Universal Dependencies parse to Flat Syntax annotation.
    """

    def __init__(self, language: str = 'english'):
        self.language = language
        self.boundary_detector = BoundaryDetector()
        self.ce_labeler = CELabeler()

    def convert(self, ud_parse: Dict) -> FlatSyntaxAnnotation:
        """
        Main conversion pipeline.

        Args:
            ud_parse: UD parse from Trankit service

        Returns:
            FlatSyntaxAnnotation object
        """
        # Step 1: Add boundary markers
        boundaries = self.boundary_detector.detect_boundaries(ud_parse)

        # Step 2: Assign CE labels (three tiers)
        phrasal_ces = self.ce_labeler.label_phrasal(ud_parse, boundaries)
        clausal_ces = self.ce_labeler.label_clausal(ud_parse, boundaries)
        sentential_ces = self.ce_labeler.label_sentential(ud_parse, boundaries)

        # Step 3: Detect special patterns (interruption, MWE)
        interruptions = self.detect_interruptions(ud_parse, boundaries)
        mwes = self.detect_multiword_expressions(ud_parse)

        return FlatSyntaxAnnotation(
            tokens=ud_parse['sentences'][0]['tokens'],
            boundaries=boundaries,
            phrasal_ces=phrasal_ces,
            clausal_ces=clausal_ces,
            sentential_ces=sentential_ces,
            interruptions=interruptions,
            multiword_expressions=mwes
        )
```

### 2.2 Boundary Detection

```python
class BoundaryDetector:
    """Detects phrase, clause, and sentence boundaries from UD parse."""

    def detect_boundaries(self, ud_parse: Dict) -> BoundaryMarkers:
        """
        Add boundary markers (+, #, .) based on UD structure.
        """
        tokens = ud_parse['sentences'][0]['tokens']
        boundaries = BoundaryMarkers()

        # Detect phrase boundaries ('+')
        phrase_boundaries = self._detect_phrase_boundaries(tokens)
        boundaries.add_phrase_boundaries(phrase_boundaries)

        # Detect clause boundaries ('#')
        clause_boundaries = self._detect_clause_boundaries(tokens)
        boundaries.add_clause_boundaries(clause_boundaries)

        # Sentence boundary ('.')
        boundaries.add_sentence_boundary(len(tokens))

        return boundaries

    def _detect_phrase_boundaries(self, tokens: List[Dict]) -> List[int]:
        """
        Identify phrase boundaries using UD subtrees.

        Algorithm:
        1. For each token, get its syntactic subtree
        2. If subtree forms a valid phrase constituent, mark boundary
        3. Handle coordination specially
        """
        boundaries = []

        for i, token in enumerate(tokens):
            # Get subtree span
            subtree = self._get_subtree(token, tokens)

            # Check if this is the last token in a phrase
            if self._is_phrase_final(token, subtree, tokens):
                boundaries.append(i)

        return boundaries

    def _detect_clause_boundaries(self, tokens: List[Dict]) -> List[int]:
        """
        Identify clause boundaries.

        Algorithm:
        1. Find all predicates (VERB/AUX with specific deprels)
        2. Collect their arguments and adjuncts
        3. Mark clause boundary after last element
        """
        boundaries = []

        # Find all predicates
        predicates = self._find_predicates(tokens)

        for pred in predicates:
            # Get clause span (pred + all args)
            clause_span = self._get_clause_span(pred, tokens)
            boundaries.append(clause_span.end)

        return boundaries

    def _is_phrase_final(self, token: Dict, subtree: List[Dict],
                         all_tokens: List[Dict]) -> bool:
        """
        Determine if token is the last element of its phrase.
        """
        # Check if no other tokens in subtree follow this one
        max_id = max(t['id'] for t in subtree)
        return token['id'] == max_id
```

### 2.3 CE Label Mapping

```python
class CELabeler:
    """Assigns Construction Element labels at three tiers."""

    def __init__(self):
        self.phrasal_mapper = PhrasalCEMapper()
        self.clausal_mapper = ClausualCEMapper()
        self.sentential_mapper = SententialCEMapper()

    def label_phrasal(self, ud_parse: Dict, boundaries: BoundaryMarkers) -> List[str]:
        """Assign phrasal CE labels (Head, Mod, Adm, Adp, etc.)"""
        return self.phrasal_mapper.map(ud_parse, boundaries)

    def label_clausal(self, ud_parse: Dict, boundaries: BoundaryMarkers) -> List[str]:
        """Assign clausal CE labels (Pred, Arg, CPP, Gen, etc.)"""
        return self.clausal_mapper.map(ud_parse, boundaries)

    def label_sentential(self, ud_parse: Dict, boundaries: BoundaryMarkers) -> List[str]:
        """Assign sentential CE labels (Main, Adv, Rel, Comp, etc.)"""
        return self.sentential_mapper.map(ud_parse, boundaries)


class PhrasalCEMapper:
    """Maps UD features to Phrasal CE labels."""

    # Mapping rules from UD UPOS to Phrasal CE
    UPOS_MAPPING = {
        "DET": "Mod",
        "ADJ": "Mod",
        "NUM": "Mod",
        "ADP": "Adp",
        "CCONJ": "Conj",
        "NOUN": "Head",
        "PROPN": "Head",
        "PRON": "Head",
    }

    def map(self, ud_parse: Dict, boundaries: BoundaryMarkers) -> List[str]:
        """Map each token to phrasal CE label."""
        tokens = ud_parse['sentences'][0]['tokens']
        labels = []

        for token in tokens:
            label = self._map_token(token, tokens)
            labels.append(label)

        return labels

    def _map_token(self, token: Dict, all_tokens: List[Dict]) -> str:
        """
        Map single token to phrasal CE label.

        Priority:
        1. Check for special deprels (nmod:poss → Gen)
        2. Check for context (advmod + modifying ADJ → Adm)
        3. Use UPOS mapping
        4. Default to Head for content words
        """
        # Special case: possessive
        if token['deprel'] == 'nmod:poss':
            return "Gen"

        # Special case: admodifier
        if token['upos'] == 'ADV' and token['deprel'] == 'advmod':
            head = self._get_head_token(token, all_tokens)
            if head and head['upos'] in ['ADJ', 'ADV']:
                return "Adm"

        # Default UPOS mapping
        return self.UPOS_MAPPING.get(token['upos'], "Head")


class ClausualCEMapper:
    """Maps UD features to Clausal CE labels."""

    DEPREL_MAPPING = {
        # Arguments
        "nsubj": "Arg",
        "obj": "Arg",
        "iobj": "Arg",
        "obl": "Arg",  # May need disambiguation (Arg vs FPM)

        # Predicate
        "root": "Pred",

        # Complex predicate parts
        "aux": "CPP",
        "cop": "CPP",

        # Modifiers
        "nmod:poss": "Gen",
        "nmod": "FPM",  # With case marking

        # Coordination
        "cc": "Conj",
    }

    def map(self, ud_parse: Dict, boundaries: BoundaryMarkers) -> List[str]:
        """Map phrases to clausal CE labels."""
        tokens = ud_parse['sentences'][0]['tokens']

        # Group tokens into phrases using boundaries
        phrases = self._group_into_phrases(tokens, boundaries)

        # Map each phrase
        labels = []
        for phrase in phrases:
            label = self._map_phrase(phrase, tokens)
            labels.append(label)

        return labels

    def _map_phrase(self, phrase: List[Dict], all_tokens: List[Dict]) -> str:
        """
        Map phrase to clausal CE label.

        Uses the head token of the phrase for determination.
        """
        head = self._find_phrase_head(phrase)

        # Check deprel mapping
        if head['deprel'] in self.DEPREL_MAPPING:
            return self.DEPREL_MAPPING[head['deprel']]

        # Default based on UPOS
        if head['upos'] == 'VERB':
            return "Pred"
        else:
            return "Arg"


class SententialCEMapper:
    """Maps UD features to Sentential CE labels."""

    DEPREL_MAPPING = {
        "root": "Main",
        "acl:relcl": "Rel",
        "acl": "Rel",
        "advcl": "Adv",
        "ccomp": "Comp",
        "xcomp": "Comp",
        "parataxis": "Main",
        "vocative": "Int",
        "discourse": "Int",
        "dislocated": "Dtch",
    }

    def map(self, ud_parse: Dict, boundaries: BoundaryMarkers) -> List[str]:
        """Map clauses to sentential CE labels."""
        tokens = ud_parse['sentences'][0]['tokens']

        # Group tokens into clauses using boundaries
        clauses = self._group_into_clauses(tokens, boundaries)

        # Map each clause
        labels = []
        for clause in clauses:
            label = self._map_clause(clause, tokens)
            labels.append(label)

        return labels

    def _map_clause(self, clause: List[Dict], all_tokens: List[Dict]) -> str:
        """Map clause to sentential CE label."""
        # Find main predicate of clause
        predicate = self._find_clause_predicate(clause)

        if not predicate:
            return "Main"  # Default

        # Use deprel of predicate
        return self.DEPREL_MAPPING.get(predicate['deprel'], "Main")
```

### 2.4 Special Pattern Detection

```python
class InterruptionDetector:
    """Detects center-embedding/interruption patterns."""

    def detect(self, ud_parse: Dict, boundaries: BoundaryMarkers) -> List[Interruption]:
        """
        Find interrupting constructions requiring { } markers.

        Algorithm:
        1. For each clause/phrase, get its span
        2. Check if any child clause/phrase has non-contiguous parent
        3. Mark as interruption if child tokens fall within parent's span
        """
        tokens = ud_parse['sentences'][0]['tokens']
        interruptions = []

        for token in tokens:
            # Check for relative clauses and other subordinate clauses
            if token['deprel'] in ['acl:relcl', 'acl', 'advcl']:
                parent_span = self._get_span(token['head'], tokens)
                child_span = self._get_subtree_span(token, tokens)

                # Check if child interrupts parent
                if self._is_interrupting(child_span, parent_span):
                    interruptions.append(Interruption(
                        start=child_span.start,
                        end=child_span.end,
                        type=token['deprel']
                    ))

        return interruptions

    def _is_interrupting(self, child_span: Span, parent_span: Span) -> bool:
        """Check if child span interrupts parent span."""
        return (child_span.start > parent_span.start and
                child_span.end < parent_span.end)


class MultiwordExpressionDetector:
    """Detects fixed multiword expressions."""

    def detect(self, ud_parse: Dict) -> List[MWE]:
        """
        Find multiword expressions to join with ^.

        UD relations indicating MWE:
        - fixed: fixed multiword expressions
        - flat: flat multiword expressions (names)
        - compound: some compounds
        """
        tokens = ud_parse['sentences'][0]['tokens']
        mwes = []

        for token in tokens:
            if token['deprel'] in ['fixed', 'flat', 'compound:prt']:
                # Find the full MWE span
                mwe_tokens = self._get_mwe_tokens(token, tokens)
                mwes.append(MWE(
                    tokens=mwe_tokens,
                    type=token['deprel']
                ))

        return mwes
```

---

## PHASE 3: LLM-Based Refinement

### 3.1 Refinement Pipeline

```python
class LLMRefiner:
    """
    Refines rule-based flat syntax annotation using LLM.
    Focuses on ambiguous cases requiring semantic/pragmatic reasoning.
    """

    def __init__(self, llm_client, few_shot_examples: FewShotExamples):
        self.llm = llm_client
        self.examples = few_shot_examples
        self.disambiguators = {
            'ARG_vs_FPM': ArgFPMDisambiguator(llm_client, few_shot_examples),
            'MAIN_vs_ADV': MainAdvDisambiguator(llm_client, few_shot_examples),
            'PRED_vs_CPP': PredCPPDisambiguator(llm_client, few_shot_examples),
        }

    def refine(self, annotation: FlatSyntaxAnnotation,
               ud_parse: Dict) -> FlatSyntaxAnnotation:
        """
        Two-stage refinement:
        1. Targeted disambiguation of known ambiguous patterns
        2. Holistic review for consistency
        """
        # Stage 1: Identify and fix ambiguous patterns
        problems = self._detect_ambiguous_patterns(annotation, ud_parse)

        for problem in problems:
            disambiguator = self.disambiguators.get(problem.type)
            if disambiguator:
                correction = disambiguator.disambiguate(problem, annotation, ud_parse)
                annotation.apply_correction(correction)

        # Stage 2: Holistic review (optional, for complex sentences)
        if self._needs_holistic_review(annotation):
            annotation = self._holistic_review(annotation, ud_parse)

        return annotation

    def _detect_ambiguous_patterns(self, annotation: FlatSyntaxAnnotation,
                                   ud_parse: Dict) -> List[Problem]:
        """
        Identify patterns requiring LLM disambiguation.

        Patterns:
        1. Clausal CE labeled 'Arg' with UD deprel 'obl' → Arg vs FPM
        2. Coordinate clauses or subordinate with assertion → Main vs Adv
        3. Multiple verbs in clause → Pred vs CPP
        4. Non-contiguous spans → Interruption detection
        """
        problems = []

        # Check for Arg vs FPM ambiguity
        for i, ce_label in enumerate(annotation.clausal_ces):
            token = ud_parse['sentences'][0]['tokens'][i]
            if ce_label == 'Arg' and token['deprel'] == 'obl':
                problems.append(Problem(
                    type='ARG_vs_FPM',
                    token_id=i,
                    context={'token': token, 'ce_label': ce_label}
                ))

        # Check for Main vs Adv ambiguity
        # ... similar logic

        return problems
```

### 3.2 Disambiguation Modules

```python
class ArgFPMDisambiguator:
    """Disambiguates Arg vs FPM for oblique phrases."""

    def __init__(self, llm_client, few_shot_examples):
        self.llm = llm_client
        self.examples = few_shot_examples.get_category('ARG_vs_FPM')

    def disambiguate(self, problem: Problem, annotation: FlatSyntaxAnnotation,
                     ud_parse: Dict) -> Correction:
        """
        Use LLM to determine if oblique is Arg or FPM.
        """
        prompt = self._build_prompt(problem, annotation, ud_parse)
        response = self.llm.complete(prompt, temperature=0)

        # Parse LLM response
        decision = self._parse_response(response)

        return Correction(
            token_id=problem.token_id,
            old_label='Arg',
            new_label=decision['label'],
            rationale=decision['rationale']
        )

    def _build_prompt(self, problem: Problem, annotation: FlatSyntaxAnnotation,
                     ud_parse: Dict) -> str:
        """
        Build few-shot prompt for disambiguation.
        """
        token = problem.context['token']
        sentence = self._reconstruct_sentence(ud_parse)

        prompt = f"""You are a linguistic annotator specializing in Flat Syntax.

Task: Determine if an oblique phrase is an Argument (Arg) or Flagged Phrase Modifier (FPM).

Principles:
- Arg: Participant/entity involved in the event (required by verb semantics)
- FPM: Circumstantial modifier (location, time, manner - optional)

Examples:
{self._format_examples(self.examples)}

Current Sentence: {sentence}
Oblique Phrase: "{token['text']}" (depends on verb "{self._get_verb(token, ud_parse)}")

Question: Is this phrase an Arg (participant) or FPM (modifier)?

Provide answer in format:
LABEL: [Arg or FPM]
RATIONALE: [Brief explanation]
"""
        return prompt

    def _format_examples(self, examples: List[Example]) -> str:
        """Format few-shot examples for prompt."""
        formatted = []
        for ex in examples:
            formatted.append(f"""
Example:
Sentence: {ex.sentence}
Oblique: "{ex.oblique_phrase}"
Label: {ex.label}
Reason: {ex.rationale}
""")
        return "\n".join(formatted)
```

### 3.3 LLM Client Interface

```python
class LLMClient:
    """
    Abstract interface for LLM providers.
    Primary: Ollama (local)
    Optional: OpenAI, Anthropic (cloud)
    """

    def complete(self, prompt: str, temperature: float = 0,
                 max_tokens: int = 500) -> str:
        """Get completion from LLM."""
        raise NotImplementedError


class OllamaClient(LLMClient):
    """
    Ollama local LLM client (PRIMARY).

    Connects to local Ollama service for private, free inference.
    Default model: llama3.1:8b
    """

    def __init__(self, model: str = "llama3.1:8b",
                 base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama client.

        Args:
            model: Ollama model name (e.g., 'llama3.1:8b', 'llama3.1:70b', 'mistral')
            base_url: Ollama service URL (default: http://localhost:11434)
        """
        self.model = model
        self.base_url = base_url
        self.session = requests.Session()

    def complete(self, prompt: str, temperature: float = 0,
                 max_tokens: int = 500) -> str:
        """
        Get completion from Ollama.

        Makes POST request to /api/generate endpoint.
        """
        response = self.session.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "temperature": temperature,
                "stream": False,
                "options": {
                    "num_predict": max_tokens
                }
            },
            timeout=60
        )

        if response.status_code == 200:
            return response.json()["response"]
        else:
            raise OllamaServiceError(
                f"Ollama returned status {response.status_code}: {response.text}"
            )

    def is_available(self) -> bool:
        """Check if Ollama service is running."""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def list_models(self) -> List[str]:
        """List available models in Ollama."""
        response = self.session.get(f"{self.base_url}/api/tags")
        if response.status_code == 200:
            return [model['name'] for model in response.json()['models']]
        return []


class OpenAIClient(LLMClient):
    """OpenAI API client (OPTIONAL - for comparison)."""

    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def complete(self, prompt: str, temperature: float = 0,
                 max_tokens: int = 500) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content


class AnthropicClient(LLMClient):
    """Anthropic Claude API client (OPTIONAL - for comparison)."""

    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def complete(self, prompt: str, temperature: float = 0,
                 max_tokens: int = 500) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text


class OllamaServiceError(Exception):
    """Exception raised when Ollama service fails."""
    pass
```

---

## PHASE 4: Output Formatting

### 4.1 CLDF Format Output

```python
class CLDFFormatter:
    """Formats flat syntax annotation as CLDF-compatible CSV."""

    @staticmethod
    def to_csv(annotation: FlatSyntaxAnnotation, output_path: str):
        """
        Export annotation to CLDF CSV format.

        Format:
        - Row 1: Analyzed_Text (with boundary markers)
        - Row 2: Phrasal_CEs
        - Row 3: Clausal_CEs
        - Row 4: Sentential_CEs
        - Row 5: Translation (optional)
        """
        rows = [
            ['Analyzed_Text', annotation.to_text_with_boundaries()],
            ['Phrasal_CEs', ' '.join(annotation.phrasal_ces)],
            ['Clausal_CEs', ' '.join(annotation.clausal_ces)],
            ['Sentential_CEs', ' '.join(annotation.sentential_ces)],
        ]

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(rows)

    @staticmethod
    def to_table(annotation: FlatSyntaxAnnotation) -> str:
        """
        Format as aligned table (markdown-style).

        Example output:
        ```
        Analyzed Text:  The two brothers + might + lose + the game .
        Phrasal CEs:    Mod Mod Head      Head    Head   Mod  Head  Head
        Clausal CEs:    Arg               CPP     Pred   Arg
        Sentential CEs: Main
        ```
        """
        lines = []

        # Get tokens with boundaries
        text_line = "Analyzed Text:  " + annotation.to_text_with_boundaries()
        lines.append(text_line)

        # Align CE labels under tokens
        phrasal_line = "Phrasal CEs:    " + annotation.align_labels(
            annotation.phrasal_ces
        )
        lines.append(phrasal_line)

        clausal_line = "Clausal CEs:    " + annotation.align_labels(
            annotation.clausal_ces, level='clausal'
        )
        lines.append(clausal_line)

        sentential_line = "Sentential CEs: " + annotation.align_labels(
            annotation.sentential_ces, level='sentential'
        )
        lines.append(sentential_line)

        return '\n'.join(lines)
```

### 4.2 JSON Output

```python
class JSONFormatter:
    """Formats annotation as structured JSON."""

    @staticmethod
    def to_json(annotation: FlatSyntaxAnnotation, output_path: str):
        """
        Export as JSON with full annotation detail.

        Includes:
        - Original tokens
        - Boundary markers
        - CE labels at all tiers
        - Metadata (language, parser version, etc.)
        """
        data = {
            'tokens': [
                {
                    'id': t.id,
                    'text': t.text,
                    'lemma': t.lemma,
                    'upos': t.upos,
                    'phrasal_ce': annotation.phrasal_ces[t.id - 1],
                    'clausal_ce': annotation.clausal_ces[t.id - 1],
                    'sentential_ce': annotation.sentential_ces[t.id - 1],
                }
                for t in annotation.tokens
            ],
            'boundaries': {
                'phrase': annotation.boundaries.phrase_boundaries,
                'clause': annotation.boundaries.clause_boundaries,
                'sentence': annotation.boundaries.sentence_boundaries,
            },
            'interruptions': [
                {'start': i.start, 'end': i.end, 'type': i.type}
                for i in annotation.interruptions
            ],
            'multiword_expressions': [
                {'tokens': mwe.token_ids, 'text': mwe.text}
                for mwe in annotation.multiword_expressions
            ],
            'metadata': {
                'language': annotation.language,
                'parser': 'Trankit',
                'timestamp': datetime.now().isoformat(),
            }
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
```

---

## PHASE 5: Complete Pipeline Integration

### 5.1 Main Pipeline

```python
class FlatSyntaxPipeline:
    """
    Complete pipeline from text to flat syntax annotation.
    """

    def __init__(self, config: PipelineConfig):
        # Initialize components
        self.trankit_client = TrankitClient(
            service_url=config.trankit_url,
            language=config.language
        )
        self.converter = FlatSyntaxConverter(language=config.language)

        if config.use_llm_refinement:
            llm_client = self._init_llm_client(config.llm_config)
            few_shot_examples = FewShotExamples.load(config.few_shot_path)
            self.refiner = LLMRefiner(llm_client, few_shot_examples)
        else:
            self.refiner = None

    def annotate(self, text: str) -> FlatSyntaxAnnotation:
        """
        Annotate text with flat syntax.

        Pipeline:
        1. Parse with Trankit service
        2. Convert UD to Flat Syntax (rule-based)
        3. Refine with LLM (optional)
        4. Return annotation
        """
        # Step 1: Parse
        ud_parse = self.trankit_client.parse(text)

        # Step 2: Convert
        annotation = self.converter.convert(ud_parse)

        # Step 3: Refine (if enabled)
        if self.refiner:
            annotation = self.refiner.refine(annotation, ud_parse)

        return annotation

    def annotate_batch(self, texts: List[str],
                       batch_size: int = 100) -> List[FlatSyntaxAnnotation]:
        """
        Annotate multiple texts efficiently.
        """
        annotations = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            # Parse batch
            ud_parses = self._parse_batch(batch)

            # Convert batch
            batch_annotations = [
                self.converter.convert(ud) for ud in ud_parses
            ]

            # Refine batch (if enabled)
            if self.refiner:
                batch_annotations = [
                    self.refiner.refine(ann, ud)
                    for ann, ud in zip(batch_annotations, ud_parses)
                ]

            annotations.extend(batch_annotations)

        return annotations

    def _parse_batch(self, texts: List[str]) -> List[Dict]:
        """Parse multiple texts with Trankit service."""
        # Can be parallelized if service supports concurrent requests
        return [self.trankit_client.parse(text) for text in texts]
```

### 5.2 Configuration

```python
@dataclass
class PipelineConfig:
    """Configuration for complete pipeline."""

    # Trankit service
    trankit_url: str = "http://localhost:5000"
    language: str = "english"

    # LLM refinement
    use_llm_refinement: bool = True
    llm_config: Optional[LLMConfig] = None
    few_shot_path: Optional[str] = None

    # Output
    output_format: str = "json"  # json, csv, table
    output_path: Optional[str] = None

    # Processing
    batch_size: int = 100
    num_workers: int = 1

    @classmethod
    def from_yaml(cls, path: str):
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)


@dataclass
class LLMConfig:
    """Configuration for LLM client."""

    provider: str = "ollama"  # ollama (default), openai, anthropic
    model: str = "llama3.1:8b"  # Default Ollama model
    base_url: str = "http://localhost:11434"  # Ollama service URL
    api_key: Optional[str] = None  # Only for OpenAI/Anthropic
    temperature: float = 0
    max_tokens: int = 500
```

### 5.3 Command-Line Interface

```python
import click

@click.command()
@click.argument('input_text', type=str)
@click.option('--config', '-c', type=click.Path(exists=True),
              help='Path to configuration YAML file')
@click.option('--output', '-o', type=click.Path(),
              help='Output file path')
@click.option('--format', '-f', type=click.Choice(['json', 'csv', 'table']),
              default='table', help='Output format')
@click.option('--no-llm', is_flag=True,
              help='Disable LLM refinement (rule-based only)')
def annotate(input_text: str, config: str, output: str,
             format: str, no_llm: bool):
    """
    Annotate text with Flat Syntax.

    Example:
        flat-syntax annotate "The dog chased the cat." --format table
    """
    # Load configuration
    if config:
        pipeline_config = PipelineConfig.from_yaml(config)
    else:
        pipeline_config = PipelineConfig()

    if no_llm:
        pipeline_config.use_llm_refinement = False

    # Initialize pipeline
    pipeline = FlatSyntaxPipeline(pipeline_config)

    # Annotate
    annotation = pipeline.annotate(input_text)

    # Output
    if format == 'table':
        result = CLDFFormatter.to_table(annotation)
        click.echo(result)
    elif format == 'json':
        if output:
            JSONFormatter.to_json(annotation, output)
            click.echo(f"Saved to {output}")
        else:
            click.echo(json.dumps(annotation.to_dict(), indent=2))
    elif format == 'csv':
        if output:
            CLDFFormatter.to_csv(annotation, output)
            click.echo(f"Saved to {output}")
        else:
            click.echo("CSV format requires --output path")


@click.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--config', '-c', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), required=True)
@click.option('--batch-size', '-b', type=int, default=100)
def annotate_file(input_file: str, config: str, output: str, batch_size: int):
    """
    Annotate sentences from a file (one per line).

    Example:
        flat-syntax annotate-file sentences.txt -o output.json
    """
    # Load configuration
    if config:
        pipeline_config = PipelineConfig.from_yaml(config)
    else:
        pipeline_config = PipelineConfig()

    pipeline_config.batch_size = batch_size

    # Load sentences
    with open(input_file) as f:
        sentences = [line.strip() for line in f if line.strip()]

    # Initialize pipeline
    pipeline = FlatSyntaxPipeline(pipeline_config)

    # Annotate
    click.echo(f"Annotating {len(sentences)} sentences...")
    with click.progressbar(sentences) as bar:
        annotations = pipeline.annotate_batch(list(bar))

    # Save output
    with open(output, 'w', encoding='utf-8') as f:
        json.dump([ann.to_dict() for ann in annotations], f, indent=2)

    click.echo(f"Saved {len(annotations)} annotations to {output}")


@click.group()
def cli():
    """Flat Syntax Annotation Tool"""
    pass

cli.add_command(annotate)
cli.add_command(annotate_file)

if __name__ == '__main__':
    cli()
```

---

## IMPLEMENTATION ROADMAP

### Week 1: Trankit Service Integration
- [ ] Implement `TrankitClient` with REST API communication
- [ ] Test service connectivity and response parsing
- [ ] Handle error cases (timeout, connection errors)
- [ ] Create configuration module

### Week 2: Rule-Based Converter
- [ ] Implement `BoundaryDetector` (phrase, clause, sentence boundaries)
- [ ] Implement `CELabeler` (three-tier mapping from UD)
- [ ] Implement special pattern detectors (interruption, MWE)
- [ ] Create test suite with 20-30 example sentences

### Week 3: Output Formatting
- [ ] Implement CLDF CSV formatter
- [ ] Implement JSON formatter
- [ ] Implement table formatter (human-readable)
- [ ] Create CLI interface

### Week 4: Ollama LLM Integration
- [ ] Implement `OllamaClient` with local service communication
- [ ] Test Ollama connectivity and model availability
- [ ] Design few-shot prompts for key ambiguities
- [ ] Implement disambiguation modules (Arg vs FPM, Main vs Adv, etc.)
- [ ] Create 20-50 gold standard examples for few-shot learning

### Week 5: Testing & Refinement
- [ ] Comprehensive testing on diverse sentences
- [ ] Error analysis and rule refinement
- [ ] Performance optimization (batch processing)
- [ ] Documentation

---

## DEPENDENCIES

```
# requirements.txt
# Core dependencies
requests>=2.31.0
click>=8.1.0
pyyaml>=6.0

# Optional: Cloud LLM providers (for comparison/benchmarking only)
# openai>=1.0.0
# anthropic>=0.7.0
```

### External Services Required

1. **Trankit Parser Service** (Docker)
   - Must be running on `http://localhost:5000` (configurable)
   - Provides UD parsing via REST API

2. **Ollama Service** (Local LLM)
   - Must be running on `http://localhost:11434` (default)
   - Model `llama3.1:8b` must be downloaded
   - Install: `curl -fsSL https://ollama.com/install.sh | sh`
   - Pull model: `ollama pull llama3.1:8b`

---

## EXAMPLE USAGE

### Basic Usage (CLI)

```bash
# Annotate single sentence
flat-syntax annotate "The tree that fell on our house had died."

# Output:
# Analyzed Text:  The tree + {that + fell + on {our} house} had + died .
# Phrasal CEs:    Mod Head   Head   Head  Adp Head Head   Head  Head  Head
# Clausal CEs:    Arg        Conj   Pred  Arg   Gen      CPP   Pred
# Sentential CEs: Main       Rel                          Main

# Save as JSON
flat-syntax annotate "The dog chased the cat." -o output.json -f json

# Annotate file (batch)
flat-syntax annotate-file sentences.txt -o annotations.json
```

### Python API

```python
from flat_syntax import FlatSyntaxPipeline, PipelineConfig, LLMConfig

# Initialize pipeline with Ollama (default)
config = PipelineConfig(
    trankit_url="http://localhost:5000",
    language="english",
    use_llm_refinement=True,
    llm_config=LLMConfig(
        provider="ollama",
        model="llama3.1:8b",
        base_url="http://localhost:11434"
    )
)
pipeline = FlatSyntaxPipeline(config)

# Annotate single sentence
annotation = pipeline.annotate("The dog chased the cat.")

# Print formatted output
from flat_syntax import CLDFFormatter
print(CLDFFormatter.to_table(annotation))

# Batch annotation
sentences = [
    "The dog ran.",
    "She walked to school.",
    "The tree that fell had died."
]
annotations = pipeline.annotate_batch(sentences)

# Export to CSV
for i, ann in enumerate(annotations):
    CLDFFormatter.to_csv(ann, f"annotation_{i}.csv")
```

### Configuration File Example

```yaml
# config.yaml
trankit_url: http://localhost:5000
language: english

use_llm_refinement: true

llm_config:
  provider: ollama
  model: llama3.1:8b
  base_url: http://localhost:11434
  temperature: 0
  max_tokens: 500

output_format: json
batch_size: 100
```

Then use with:
```bash
flat-syntax annotate "Some text." --config config.yaml
```

---

## TESTING STRATEGY

### Phase 1: Rule-Based Baseline (No LLM)

**Goal**: Establish baseline accuracy with pure rule-based conversion

```python
# Test without LLM
config = PipelineConfig(
    trankit_url="http://localhost:5000",
    use_llm_refinement=False  # Disable LLM
)
pipeline = FlatSyntaxPipeline(config)
annotation = pipeline.annotate("The dog chased the cat.")
```

**Metrics to collect**:
- Boundary detection accuracy (%, #, +)
- CE label accuracy (phrasal, clausal, sentential)
- Processing time per sentence
- Common error patterns

### Phase 2: Ollama Few-Shot Enhancement

**Goal**: Measure improvement with Llama 3.1:8B few-shot disambiguation

```python
# Test with Ollama
config = PipelineConfig(
    trankit_url="http://localhost:5000",
    use_llm_refinement=True,
    llm_config=LLMConfig(provider="ollama", model="llama3.1:8b")
)
pipeline = FlatSyntaxPipeline(config)
```

**Test cases** (20-50 sentences covering):
1. Simple sentences (baseline)
2. Arg vs FPM ambiguity
3. Main vs Adv coordination
4. Complex predicates
5. Center-embedding/interruption
6. Multiword expressions

**Compare**:
- Accuracy improvement over rule-based
- Types of errors fixed by LLM
- Types of errors introduced by LLM
- Processing time increase

### Phase 3: Model Comparison (Optional)

**Goal**: Compare different Ollama models

```python
models_to_test = [
    "llama3.1:8b",      # Default (fast, good)
    "llama3.1:70b",     # Larger (slower, better?)
    "mistral",          # Alternative architecture
    "mixtral",          # Mixture of experts
]

for model in models_to_test:
    config.llm_config.model = model
    # Run test suite and collect metrics
```

**Metrics**:
- Accuracy per model
- Inference time per model
- Memory usage per model

### Phase 4: Cloud LLM Comparison (Optional)

**Goal**: See if commercial models significantly outperform local

```python
# Compare Ollama vs GPT-4 vs Claude
providers = [
    ("ollama", "llama3.1:8b", None),
    ("openai", "gpt-4", api_key),
    ("anthropic", "claude-3-5-sonnet-20241022", api_key)
]
```

**Decision criteria**:
- Is accuracy improvement worth the cost?
- Are local models "good enough"?

### Evaluation Metrics

```python
# Automated evaluation against gold standard
class FlatSyntaxEvaluator:
    def evaluate(self, predicted: FlatSyntaxAnnotation,
                 gold: FlatSyntaxAnnotation) -> Metrics:
        """
        Compare predicted vs gold annotation.

        Returns:
            - boundary_f1: F1 score for boundary markers
            - phrasal_ce_acc: Accuracy of phrasal CE labels
            - clausal_ce_acc: Accuracy of clausal CE labels
            - sentential_ce_acc: Accuracy of sentential CE labels
            - exact_match: Percentage of perfectly annotated sentences
        """
        pass
```

### Test Suite Structure

```
tests/
├── test_data/
│   ├── simple_sentences.txt       # 20 simple sentences
│   ├── ambiguous_obliques.txt     # 20 Arg vs FPM cases
│   ├── coordination.txt           # 20 Main vs Adv cases
│   ├── complex_predicates.txt     # 20 Pred vs CPP cases
│   └── interruptions.txt          # 20 center-embedding cases
├── gold_annotations/
│   └── *.json                     # Manually verified annotations
└── test_pipeline.py               # Automated test runner
```

---

## PHASED IMPLEMENTATION PLAN

### Phase 1: MVP (Rule-Based Only) - Week 1-3

**Deliverable**: Working pipeline without LLM
- Trankit client
- UD-to-Flat converter
- Output formatters
- CLI interface

**Test**: 20 simple sentences, measure baseline accuracy

### Phase 2: Ollama Integration - Week 4

**Deliverable**: Add Llama 3.1:8B few-shot disambiguation
- OllamaClient implementation
- Few-shot prompt design
- Disambiguation modules for top 3 ambiguities

**Test**: 50 diverse sentences, compare with Phase 1 baseline

### Phase 3: Refinement - Week 5

**Deliverable**: Improved accuracy and robustness
- Error analysis from Phase 2
- Refine rules and prompts
- Add more few-shot examples
- Handle edge cases

**Test**: 100+ sentence test suite, final evaluation

### Phase 4: Optional Enhancements - Future

- Support for multiple languages
- Fine-tuned models (if needed)
- Web interface
- Batch processing optimizations
- Conversion to other formats (UD, CoNLL)

---

## COST & PERFORMANCE ANALYSIS

### Local Ollama (llama3.1:8b)
- **Cost**: $0 (free)
- **Speed**: ~1-2 seconds per sentence (depends on hardware)
- **Privacy**: All data stays local
- **Requirements**: ~8GB VRAM or 16GB RAM

### Cloud APIs (for comparison)
- **GPT-4**: ~$0.03-0.06 per 1000 tokens (~$3-6 per 1000 sentences)
- **Claude 3.5 Sonnet**: Similar pricing
- **Speed**: ~0.5-1 second per sentence
- **Privacy**: Data sent to external service

### Recommendation
Start with **local Ollama** (free, private, fast enough). Only consider cloud if:
1. Accuracy significantly lower than needed
2. Processing large-scale corpora (millions of sentences)
3. Budget allows and privacy not a concern

---

## NEXT STEPS

1. **Confirm Trankit Service API**: Document the exact endpoint structure and response format of your Trankit Docker service
2. **Set Up Test Environment**:
   - Verify Trankit service is running
   - Verify Ollama is running with llama3.1:8b
   - Create initial test sentence set (20 sentences)
3. **Begin Implementation**: Start with Phase 1 MVP (rule-based only)
4. **Create Gold Standard**: Manually annotate 20-50 test sentences

Would you like me to proceed with implementing any specific component first?
