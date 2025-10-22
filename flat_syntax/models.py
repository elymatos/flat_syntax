"""
Data models for Flat Syntax annotations.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum


class BoundaryType(Enum):
    """Types of boundaries in Flat Syntax."""
    PHRASE = "+"      # Phrase boundary
    CLAUSE = "#"      # Clause boundary
    SENTENCE = "."    # Sentence boundary
    INTERRUPTION_START = "{"  # Start of interruption
    INTERRUPTION_END = "}"    # End of interruption
    MWE_CONNECTOR = "^"       # Multiword expression connector


@dataclass
class BoundaryMarkers:
    """
    Container for boundary markers in Flat Syntax.

    Stores positions where boundaries should be inserted between tokens.
    """
    phrase_boundaries: List[int] = field(default_factory=list)
    clause_boundaries: List[int] = field(default_factory=list)
    sentence_boundaries: List[int] = field(default_factory=list)
    interruptions: List[Tuple[int, int]] = field(default_factory=list)  # (start, end) pairs

    def add_phrase_boundary(self, position: int):
        """Add a phrase boundary after the given token position."""
        if position not in self.phrase_boundaries:
            self.phrase_boundaries.append(position)

    def add_phrase_boundaries(self, positions: List[int]):
        """Add multiple phrase boundaries."""
        for pos in positions:
            self.add_phrase_boundary(pos)

    def add_clause_boundary(self, position: int):
        """Add a clause boundary after the given token position."""
        if position not in self.clause_boundaries:
            self.clause_boundaries.append(position)

    def add_clause_boundaries(self, positions: List[int]):
        """Add multiple clause boundaries."""
        for pos in positions:
            self.add_clause_boundary(pos)

    def add_sentence_boundary(self, position: int):
        """Add a sentence boundary after the given token position."""
        if position not in self.sentence_boundaries:
            self.sentence_boundaries.append(position)

    def add_interruption(self, start: int, end: int):
        """Add an interruption/center-embedding marker."""
        self.interruptions.append((start, end))

    def get_boundary_at(self, position: int) -> Optional[str]:
        """
        Get the boundary marker at the given position.

        Priority: sentence > clause > phrase
        """
        if position in self.sentence_boundaries:
            return BoundaryType.SENTENCE.value
        elif position in self.clause_boundaries:
            return BoundaryType.CLAUSE.value
        elif position in self.phrase_boundaries:
            return BoundaryType.PHRASE.value
        return None


@dataclass
class Interruption:
    """Represents an interrupting construction (center-embedding)."""
    start: int  # Token position where interruption starts
    end: int    # Token position where interruption ends
    type: str   # Type of interruption (e.g., 'acl:relcl', 'advcl')


@dataclass
class MultiwordExpression:
    """Represents a multiword expression."""
    token_ids: List[int]  # Token IDs that form the MWE
    text: str             # Surface form
    lemma: str            # Lemma form
    pos: str              # Part of speech
    type: str             # MWE type (fixed, flat, compound)


@dataclass
class Token:
    """Enhanced token with Flat Syntax annotations."""
    id: int
    text: str
    lemma: str
    upos: str
    feats: Optional[str]
    head: int
    deprel: str

    # Flat Syntax labels
    phrasal_ce: Optional[str] = None
    clausal_ce: Optional[str] = None
    sentential_ce: Optional[str] = None

    # Additional info
    is_mwe_part: bool = False
    mwe_position: Optional[int] = None

    @classmethod
    def from_ud_token(cls, ud_token: Dict) -> 'Token':
        """Create Token from UD parse token dictionary."""
        # Handle contracted forms - use first token
        if isinstance(ud_token['id'], list):
            # For contracted forms, use the expanded tokens
            if 'expanded' in ud_token and ud_token['expanded']:
                first_expanded = ud_token['expanded'][0]
                return cls(
                    id=first_expanded['id'],
                    text=first_expanded['text'],
                    lemma=first_expanded['lemma'],
                    upos=first_expanded['upos'],
                    feats=first_expanded.get('feats'),
                    head=first_expanded['head'],
                    deprel=first_expanded['deprel']
                )

        return cls(
            id=ud_token['id'],
            text=ud_token['text'],
            lemma=ud_token['lemma'],
            upos=ud_token['upos'],
            feats=ud_token.get('feats'),
            head=ud_token['head'],
            deprel=ud_token['deprel']
        )


@dataclass
class Phrase:
    """Represents a phrase (group of tokens)."""
    tokens: List[Token]
    head_token: Token
    clausal_ce: Optional[str] = None

    @property
    def start_id(self) -> int:
        """Get the ID of the first token."""
        return self.tokens[0].id if self.tokens else 0

    @property
    def end_id(self) -> int:
        """Get the ID of the last token."""
        return self.tokens[-1].id if self.tokens else 0


@dataclass
class Clause:
    """Represents a clause (group of phrases)."""
    phrases: List[Phrase]
    predicate: Optional[Token] = None
    sentential_ce: Optional[str] = None

    @property
    def start_id(self) -> int:
        """Get the ID of the first token."""
        return self.phrases[0].start_id if self.phrases else 0

    @property
    def end_id(self) -> int:
        """Get the ID of the last token."""
        return self.phrases[-1].end_id if self.phrases else 0


@dataclass
class FlatSyntaxAnnotation:
    """
    Complete Flat Syntax annotation for a sentence.

    Contains:
    - Original tokens with CE labels at all three tiers
    - Boundary markers
    - Multiword expressions
    - Interruptions
    - Metadata
    """
    tokens: List[Token]
    boundaries: BoundaryMarkers
    multiword_expressions: List[MultiwordExpression] = field(default_factory=list)
    interruptions: List[Interruption] = field(default_factory=list)

    # Metadata
    language: str = "english"
    source_text: str = ""

    def to_text_with_boundaries(self) -> str:
        """
        Generate text representation with boundary markers.

        Example: "The dog + chased + the cat ."
        """
        result = []

        for i, token in enumerate(self.tokens):
            # Check if this token is part of an MWE
            mwe_tokens = self._get_mwe_tokens_at(i)

            if mwe_tokens:
                # Join MWE tokens with ^
                mwe_text = "^".join([t.text for t in mwe_tokens])
                result.append(mwe_text)
                # Skip the rest of the MWE tokens
                continue

            # Skip if this token was already added as part of MWE
            if self._is_inside_mwe(i):
                continue

            result.append(token.text)

            # Add boundary marker if present
            boundary = self.boundaries.get_boundary_at(i)
            if boundary:
                result.append(boundary)

        return " ".join(result)

    def _get_mwe_tokens_at(self, position: int) -> Optional[List[Token]]:
        """Get MWE tokens starting at position, or None."""
        for mwe in self.multiword_expressions:
            if mwe.token_ids and mwe.token_ids[0] == self.tokens[position].id:
                return [self.tokens[i] for i, t in enumerate(self.tokens)
                        if t.id in mwe.token_ids]
        return None

    def _is_inside_mwe(self, position: int) -> bool:
        """Check if token at position is inside an MWE (but not first token)."""
        token_id = self.tokens[position].id
        for mwe in self.multiword_expressions:
            if token_id in mwe.token_ids and mwe.token_ids[0] != token_id:
                return True
        return False

    def get_phrasal_ces(self) -> List[str]:
        """Get list of phrasal CE labels."""
        return [t.phrasal_ce or "" for t in self.tokens]

    def get_clausal_ces(self) -> List[str]:
        """Get list of clausal CE labels (at phrase level)."""
        # Group tokens into phrases and return CE for each phrase
        result = []
        current_phrase_ce = None

        for i, token in enumerate(self.tokens):
            if token.clausal_ce:
                current_phrase_ce = token.clausal_ce

            # Add CE label for this token's position
            result.append(current_phrase_ce or "")

            # Reset at phrase boundaries
            if i in self.boundaries.phrase_boundaries:
                current_phrase_ce = None

        return result

    def get_sentential_ces(self) -> List[str]:
        """Get list of sentential CE labels (at clause level)."""
        result = []
        current_clause_ce = None

        for i, token in enumerate(self.tokens):
            if token.sentential_ce:
                current_clause_ce = token.sentential_ce

            result.append(current_clause_ce or "")

            # Reset at clause boundaries
            if i in self.boundaries.clause_boundaries:
                current_clause_ce = None

        return result

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'tokens': [
                {
                    'id': t.id,
                    'text': t.text,
                    'lemma': t.lemma,
                    'upos': t.upos,
                    'phrasal_ce': t.phrasal_ce,
                    'clausal_ce': t.clausal_ce,
                    'sentential_ce': t.sentential_ce,
                }
                for t in self.tokens
            ],
            'boundaries': {
                'phrase': self.boundaries.phrase_boundaries,
                'clause': self.boundaries.clause_boundaries,
                'sentence': self.boundaries.sentence_boundaries,
            },
            'interruptions': [
                {'start': i.start, 'end': i.end, 'type': i.type}
                for i in self.interruptions
            ],
            'multiword_expressions': [
                {
                    'token_ids': mwe.token_ids,
                    'text': mwe.text,
                    'lemma': mwe.lemma,
                    'pos': mwe.pos,
                    'type': mwe.type
                }
                for mwe in self.multiword_expressions
            ],
            'analyzed_text': self.to_text_with_boundaries(),
            'metadata': {
                'language': self.language,
                'source_text': self.source_text,
            }
        }
