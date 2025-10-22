"""
Construction Element (CE) labeling for Flat Syntax.

Maps Universal Dependencies features to Flat Syntax CE labels at three tiers:
- Phrasal: Head, Mod, Adm, Adp, Lnk, Clf, Idx, Conj
- Clausal: Pred, Arg, CPP, Gen, FPM, Conj
- Sentential: Main, Adv, Rel, Comp, Dtch, Int
"""

from typing import Dict, List, Optional
from flat_syntax.models import Token, BoundaryMarkers


class PhrasalCELabeler:
    """Maps UD features to Phrasal CE labels (word-level)."""

    # Mapping from UPOS to Phrasal CE
    UPOS_MAPPING = {
        "DET": "Mod",
        "ADJ": "Mod",
        "NUM": "Mod",
        "ADP": "Adp",
        "CCONJ": "Conj",
        "SCONJ": "Lnk",
        "NOUN": "Head",
        "PROPN": "Head",
        "PRON": "Head",
        "VERB": "Head",
        "AUX": "Head",
        "ADV": "Adm",
        "PART": "Adp",
        "PUNCT": "Head",
    }

    def label(self, tokens: List[Dict]) -> List[str]:
        """
        Map each token to phrasal CE label.

        Args:
            tokens: List of UD tokens

        Returns:
            List of phrasal CE labels (one per token)
        """
        labels = []

        for i, token in enumerate(tokens):
            label = self._label_token(token, tokens, i)
            labels.append(label)

        return labels

    def _label_token(self, token: Dict, all_tokens: List[Dict],
                    token_idx: int) -> str:
        """
        Map single token to phrasal CE label.

        Priority:
        1. Check for special deprels (nmod:poss → Gen)
        2. Check for context-specific patterns
        3. Use UPOS mapping
        4. Default to Head for content words
        """
        deprel = token.get('deprel', '')
        upos = token.get('upos', '')

        # Special case: possessive/genitive
        if 'poss' in deprel or deprel == 'nmod:poss':
            return "Gen"

        # Special case: classifier (rare, language-specific)
        if deprel == 'clf':
            return "Clf"

        # Special case: index (for indexical expressions)
        if deprel == 'discourse' and upos == 'INTJ':
            return "Idx"

        # Special case: admodifier (ADV modifying ADJ/ADV)
        if upos == 'ADV' and deprel == 'advmod':
            head_token = self._get_head_token(token, all_tokens)
            if head_token and head_token.get('upos') in ['ADJ', 'ADV']:
                return "Adm"

        # Special case: subordinating conjunction
        if upos == 'SCONJ' or deprel == 'mark':
            return "Lnk"

        # Default UPOS mapping
        label = self.UPOS_MAPPING.get(upos)
        if label:
            return label

        # Default fallback
        return "Head"

    def _get_head_token(self, token: Dict, all_tokens: List[Dict]) -> Optional[Dict]:
        """Get the head token of the given token."""
        head_id = token.get('head')
        if head_id is None or head_id == 0:
            return None

        for t in all_tokens:
            if t['id'] == head_id:
                return t

        return None


class ClausualCELabeler:
    """Maps UD features to Clausal CE labels (phrase-level)."""

    # Mapping from deprel to Clausal CE
    DEPREL_MAPPING = {
        # Arguments (core and oblique)
        "nsubj": "Arg",
        "obj": "Arg",
        "iobj": "Arg",
        "csubj": "Arg",
        "obl": "Arg",  # May need disambiguation (Arg vs FPM) in Phase 2

        # Predicate
        "root": "Pred",

        # Complex predicate parts
        "aux": "CPP",
        "cop": "CPP",
        "compound:prt": "CPP",  # Phrasal verb particles

        # Possessive/genitive
        "nmod:poss": "Gen",
        "nmod": "FPM",  # Nominal modifiers with case marking

        # Coordination
        "cc": "Conj",
        "conj": "Arg",  # Coordinated element gets label of first conjunct

        # Clausal complements (will be separate clauses)
        "ccomp": "Comp",
        "xcomp": "Comp",
        "advcl": "Adv",
        "acl": "Rel",
        "acl:relcl": "Rel",
    }

    def label(self, tokens: List[Dict], boundaries: BoundaryMarkers) -> List[str]:
        """
        Map phrases to clausal CE labels.

        Args:
            tokens: List of UD tokens
            boundaries: Boundary markers (for grouping into phrases)

        Returns:
            List of clausal CE labels (one per token, but same label for phrase)
        """
        labels = []
        phrase_groups = self._group_into_phrases(tokens, boundaries)

        for phrase in phrase_groups:
            # Get the head token of the phrase
            head_token = self._find_phrase_head(phrase, tokens)

            # Map phrase to clausal CE label
            label = self._label_phrase(head_token, tokens)

            # Apply same label to all tokens in phrase
            for _ in phrase:
                labels.append(label)

        return labels

    def _group_into_phrases(self, tokens: List[Dict],
                           boundaries: BoundaryMarkers) -> List[List[int]]:
        """
        Group token indices into phrases based on boundaries.

        Returns:
            List of phrases, where each phrase is a list of token indices
        """
        phrases = []
        current_phrase = []

        for i in range(len(tokens)):
            current_phrase.append(i)

            # Check if there's a phrase boundary after this token
            if i in boundaries.phrase_boundaries or i == len(tokens) - 1:
                phrases.append(current_phrase)
                current_phrase = []

        # Add remaining tokens if any
        if current_phrase:
            phrases.append(current_phrase)

        return phrases

    def _find_phrase_head(self, phrase: List[int],
                         tokens: List[Dict]) -> Dict:
        """
        Find the syntactic head of a phrase.

        The head is typically the token whose head is outside the phrase,
        or the highest token in the dependency tree within the phrase.
        """
        phrase_tokens = [tokens[i] for i in phrase]

        # Find token whose head is outside phrase or is root
        for i in phrase:
            token = tokens[i]
            head_id = token['head']

            # Root of sentence
            if head_id == 0:
                return token

            # Head is outside phrase
            head_idx = self._get_token_index_by_id(head_id, tokens)
            if head_idx is not None and head_idx not in phrase:
                return token

        # Fallback: return first content word
        for i in phrase:
            token = tokens[i]
            if token['upos'] in ['NOUN', 'PROPN', 'VERB', 'ADJ', 'PRON']:
                return token

        # Last fallback: return first token
        return tokens[phrase[0]]

    def _label_phrase(self, head_token: Dict, all_tokens: List[Dict]) -> str:
        """
        Map phrase to clausal CE label based on its head token.
        """
        deprel = head_token.get('deprel', '')
        upos = head_token.get('upos', '')

        # Check deprel mapping first
        label = self.DEPREL_MAPPING.get(deprel)
        if label:
            return label

        # Default based on UPOS
        if upos == 'VERB':
            return "Pred"
        elif upos in ['NOUN', 'PROPN', 'PRON']:
            return "Arg"
        else:
            return "FPM"  # Default for modifiers

    def _get_token_index_by_id(self, token_id: int,
                               tokens: List[Dict]) -> Optional[int]:
        """Get the list index of a token given its ID."""
        for i, token in enumerate(tokens):
            if token['id'] == token_id:
                return i
        return None


class SententialCELabeler:
    """Maps UD features to Sentential CE labels (clause-level)."""

    # Mapping from deprel to Sentential CE
    DEPREL_MAPPING = {
        "root": "Main",
        "parataxis": "Main",  # Paratactic clauses are typically main

        # Subordinate clauses
        "acl:relcl": "Rel",
        "acl": "Rel",
        "advcl": "Adv",

        # Complement clauses
        "ccomp": "Comp",
        "xcomp": "Comp",

        # Other
        "dislocated": "Dtch",
        "vocative": "Int",
        "discourse": "Int",
    }

    def label(self, tokens: List[Dict], boundaries: BoundaryMarkers) -> List[str]:
        """
        Map clauses to sentential CE labels.

        Args:
            tokens: List of UD tokens
            boundaries: Boundary markers (for grouping into clauses)

        Returns:
            List of sentential CE labels (one per token, but same for clause)
        """
        labels = []
        clause_groups = self._group_into_clauses(tokens, boundaries)

        for clause in clause_groups:
            # Find the main predicate of the clause
            predicate = self._find_clause_predicate(clause, tokens)

            # Map clause to sentential CE label
            label = self._label_clause(predicate, tokens)

            # Apply same label to all tokens in clause
            for _ in clause:
                labels.append(label)

        return labels

    def _group_into_clauses(self, tokens: List[Dict],
                           boundaries: BoundaryMarkers) -> List[List[int]]:
        """
        Group token indices into clauses based on boundaries.

        Returns:
            List of clauses, where each clause is a list of token indices
        """
        clauses = []
        current_clause = []

        for i in range(len(tokens)):
            current_clause.append(i)

            # Check if there's a clause boundary after this token
            if i in boundaries.clause_boundaries or i == len(tokens) - 1:
                clauses.append(current_clause)
                current_clause = []

        # Add remaining tokens if any
        if current_clause:
            clauses.append(current_clause)

        return clauses

    def _find_clause_predicate(self, clause: List[int],
                               tokens: List[Dict]) -> Optional[Dict]:
        """
        Find the main predicate (verb) of a clause.
        """
        # Look for VERB or AUX in the clause
        for i in clause:
            token = tokens[i]
            if token['upos'] in ['VERB', 'AUX']:
                # Prefer actual VERBs over AUX
                if token['upos'] == 'VERB':
                    return token

        # If no VERB, return first AUX
        for i in clause:
            token = tokens[i]
            if token['upos'] == 'AUX':
                return token

        # No predicate found, return None
        return None

    def _label_clause(self, predicate: Optional[Dict],
                     all_tokens: List[Dict]) -> str:
        """
        Map clause to sentential CE label based on its predicate.
        """
        if not predicate:
            return "Main"  # Default

        deprel = predicate.get('deprel', '')

        # Check deprel mapping
        label = self.DEPREL_MAPPING.get(deprel)
        if label:
            return label

        # Special case: coordinated clauses
        if deprel == 'conj':
            # Check if coordinating with a main clause
            head_id = predicate.get('head')
            if head_id:
                head_token = self._get_token_by_id(head_id, all_tokens)
                if head_token and head_token.get('deprel') == 'root':
                    return "Main"  # Coordinated with root = main clause
                else:
                    return "Adv"  # Otherwise likely subordinate

        # Default
        return "Main"

    def _get_token_by_id(self, token_id: int,
                        tokens: List[Dict]) -> Optional[Dict]:
        """Get token by its ID."""
        for token in tokens:
            if token['id'] == token_id:
                return token
        return None


class CELabeler:
    """Main CE labeler that coordinates all three tiers."""

    def __init__(self):
        self.phrasal_labeler = PhrasalCELabeler()
        self.clausal_labeler = ClausualCELabeler()
        self.sentential_labeler = SententialCELabeler()

    def label_all_tiers(self, tokens: List[Dict],
                       boundaries: BoundaryMarkers) -> tuple:
        """
        Label all three tiers of CE labels.

        Args:
            tokens: List of UD tokens
            boundaries: Boundary markers

        Returns:
            Tuple of (phrasal_labels, clausal_labels, sentential_labels)
        """
        phrasal_labels = self.phrasal_labeler.label(tokens)
        clausal_labels = self.clausal_labeler.label(tokens, boundaries)
        sentential_labels = self.sentential_labeler.label(tokens, boundaries)

        return phrasal_labels, clausal_labels, sentential_labels
