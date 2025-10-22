"""
Boundary detection for Flat Syntax.

Detects phrase (+), clause (#), and sentence (.) boundaries from UD parse.
"""

from typing import Dict, List, Set, Tuple, Optional
from flat_syntax.models import BoundaryMarkers, Interruption


class BoundaryDetector:
    """Detects phrase, clause, and sentence boundaries from UD parse."""

    def detect_boundaries(self, ud_parse: Dict) -> BoundaryMarkers:
        """
        Add boundary markers (+, #, .) based on UD structure.

        Args:
            ud_parse: UD parse result from Trankit

        Returns:
            BoundaryMarkers object with all boundaries marked
        """
        boundaries = BoundaryMarkers()

        if not ud_parse.get('sentences'):
            return boundaries

        sentence = ud_parse['sentences'][0]
        tokens = sentence['tokens']

        # Filter out contracted tokens (those with list IDs)
        tokens = self._expand_contracted_tokens(tokens)

        # Detect phrase boundaries ('+')
        phrase_boundaries = self._detect_phrase_boundaries(tokens)
        boundaries.add_phrase_boundaries(phrase_boundaries)

        # Detect clause boundaries ('#')
        clause_boundaries = self._detect_clause_boundaries(tokens)
        boundaries.add_clause_boundaries(clause_boundaries)

        # Sentence boundary ('.')
        if tokens:
            last_token_idx = len(tokens) - 1
            boundaries.add_sentence_boundary(last_token_idx)

        return boundaries

    def _expand_contracted_tokens(self, tokens: List[Dict]) -> List[Dict]:
        """
        Expand contracted tokens (e.g., Portuguese "no" -> "em" + "o").

        Returns a flat list of tokens with contracted forms expanded.
        """
        expanded = []

        for token in tokens:
            if isinstance(token['id'], list) and 'expanded' in token:
                # Add expanded tokens
                expanded.extend(token['expanded'])
            else:
                # Regular token
                expanded.append(token)

        return expanded

    def _detect_phrase_boundaries(self, tokens: List[Dict]) -> List[int]:
        """
        Identify phrase boundaries using UD subtrees.

        A phrase boundary is placed after the last token of each phrase.
        Phrases are typically:
        - A content word (head) with its dependents (modifiers)
        - Includes det, amod, nummod, etc.

        Algorithm:
        1. Find phrase heads (tokens whose head is outside their subtree)
        2. For each head, find rightmost token in its subtree
        3. Mark boundary after rightmost token
        """
        boundaries = []

        # Build dependency tree structure
        children_map = self._build_children_map(tokens)

        # Find main clause arguments and predicates
        # These form phrase boundaries
        phrase_heads = set()

        for i, token in enumerate(tokens):
            deprel = token.get('deprel', '')
            upos = token.get('upos', '')

            # Heads of phrases: subjects, objects, predicates
            if deprel in ['nsubj', 'obj', 'iobj', 'obl', 'root', 'csubj']:
                phrase_heads.add(token['id'])

        # For each phrase head, find rightmost token
        for head_id in phrase_heads:
            descendants = self._get_descendants(head_id, children_map)
            descendants.append(head_id)  # Include head itself

            rightmost_idx = self._get_rightmost_token_idx(descendants, tokens)
            if rightmost_idx is not None:
                boundaries.append(rightmost_idx)

        # Remove duplicates and sort
        boundaries = sorted(set(boundaries))

        # Remove the last boundary if it's at the end (will be sentence boundary)
        if boundaries and boundaries[-1] == len(tokens) - 1:
            boundaries = boundaries[:-1]

        return boundaries

    def _detect_clause_boundaries(self, tokens: List[Dict]) -> List[int]:
        """
        Identify clause boundaries.

        A clause boundary is placed after the last token of each clause.
        Clauses are identified by finding predicates (main verbs) and their
        arguments/adjuncts.

        Algorithm:
        1. Find all predicates (VERB/AUX that are root or clausal heads)
        2. For each predicate, find its clause span (predicate + args)
        3. Mark boundary after the rightmost token in the clause
        """
        boundaries = []

        # Find all predicates
        predicates = self._find_predicates(tokens)

        # Build dependency tree
        children_map = self._build_children_map(tokens)

        for pred_idx, pred_token in predicates:
            # Get clause span: predicate + all its arguments and adjuncts
            clause_tokens = self._get_clause_span(
                pred_token['id'], tokens, children_map
            )

            if clause_tokens:
                # Find rightmost token in clause
                rightmost_idx = max(clause_tokens)
                boundaries.append(rightmost_idx)

        # Remove duplicates and sort
        boundaries = sorted(set(boundaries))

        # Remove the last boundary if it's at the end
        if boundaries and boundaries[-1] == len(tokens) - 1:
            boundaries = boundaries[:-1]

        return boundaries

    def _find_predicates(self, tokens: List[Dict]) -> List[Tuple[int, Dict]]:
        """
        Find all predicates (main verbs) in the sentence.

        Predicates are typically:
        - VERB or AUX with deprel 'root'
        - VERB or AUX with clausal deprels (ccomp, xcomp, advcl, acl)
        """
        predicates = []

        for i, token in enumerate(tokens):
            upos = token.get('upos', '')
            deprel = token.get('deprel', '')

            # Check if this is a predicate
            if upos in ['VERB', 'AUX']:
                if deprel == 'root' or deprel in [
                    'ccomp', 'xcomp', 'advcl', 'acl', 'acl:relcl'
                ]:
                    predicates.append((i, token))

        return predicates

    def _get_clause_span(self, pred_id: int, tokens: List[Dict],
                        children_map: Dict[int, List[int]]) -> List[int]:
        """
        Get all token indices that belong to the clause headed by pred_id.

        Includes:
        - The predicate itself
        - All core arguments (nsubj, obj, iobj, csubj, ccomp, xcomp)
        - Oblique arguments (obl)
        - Adjuncts that are not clauses
        """
        clause_tokens = []

        # Add predicate
        pred_idx = self._get_token_index_by_id(pred_id, tokens)
        if pred_idx is not None:
            clause_tokens.append(pred_idx)

        # Get all children of predicate
        children = children_map.get(pred_id, [])

        for child_id in children:
            child_idx = self._get_token_index_by_id(child_id, tokens)
            if child_idx is None:
                continue

            child_token = tokens[child_idx]
            deprel = child_token['deprel']

            # Include core arguments and non-clausal adjuncts
            if deprel in [
                'nsubj', 'obj', 'iobj', 'obl', 'csubj',
                'aux', 'cop', 'mark', 'advmod', 'neg'
            ]:
                # Add child and all its descendants
                descendants = self._get_descendants(child_id, children_map)
                clause_tokens.append(child_idx)
                for desc_id in descendants:
                    desc_idx = self._get_token_index_by_id(desc_id, tokens)
                    if desc_idx is not None:
                        clause_tokens.append(desc_idx)

            # Don't include subordinate clauses (they form their own clauses)
            # These will be handled by their own predicate

        return clause_tokens

    def _build_children_map(self, tokens: List[Dict]) -> Dict[int, List[int]]:
        """
        Build a map from head ID to list of dependent IDs.

        Returns:
            Dict mapping head_id -> [child_id1, child_id2, ...]
        """
        children_map = {}

        for token in tokens:
            token_id = token['id']
            head_id = token['head']

            if head_id not in children_map:
                children_map[head_id] = []

            children_map[head_id].append(token_id)

        return children_map

    def _get_descendants(self, token_id: int,
                        children_map: Dict[int, List[int]]) -> List[int]:
        """
        Get all descendants of a token (recursive).

        Returns list of token IDs.
        """
        descendants = []
        children = children_map.get(token_id, [])

        for child_id in children:
            descendants.append(child_id)
            # Recursively get descendants of child
            descendants.extend(self._get_descendants(child_id, children_map))

        return descendants

    def _get_rightmost_token_idx(self, token_ids: List[int],
                                 tokens: List[Dict]) -> Optional[int]:
        """
        Get the index of the rightmost token given a list of token IDs.
        """
        indices = []
        for token_id in token_ids:
            idx = self._get_token_index_by_id(token_id, tokens)
            if idx is not None:
                indices.append(idx)

        return max(indices) if indices else None

    def _get_token_index_by_id(self, token_id: int,
                               tokens: List[Dict]) -> Optional[int]:
        """Get the list index of a token given its ID."""
        for i, token in enumerate(tokens):
            if token['id'] == token_id:
                return i
        return None


class InterruptionDetector:
    """Detects center-embedding/interruption patterns."""

    def detect(self, ud_parse: Dict, boundaries: BoundaryMarkers) -> List[Interruption]:
        """
        Find interrupting constructions requiring { } markers.

        Interruptions occur when a subordinate clause appears in the middle
        of its parent clause (center-embedding).

        Algorithm:
        1. For each subordinate clause (relative, adverbial, etc.)
        2. Check if it interrupts its parent clause
        3. Mark with { } if it does
        """
        interruptions = []

        if not ud_parse.get('sentences'):
            return interruptions

        sentence = ud_parse['sentences'][0]
        tokens = sentence['tokens']

        # Expand contracted tokens
        tokens = self._expand_contracted_tokens(tokens)

        for i, token in enumerate(tokens):
            deprel = token.get('deprel', '')

            # Check for subordinate clauses that might interrupt
            if deprel in ['acl:relcl', 'acl', 'advcl']:
                parent_id = token['head']

                # Get span of parent clause
                parent_span = self._get_clause_span(parent_id, tokens)

                # Get span of subordinate clause
                sub_span = self._get_clause_span(token['id'], tokens)

                # Check if subordinate interrupts parent
                if self._is_interrupting(sub_span, parent_span):
                    interruptions.append(Interruption(
                        start=min(sub_span),
                        end=max(sub_span),
                        type=deprel
                    ))

        return interruptions

    def _expand_contracted_tokens(self, tokens: List[Dict]) -> List[Dict]:
        """Expand contracted tokens."""
        expanded = []
        for token in tokens:
            if isinstance(token['id'], list) and 'expanded' in token:
                expanded.extend(token['expanded'])
            else:
                expanded.append(token)
        return expanded

    def _get_clause_span(self, token_id: int, tokens: List[Dict]) -> List[int]:
        """Get all token indices in the clause headed by token_id."""
        span = []

        # Build children map
        children_map = {}
        for token in tokens:
            head_id = token['head']
            if head_id not in children_map:
                children_map[head_id] = []
            children_map[head_id].append(token['id'])

        # Get all descendants
        def get_descendants(tid):
            result = [tid]
            for child_id in children_map.get(tid, []):
                result.extend(get_descendants(child_id))
            return result

        descendants = get_descendants(token_id)

        # Convert IDs to indices
        for desc_id in descendants:
            for i, token in enumerate(tokens):
                if token['id'] == desc_id:
                    span.append(i)
                    break

        return span

    def _is_interrupting(self, sub_span: List[int],
                        parent_span: List[int]) -> bool:
        """
        Check if subordinate clause interrupts parent clause.

        An interruption occurs when the subordinate clause is completely
        contained within the parent clause (not at the edges).
        """
        if not sub_span or not parent_span:
            return False

        sub_start = min(sub_span)
        sub_end = max(sub_span)
        parent_start = min(parent_span)
        parent_end = max(parent_span)

        # Subordinate is inside parent, and not at the edges
        return (sub_start > parent_start and
                sub_end < parent_end)
