"""
Main converter from Universal Dependencies to Flat Syntax.
"""

from typing import Dict, List
from flat_syntax.models import (
    FlatSyntaxAnnotation,
    Token,
    MultiwordExpression,
    BoundaryMarkers
)
from flat_syntax.boundaries import BoundaryDetector, InterruptionDetector
from flat_syntax.ce_labelers import CELabeler


class FlatSyntaxConverter:
    """
    Converts Universal Dependencies parse to Flat Syntax annotation.
    """

    def __init__(self, language: str = 'english'):
        """
        Initialize converter.

        Args:
            language: Language code for parsing
        """
        self.language = language
        self.boundary_detector = BoundaryDetector()
        self.interruption_detector = InterruptionDetector()
        self.ce_labeler = CELabeler()

    def convert(self, ud_parse: Dict, source_text: str = "") -> FlatSyntaxAnnotation:
        """
        Main conversion pipeline.

        Args:
            ud_parse: UD parse from Trankit service
            source_text: Original input text

        Returns:
            FlatSyntaxAnnotation object
        """
        if not ud_parse.get('sentences'):
            # Empty parse
            return FlatSyntaxAnnotation(
                tokens=[],
                boundaries=BoundaryMarkers(),
                language=self.language,
                source_text=source_text
            )

        sentence = ud_parse['sentences'][0]

        # Step 1: Extract and process tokens
        tokens_data = self._process_tokens(sentence['tokens'])

        # Step 2: Detect multiword expressions
        mwes = self._extract_multiword_expressions(ud_parse)

        # Step 3: Detect boundaries
        boundaries = self.boundary_detector.detect_boundaries(ud_parse)

        # Step 4: Detect interruptions
        interruptions = self.interruption_detector.detect(ud_parse, boundaries)

        # Step 5: Assign CE labels (three tiers)
        phrasal_ces, clausal_ces, sentential_ces = self.ce_labeler.label_all_tiers(
            tokens_data, boundaries
        )

        # Step 6: Create Token objects with CE labels
        tokens = []
        for i, token_data in enumerate(tokens_data):
            token = Token.from_ud_token(token_data)
            token.phrasal_ce = phrasal_ces[i]
            token.clausal_ce = clausal_ces[i]
            token.sentential_ce = sentential_ces[i]

            # Mark if part of MWE
            for mwe in mwes:
                if token.id in mwe.token_ids:
                    token.is_mwe_part = True
                    token.mwe_position = mwe.token_ids.index(token.id)

            tokens.append(token)

        # Step 7: Create annotation
        annotation = FlatSyntaxAnnotation(
            tokens=tokens,
            boundaries=boundaries,
            multiword_expressions=mwes,
            interruptions=interruptions,
            language=self.language,
            source_text=source_text or sentence.get('text', '')
        )

        return annotation

    def _process_tokens(self, ud_tokens: List[Dict]) -> List[Dict]:
        """
        Process tokens, handling contracted forms.

        Expands contracted tokens (e.g., Portuguese "no" -> "em" + "o").

        Returns:
            List of processed tokens (with contracted forms expanded)
        """
        processed = []

        for token in ud_tokens:
            if isinstance(token['id'], list) and 'expanded' in token:
                # Contracted form - add expanded tokens
                for expanded in token['expanded']:
                    processed.append(expanded)
            else:
                # Regular token
                processed.append(token)

        return processed

    def _extract_multiword_expressions(self, ud_parse: Dict) -> List[MultiwordExpression]:
        """
        Extract multiword expressions from UD parse.

        Trankit provides MWEs in two ways:
        1. Top-level 'mwes' array (clean summary)
        2. Token-level 'mwe_*' fields (detailed info)

        We use the top-level array when available.
        """
        mwes = []

        # Use top-level MWE array if available
        if 'mwes' in ud_parse and ud_parse['mwes']:
            for mwe_data in ud_parse['mwes']:
                # Get token IDs from span [start, end) exclusive
                span = mwe_data['span']
                token_ids = list(range(span[0], span[1]))

                mwe = MultiwordExpression(
                    token_ids=token_ids,
                    text=mwe_data['text'],
                    lemma=mwe_data['lemma'],
                    pos=mwe_data['pos'],
                    type=mwe_data['type']
                )
                mwes.append(mwe)

        # Fallback: extract from token-level fields
        elif ud_parse.get('sentences'):
            sentence = ud_parse['sentences'][0]
            seen_mwes = set()

            for token in sentence['tokens']:
                if 'mwe_span' in token:
                    span_tuple = tuple(token['mwe_span'])

                    # Only process each MWE once (using span as key)
                    if span_tuple not in seen_mwes:
                        seen_mwes.add(span_tuple)

                        # Get token IDs from span
                        token_ids = list(range(span_tuple[0], span_tuple[1]))

                        mwe = MultiwordExpression(
                            token_ids=token_ids,
                            text=token.get('mwe_lemma', ''),
                            lemma=token.get('mwe_lemma', ''),
                            pos=token.get('mwe_pos', ''),
                            type=token.get('mwe_type', '')
                        )
                        mwes.append(mwe)

        return mwes


def main():
    """Test the converter with examples."""
    from flat_syntax import TrankitClient
    from flat_syntax.formatters import TableFormatter

    print("Testing Flat Syntax Converter")
    print("=" * 80)
    print()

    # Initialize components
    client = TrankitClient(language="english")
    converter = FlatSyntaxConverter(language="english")
    formatter = TableFormatter()

    # Test sentences
    test_sentences = [
        "The dog chased the cat.",
        "The two brothers might lose the game.",
        "She walked to school.",
    ]

    for sentence in test_sentences:
        print(f"Input: {sentence}")
        print("-" * 80)

        try:
            # Parse with Trankit
            ud_parse = client.parse(sentence)

            # Convert to Flat Syntax
            annotation = converter.convert(ud_parse, sentence)

            # Display formatted output
            print(formatter.format(annotation))
            print()

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            print()


if __name__ == '__main__':
    main()
