#!/usr/bin/env python3
"""
Comprehensive test of the Flat Syntax converter.
"""

from flat_syntax import TrankitClient, FlatSyntaxConverter, TableFormatter, JSONFormatter

def main():
    print("=" * 80)
    print("FLAT SYNTAX CONVERTER - COMPREHENSIVE TEST")
    print("=" * 80)
    print()

    # Initialize
    formatter = TableFormatter()

    # Test English
    print("ENGLISH TESTS")
    print("-" * 80)
    print()

    client_en = TrankitClient(language="english")
    converter_en = FlatSyntaxConverter(language="english")

    english_sentences = [
        "The dog chased the cat.",
        "The two brothers might lose the game.",
        "She walked to school.",
        "The tree fell.",
        "I ate and left.",
    ]

    for sentence in english_sentences:
        print(f"Input: {sentence}")
        try:
            ud_parse = client_en.parse(sentence)
            annotation = converter_en.convert(ud_parse, sentence)
            print(formatter.format(annotation))
        except Exception as e:
            print(f"Error: {e}")
        print()

    # Test Portuguese
    print("\n")
    print("PORTUGUESE TESTS (with MWEs)")
    print("-" * 80)
    print()

    client_pt = TrankitClient(language="portuguese")
    converter_pt = FlatSyntaxConverter(language="portuguese")

    portuguese_sentences = [
        "Roberto marcou o gol contra no jogo",
        "O menino correu para casa",
    ]

    for sentence in portuguese_sentences:
        print(f"Input: {sentence}")
        try:
            ud_parse = client_pt.parse(sentence)
            annotation = converter_pt.convert(ud_parse, sentence)
            print(formatter.format(annotation))

            # Show MWEs if any
            if annotation.multiword_expressions:
                print()
                print("Multiword Expressions:")
                for mwe in annotation.multiword_expressions:
                    print(f"  - {mwe.text} (type: {mwe.type})")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        print()

    # JSON export example
    print("\n")
    print("JSON EXPORT EXAMPLE")
    print("-" * 80)
    print()

    sentence = "The dog chased the cat."
    ud_parse = client_en.parse(sentence)
    annotation = converter_en.convert(ud_parse, sentence)

    json_formatter = JSONFormatter()
    json_output = json_formatter.format(annotation)
    print(json_output)


if __name__ == '__main__':
    main()
