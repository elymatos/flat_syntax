#!/usr/bin/env python3
"""Test TrankitClient with Portuguese sentence including MWEs and contractions."""

import json
from flat_syntax import TrankitClient

def main():
    print("Testing Trankit Client with Portuguese...")
    print("-" * 80)

    client = TrankitClient(language="portuguese")

    # Test sentence from TRANKIT_API.md
    test_sentence = "Roberto marcou o gol contra no jogo"
    print(f"Parsing: '{test_sentence}'")
    print("(Roberto scored the own goal in the game)")
    print()

    try:
        result = client.parse(test_sentence, language="portuguese")

        print(f"✓ Parse successful!")
        print(f"  Language: {result.get('language')}")
        print(f"  Processing time: {result.get('processing_time'):.3f}s")
        print(f"  MWE count: {result.get('mwe_count', 0)}")
        print()

        # Display MWEs
        if result.get('mwes'):
            print("Multiword Expressions detected:")
            for mwe in result['mwes']:
                print(f"  - '{mwe['text']}' (lemma: {mwe['lemma']}, type: {mwe['type']})")
                print(f"    Tokens: {mwe['tokens']} (span: {mwe['span']})")
            print()

        # Display tokens
        if result.get('sentences'):
            sentence = result['sentences'][0]
            print(f"Tokens ({len(sentence['tokens'])} total):")
            print("-" * 90)
            print(f"{'ID':<6} {'Text':<12} {'Lemma':<12} {'UPOS':<8} {'Head':<6} {'DepRel':<12} {'MWE':<12}")
            print("-" * 90)

            for token in sentence['tokens']:
                tid = token['id']
                mwe_info = ""
                if 'mwe_lemma' in token:
                    mwe_info = f"{token['mwe_lemma']}"

                # Handle contracted forms
                if isinstance(tid, list):
                    print(f"{str(tid):<6} {token['text']:<12} {'(contracted)':<12}")
                    for expanded in token.get('expanded', []):
                        print(f"  {expanded['id']:<4} {expanded['text']:<12} "
                              f"{expanded['lemma']:<12} {expanded['upos']:<8} "
                              f"{expanded['head']:<6} {expanded['deprel']:<12}")
                else:
                    print(f"{tid:<6} {token['text']:<12} {token.get('lemma', ''):<12} "
                          f"{token['upos']:<8} {token['head']:<6} {token['deprel']:<12} {mwe_info:<12}")

        print()
        print("Full JSON response:")
        print(json.dumps(result, indent=2, ensure_ascii=False))

    except Exception as e:
        print(f"✗ Parse failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
