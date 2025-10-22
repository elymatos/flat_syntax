"""
Tests for TrankitClient.
"""

import pytest
from flat_syntax import TrankitClient
from flat_syntax.trankit_client import TrankitServiceError


class TestTrankitClient:
    """Test suite for TrankitClient."""

    @pytest.fixture
    def client(self):
        """Create a client instance."""
        return TrankitClient()

    def test_health_check(self, client):
        """Test service health check."""
        assert client.health_check() is True

    def test_parse_simple_english(self, client):
        """Test parsing a simple English sentence."""
        result = client.parse("The dog chased the cat.", language="english")

        assert result is not None
        assert 'sentences' in result
        assert len(result['sentences']) == 1
        assert result['language'] == 'english'

        sentence = result['sentences'][0]
        assert 'tokens' in sentence
        assert len(sentence['tokens']) == 6  # "The dog chased the cat ."

        # Check first token
        first_token = sentence['tokens'][0]
        assert first_token['text'] == 'The'
        assert first_token['upos'] == 'DET'
        assert first_token['lemma'] == 'the'

    def test_parse_portuguese_with_mwe(self, client):
        """Test parsing Portuguese sentence with MWE."""
        result = client.parse(
            "Roberto marcou o gol contra no jogo",
            language="portuguese"
        )

        assert result is not None
        assert result['language'] == 'portuguese'
        assert result['mwe_count'] == 1

        # Check MWE detected
        assert len(result['mwes']) == 1
        mwe = result['mwes'][0]
        assert mwe['text'] == 'gol contra'
        assert mwe['lemma'] == 'gol contra'
        assert mwe['type'] == 'fixed'

        # Check contracted form
        sentence = result['sentences'][0]
        tokens = sentence['tokens']

        # Token with id [6,7] should be the contracted "no"
        contracted_token = None
        for token in tokens:
            if isinstance(token['id'], list) and token['id'] == [6, 7]:
                contracted_token = token
                break

        assert contracted_token is not None
        assert contracted_token['text'] == 'no'
        assert 'expanded' in contracted_token
        assert len(contracted_token['expanded']) == 2

    def test_parse_with_retry(self, client):
        """Test parsing with retry logic."""
        result = client.parse_with_retry("The tree fell.")

        assert result is not None
        assert 'sentences' in result

    def test_get_token_by_id(self, client):
        """Test getting token by ID."""
        result = client.parse("The dog ran.")

        # Get token with ID 2 (should be "dog")
        token = client.get_token_by_id(result, 0, 2)
        assert token is not None
        assert token['text'] == 'dog'
        assert token['id'] == 2

    def test_parse_batch(self, client):
        """Test batch parsing."""
        texts = [
            "The dog ran.",
            "She walked to school.",
            "The tree fell."
        ]

        results = client.parse_batch(texts)

        assert len(results) == 3
        for result in results:
            assert result is not None
            assert 'sentences' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
