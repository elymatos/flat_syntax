"""
Client for communicating with Trankit parser Docker service.
"""

import time
from typing import Dict, List, Optional
import requests

from flat_syntax.config import ServiceConfig


class TrankitServiceError(Exception):
    """Exception raised when Trankit service fails."""
    pass


class TrankitClient:
    """
    Client for communicating with Trankit parser Docker service.

    The service provides Universal Dependencies parsing via REST API.
    """

    def __init__(self, service_url: str = "http://localhost:8406",
                 language: str = 'english',
                 config: Optional[ServiceConfig] = None):
        """
        Initialize Trankit client.

        Args:
            service_url: URL of Docker service (e.g., 'http://localhost:8406')
            language: Language code for parsing (e.g., 'english', 'portuguese')
            config: Optional ServiceConfig object
        """
        if config:
            self.service_url = config.url
            self.language = config.language
            self.timeout = config.timeout
            self.max_retries = config.max_retries
            self.retry_delay = config.retry_delay
        else:
            self.service_url = service_url
            self.language = language
            self.timeout = 30
            self.max_retries = 3
            self.retry_delay = 1.0

        self.session = requests.Session()

    def parse(self, text: str, language: Optional[str] = None,
              mwe_enabled: bool = True) -> Dict:
        """
        Parse text and return UD parse.

        Args:
            text: Input sentence or text
            language: Language code (uses default if not specified)
            mwe_enabled: Enable multiword expression detection

        Returns:
            Dictionary containing:
            - sentences: List of sentence objects with tokens
            - mwe_count: Number of MWEs detected
            - mwes: Array of MWE objects (if mwe_enabled=true)
            - language: Language used for parsing
            - processing_time: Time taken by parser

        Raises:
            TrankitServiceError: If parsing fails
        """
        lang = language or self.language

        payload = {
            'text': text,
            'language': lang,
            'mwe_enabled': mwe_enabled
        }

        try:
            response = self.session.post(
                f'{self.service_url}/api/v1/parse',
                json=payload,
                timeout=self.timeout
            )

            if response.status_code != 200:
                raise TrankitServiceError(
                    f"Parser returned status {response.status_code}: {response.text}"
                )

            return response.json()

        except requests.Timeout as e:
            raise TrankitServiceError(
                f"Parser request timed out after {self.timeout}s"
            ) from e
        except requests.ConnectionError as e:
            raise TrankitServiceError(
                f"Failed to connect to parser service at {self.service_url}"
            ) from e
        except requests.RequestException as e:
            raise TrankitServiceError(
                f"Parser request failed: {str(e)}"
            ) from e

    def parse_with_retry(self, text: str, language: Optional[str] = None,
                        mwe_enabled: bool = True) -> Dict:
        """
        Parse text with retry logic for service failures.

        Args:
            text: Input sentence or text
            language: Language code (uses default if not specified)
            mwe_enabled: Enable multiword expression detection

        Returns:
            Dictionary containing parse results

        Raises:
            TrankitServiceError: If parsing fails after all retries
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                return self.parse(text, language, mwe_enabled)
            except (requests.Timeout, requests.ConnectionError) as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    sleep_time = self.retry_delay * (2 ** attempt)
                    time.sleep(sleep_time)
                else:
                    raise TrankitServiceError(
                        f"Failed to reach parser service after {self.max_retries} attempts"
                    ) from e

        # Should never reach here, but just in case
        raise TrankitServiceError(
            f"Failed to parse after {self.max_retries} attempts"
        ) from last_error

    def parse_batch(self, texts: List[str], language: Optional[str] = None,
                   mwe_enabled: bool = True) -> List[Dict]:
        """
        Parse multiple texts.

        Args:
            texts: List of input sentences or texts
            language: Language code (uses default if not specified)
            mwe_enabled: Enable multiword expression detection

        Returns:
            List of parse result dictionaries
        """
        results = []
        for text in texts:
            try:
                result = self.parse_with_retry(text, language, mwe_enabled)
                results.append(result)
            except TrankitServiceError as e:
                # Log error but continue with other texts
                print(f"Warning: Failed to parse text '{text[:50]}...': {e}")
                results.append(None)

        return results

    def health_check(self) -> bool:
        """
        Check if Trankit service is available.

        Returns:
            True if service is responsive, False otherwise
        """
        try:
            response = self.session.get(
                f'{self.service_url}/api/v1/health',
                timeout=5
            )
            return response.status_code == 200
        except requests.RequestException:
            # Try root endpoint if /health doesn't exist
            try:
                response = self.session.get(
                    self.service_url,
                    timeout=5
                )
                return response.status_code in [200, 404]  # 404 is ok, means service is up
            except requests.RequestException:
                return False

    def get_token_by_id(self, parse_result: Dict, sentence_idx: int,
                       token_id: int) -> Optional[Dict]:
        """
        Get a specific token from parse result.

        Args:
            parse_result: Parse result from parse()
            sentence_idx: Sentence index (0-based)
            token_id: Token ID (1-based, as in UD)

        Returns:
            Token dictionary or None if not found
        """
        try:
            sentence = parse_result['sentences'][sentence_idx]
            for token in sentence['tokens']:
                # Handle both single IDs and contracted forms (list of IDs)
                tid = token['id']
                if isinstance(tid, int) and tid == token_id:
                    return token
                elif isinstance(tid, list) and token_id in tid:
                    return token
            return None
        except (KeyError, IndexError):
            return None


def main():
    """Test the Trankit client with a simple example."""
    import json

    print("Testing Trankit Client...")
    print("-" * 50)

    # Initialize client
    client = TrankitClient()

    # Check service health
    print(f"Checking service at {client.service_url}...")
    if client.health_check():
        print("✓ Service is available")
    else:
        print("✗ Service is not available")
        print("  Make sure the Trankit Docker service is running.")
        return

    print()

    # Test with a simple English sentence
    test_sentence = "The dog chased the cat."
    print(f"Parsing: '{test_sentence}'")
    print()

    try:
        result = client.parse(test_sentence, language="english")

        print(f"✓ Parse successful!")
        print(f"  Language: {result.get('language')}")
        print(f"  Processing time: {result.get('processing_time'):.3f}s")
        print(f"  MWE count: {result.get('mwe_count', 0)}")
        print()

        # Display tokens
        if result.get('sentences'):
            sentence = result['sentences'][0]
            print(f"Tokens ({len(sentence['tokens'])} total):")
            print("-" * 80)
            print(f"{'ID':<4} {'Text':<12} {'Lemma':<12} {'UPOS':<8} {'Head':<6} {'DepRel':<12}")
            print("-" * 80)

            for token in sentence['tokens']:
                tid = token['id']
                # Handle contracted forms
                if isinstance(tid, list):
                    print(f"{str(tid):<4} {'(contracted)':<12}")
                    for expanded in token.get('expanded', []):
                        print(f"  {expanded['id']:<2} {expanded['text']:<12} "
                              f"{expanded['lemma']:<12} {expanded['upos']:<8} "
                              f"{expanded['head']:<6} {expanded['deprel']:<12}")
                else:
                    print(f"{tid:<4} {token['text']:<12} {token.get('lemma', ''):<12} "
                          f"{token['upos']:<8} {token['head']:<6} {token['deprel']:<12}")

        print()
        print("Full JSON response:")
        print(json.dumps(result, indent=2, ensure_ascii=False))

    except TrankitServiceError as e:
        print(f"✗ Parse failed: {e}")


if __name__ == '__main__':
    main()
