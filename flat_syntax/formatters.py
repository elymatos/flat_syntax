"""
Output formatters for Flat Syntax annotations.

Supports multiple output formats:
- Table: Human-readable aligned text
- JSON: Structured data
- CSV/CLDF: Cross-Linguistic Data Format
"""

import json
import csv
from typing import List, TextIO
from flat_syntax.models import FlatSyntaxAnnotation


class TableFormatter:
    """Formats annotation as aligned table (human-readable)."""

    def format(self, annotation: FlatSyntaxAnnotation) -> str:
        """
        Format as aligned table.

        Example output:
        ```
        Analyzed Text:  The dog + chased + the cat .
        Phrasal CEs:    Mod Head  Head     Mod Head Head
        Clausal CEs:    Arg       Pred     Arg
        Sentential CEs: Main
        ```
        """
        lines = []

        # Get text with boundaries
        text_line = "Analyzed Text:  " + self._format_text_with_boundaries(annotation)
        lines.append(text_line)

        # Get CE labels
        phrasal_line = "Phrasal CEs:    " + self._format_phrasal_ces(annotation)
        lines.append(phrasal_line)

        clausal_line = "Clausal CEs:    " + self._format_clausal_ces(annotation)
        lines.append(clausal_line)

        sentential_line = "Sentential CEs: " + self._format_sentential_ces(annotation)
        lines.append(sentential_line)

        return '\n'.join(lines)

    def _format_text_with_boundaries(self, annotation: FlatSyntaxAnnotation) -> str:
        """Generate text with boundary markers."""
        result = []
        i = 0

        while i < len(annotation.tokens):
            token = annotation.tokens[i]

            # Check if this starts an MWE
            mwe_tokens = self._get_mwe_at(annotation, i)
            if mwe_tokens:
                # Join with ^
                mwe_text = "^".join([t.text for t in mwe_tokens])
                result.append(mwe_text)
                i += len(mwe_tokens)
                continue

            result.append(token.text)

            # Add boundary marker if present
            boundary = annotation.boundaries.get_boundary_at(i)
            if boundary:
                result.append(boundary)

            i += 1

        return " ".join(result)

    def _get_mwe_at(self, annotation: FlatSyntaxAnnotation,
                    position: int) -> List:
        """Get MWE tokens starting at position."""
        if position >= len(annotation.tokens):
            return []

        token_id = annotation.tokens[position].id

        for mwe in annotation.multiword_expressions:
            if mwe.token_ids and mwe.token_ids[0] == token_id:
                # Return the tokens
                tokens = []
                for tid in mwe.token_ids:
                    for t in annotation.tokens:
                        if t.id == tid:
                            tokens.append(t)
                            break
                return tokens

        return []

    def _format_phrasal_ces(self, annotation: FlatSyntaxAnnotation) -> str:
        """Format phrasal CE labels aligned with text."""
        result = []
        i = 0

        while i < len(annotation.tokens):
            token = annotation.tokens[i]

            # Check if this starts an MWE
            mwe_tokens = self._get_mwe_at(annotation, i)
            if mwe_tokens:
                # For MWE, show CE of first token under entire MWE
                ce_label = mwe_tokens[0].phrasal_ce or ""
                result.append(ce_label)

                # Add spaces for boundary marker if present
                last_idx = i + len(mwe_tokens) - 1
                if annotation.boundaries.get_boundary_at(last_idx):
                    result.append("")

                i += len(mwe_tokens)
                continue

            # Regular token
            result.append(token.phrasal_ce or "")

            # Space for boundary marker
            if annotation.boundaries.get_boundary_at(i):
                result.append("")

            i += 1

        return " ".join(result)

    def _format_clausal_ces(self, annotation: FlatSyntaxAnnotation) -> str:
        """Format clausal CE labels (shown for phrases)."""
        result = []
        last_ce = None
        i = 0

        while i < len(annotation.tokens):
            token = annotation.tokens[i]

            # Check if this starts an MWE
            mwe_tokens = self._get_mwe_at(annotation, i)
            if mwe_tokens:
                ce_label = mwe_tokens[0].clausal_ce or ""
                if ce_label != last_ce:
                    result.append(ce_label)
                    last_ce = ce_label
                else:
                    result.append("")

                # Space for boundary
                last_idx = i + len(mwe_tokens) - 1
                if annotation.boundaries.get_boundary_at(last_idx):
                    result.append("")
                    if last_idx in annotation.boundaries.phrase_boundaries:
                        last_ce = None

                i += len(mwe_tokens)
                continue

            # Regular token
            ce_label = token.clausal_ce or ""
            if ce_label != last_ce:
                result.append(ce_label)
                last_ce = ce_label
            else:
                result.append("")

            # Space for boundary
            if annotation.boundaries.get_boundary_at(i):
                result.append("")
                if i in annotation.boundaries.phrase_boundaries:
                    last_ce = None

            i += 1

        return " ".join(result)

    def _format_sentential_ces(self, annotation: FlatSyntaxAnnotation) -> str:
        """Format sentential CE labels (shown for clauses)."""
        result = []
        last_ce = None
        i = 0

        while i < len(annotation.tokens):
            token = annotation.tokens[i]

            # Check if this starts an MWE
            mwe_tokens = self._get_mwe_at(annotation, i)
            if mwe_tokens:
                ce_label = mwe_tokens[0].sentential_ce or ""
                if ce_label != last_ce:
                    result.append(ce_label)
                    last_ce = ce_label
                else:
                    result.append("")

                # Space for boundary
                last_idx = i + len(mwe_tokens) - 1
                if annotation.boundaries.get_boundary_at(last_idx):
                    result.append("")
                    if last_idx in annotation.boundaries.clause_boundaries:
                        last_ce = None

                i += len(mwe_tokens)
                continue

            # Regular token
            ce_label = token.sentential_ce or ""
            if ce_label != last_ce:
                result.append(ce_label)
                last_ce = ce_label
            else:
                result.append("")

            # Space for boundary
            if annotation.boundaries.get_boundary_at(i):
                result.append("")
                if i in annotation.boundaries.clause_boundaries:
                    last_ce = None

            i += 1

        return " ".join(result)


class JSONFormatter:
    """Formats annotation as structured JSON."""

    @staticmethod
    def format(annotation: FlatSyntaxAnnotation, indent: int = 2) -> str:
        """
        Export as JSON string.

        Args:
            annotation: FlatSyntaxAnnotation object
            indent: Indentation level for pretty printing

        Returns:
            JSON string
        """
        return json.dumps(annotation.to_dict(), indent=indent, ensure_ascii=False)

    @staticmethod
    def save(annotation: FlatSyntaxAnnotation, output_path: str, indent: int = 2):
        """
        Save annotation to JSON file.

        Args:
            annotation: FlatSyntaxAnnotation object
            output_path: Path to output file
            indent: Indentation level for pretty printing
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(annotation.to_dict(), f, indent=indent, ensure_ascii=False)


class CLDFFormatter:
    """Formats annotation as CLDF-compatible CSV."""

    @staticmethod
    def format(annotation: FlatSyntaxAnnotation) -> str:
        """
        Format as CSV string.

        Format:
        - Row 1: Analyzed_Text (with boundary markers)
        - Row 2: Phrasal_CEs
        - Row 3: Clausal_CEs
        - Row 4: Sentential_CEs
        """
        import io

        output = io.StringIO()
        writer = csv.writer(output)

        # Use TableFormatter to get aligned representations
        table_formatter = TableFormatter()

        # Get text with boundaries
        text_with_boundaries = table_formatter._format_text_with_boundaries(annotation)

        # Get CE labels
        phrasal = table_formatter._format_phrasal_ces(annotation)
        clausal = table_formatter._format_clausal_ces(annotation)
        sentential = table_formatter._format_sentential_ces(annotation)

        # Write rows
        writer.writerow(['Analyzed_Text', text_with_boundaries])
        writer.writerow(['Phrasal_CEs', phrasal])
        writer.writerow(['Clausal_CEs', clausal])
        writer.writerow(['Sentential_CEs', sentential])

        return output.getvalue()

    @staticmethod
    def save(annotation: FlatSyntaxAnnotation, output_path: str):
        """
        Save annotation to CSV file.

        Args:
            annotation: FlatSyntaxAnnotation object
            output_path: Path to output file
        """
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            table_formatter = TableFormatter()

            writer = csv.writer(f)

            # Get text with boundaries
            text_with_boundaries = table_formatter._format_text_with_boundaries(annotation)

            # Get CE labels
            phrasal = table_formatter._format_phrasal_ces(annotation)
            clausal = table_formatter._format_clausal_ces(annotation)
            sentential = table_formatter._format_sentential_ces(annotation)

            # Write rows
            writer.writerow(['Analyzed_Text', text_with_boundaries])
            writer.writerow(['Phrasal_CEs', phrasal])
            writer.writerow(['Clausal_CEs', clausal])
            writer.writerow(['Sentential_CEs', sentential])


class MultiFormatExporter:
    """Convenience class for exporting in multiple formats."""

    def __init__(self):
        self.table_formatter = TableFormatter()
        self.json_formatter = JSONFormatter()
        self.cldf_formatter = CLDFFormatter()

    def export(self, annotation: FlatSyntaxAnnotation,
              output_path: str, format: str = "auto"):
        """
        Export annotation in specified format.

        Args:
            annotation: FlatSyntaxAnnotation object
            output_path: Path to output file
            format: Output format ('json', 'csv', 'table', 'auto')
                   If 'auto', determines from file extension
        """
        if format == "auto":
            if output_path.endswith('.json'):
                format = "json"
            elif output_path.endswith('.csv'):
                format = "csv"
            else:
                format = "table"

        if format == "json":
            self.json_formatter.save(annotation, output_path)
        elif format == "csv":
            self.cldf_formatter.save(annotation, output_path)
        elif format == "table":
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(self.table_formatter.format(annotation))
        else:
            raise ValueError(f"Unknown format: {format}")

    def print_table(self, annotation: FlatSyntaxAnnotation):
        """Print annotation as formatted table to stdout."""
        print(self.table_formatter.format(annotation))
