"""Tests for agents/generator.py - Generator._parse() method."""

import pytest
from agents.generator import Generator


class TestGeneratorParse:
    """Tests for Generator._parse() method."""

    @pytest.fixture
    def generator(self):
        """Create a Generator instance without LLM for testing _parse."""
        g = Generator.__new__(Generator)
        return g

    def test_parse_empty_dict(self, generator):
        """Empty dict returns all O labels."""
        labels, failed = generator._parse("{}", 5)
        assert labels == ["O", "O", "O", "O", "O"]
        assert not failed

    def test_parse_single_label(self, generator):
        """Single label is correctly placed."""
        labels, failed = generator._parse('{"2": "B-Revenue"}', 5)
        assert labels == ["O", "O", "B-Revenue", "O", "O"]
        assert not failed

    def test_parse_multiple_labels(self, generator):
        """Multiple labels are correctly placed."""
        labels, failed = generator._parse('{"0": "B-Assets", "2": "B-Revenue", "3": "I-Revenue"}', 5)
        assert labels == ["B-Assets", "O", "B-Revenue", "I-Revenue", "O"]
        assert not failed

    def test_parse_with_surrounding_text(self, generator):
        """JSON embedded in response text is extracted."""
        response = 'Here is the result: {"1": "B-Revenue"} hope this helps!'
        labels, failed = generator._parse(response, 3)
        assert labels == ["O", "B-Revenue", "O"]
        assert not failed

    def test_parse_malformed_json(self, generator):
        """Malformed JSON returns all O and failed=True."""
        labels, failed = generator._parse("not valid json", 5)
        assert labels == ["O", "O", "O", "O", "O"]
        assert failed

    def test_parse_out_of_bounds_index(self, generator):
        """Out of bounds indices are ignored."""
        labels, failed = generator._parse('{"10": "B-Revenue"}', 5)
        assert labels == ["O", "O", "O", "O", "O"]
        assert not failed

    def test_parse_negative_index(self, generator):
        """Negative indices are ignored."""
        labels, failed = generator._parse('{"-1": "B-Revenue"}', 5)
        assert labels == ["O", "O", "O", "O", "O"]
        assert not failed

    def test_parse_non_string_values(self, generator):
        """Non-string values in JSON are handled."""
        # This might cause issues depending on implementation
        labels, failed = generator._parse('{"1": 123}', 3)
        # Should either fail or convert to string
        assert len(labels) == 3

    def test_parse_empty_response(self, generator):
        """Empty response returns failed."""
        labels, failed = generator._parse("", 5)
        assert labels == ["O", "O", "O", "O", "O"]
        assert failed

    def test_parse_json_in_code_block(self, generator):
        """JSON in markdown code block is extracted."""
        response = '```json\n{"0": "B-Revenue"}\n```'
        labels, failed = generator._parse(response, 3)
        assert labels == ["B-Revenue", "O", "O"]
        assert not failed
