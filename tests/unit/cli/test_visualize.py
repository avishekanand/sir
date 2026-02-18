"""Unit tests for the visualize module."""

import pytest
from unittest.mock import patch, MagicMock

from ragtune.cli.visualize import (
    ComponentBox,
    PipelineFlowRenderer,
    parse_value,
    get_available_types,
    show_diff,
    edit_component,
    edit_budget,
)


class TestComponentBox:
    """Tests for ComponentBox rendering."""

    def test_render_basic(self):
        """Test basic component box rendering."""
        box = ComponentBox("retriever", "bm25")
        lines = box.render()

        assert len(lines) == 5
        assert "RETRIEVER" in lines[1]
        assert "bm25" in lines[3]

    def test_render_long_type_truncated(self):
        """Test that long type names are truncated."""
        box = ComponentBox("reranker", "very-long-type-name-here")
        lines = box.render()

        # Type should be truncated to fit
        type_line = lines[3]
        assert len(type_line) == len(lines[0])  # Same width as top border

    def test_render_with_params(self):
        """Test rendering with params (params don't show in box but are stored)."""
        params = {"model": "gpt-4", "batch_size": 10}
        box = ComponentBox("reformulator", "llm", params)
        lines = box.render()

        assert box.params == params
        assert "REFORMULATOR" in lines[1]


class TestPipelineFlowRenderer:
    """Tests for PipelineFlowRenderer."""

    @pytest.fixture
    def sample_config(self):
        return {
            "pipeline": {
                "name": "Test Pipeline",
                "components": {
                    "retriever": {"type": "bm25", "params": {}},
                    "reformulator": {"type": "identity", "params": {}},
                    "reranker": {"type": "cross-encoder", "params": {}},
                    "assembler": {"type": "greedy", "params": {}},
                    "scheduler": {"type": "graceful-degradation", "params": {}},
                    "estimator": {"type": "baseline", "params": {}},
                },
                "budget": {
                    "limits": {
                        "tokens": 5000,
                        "rerank_docs": 50,
                        "latency_ms": 2000,
                    }
                }
            }
        }

    def test_render_returns_panel(self, sample_config):
        """Test that render returns a Rich Panel."""
        from rich.panel import Panel

        renderer = PipelineFlowRenderer(sample_config)
        result = renderer.render()

        assert isinstance(result, Panel)

    def test_get_component_info(self, sample_config):
        """Test component info extraction."""
        renderer = PipelineFlowRenderer(sample_config)

        comp_type, params = renderer._get_component_info("retriever")
        assert comp_type == "bm25"
        assert params == {}

    def test_get_component_info_missing(self, sample_config):
        """Test component info for missing component."""
        renderer = PipelineFlowRenderer(sample_config)

        comp_type, params = renderer._get_component_info("nonexistent")
        assert comp_type == "noop"
        assert params == {}

    def test_pipeline_name_in_output(self, sample_config):
        """Test that pipeline name appears in panel title."""
        renderer = PipelineFlowRenderer(sample_config)
        panel = renderer.render()

        assert "Test Pipeline" in panel.title


class TestParseValue:
    """Tests for the parse_value function."""

    def test_parse_integer(self):
        assert parse_value("42") == 42
        assert parse_value("-10") == -10
        assert parse_value("0") == 0

    def test_parse_float(self):
        assert parse_value("3.14") == 3.14
        assert parse_value("-2.5") == -2.5
        assert parse_value("0.0") == 0.0

    def test_parse_boolean_true(self):
        assert parse_value("true") is True
        assert parse_value("True") is True
        assert parse_value("TRUE") is True
        assert parse_value("yes") is True
        assert parse_value("Yes") is True

    def test_parse_boolean_false(self):
        assert parse_value("false") is False
        assert parse_value("False") is False
        assert parse_value("no") is False
        assert parse_value("No") is False

    def test_parse_none(self):
        assert parse_value("none") is None
        assert parse_value("None") is None
        assert parse_value("null") is None
        assert parse_value("") is None

    def test_parse_string(self):
        assert parse_value("hello") == "hello"
        assert parse_value("some-value") == "some-value"
        assert parse_value("path/to/file") == "path/to/file"

    def test_parse_with_whitespace(self):
        assert parse_value("  42  ") == 42
        assert parse_value("  true  ") is True
        assert parse_value("  hello  ") == "hello"


class TestGetAvailableTypes:
    """Tests for get_available_types function."""

    def test_returns_list(self):
        """Test that function returns a list."""
        result = get_available_types("retriever")
        assert isinstance(result, list)

    def test_unknown_component_returns_empty(self):
        """Test that unknown component returns empty list."""
        result = get_available_types("nonexistent_component")
        assert result == []


class TestShowDiff:
    """Tests for show_diff function."""

    def test_no_changes(self, capsys):
        """Test diff with no changes."""
        config = {"pipeline": {"name": "test"}}

        # Patch console to capture output
        with patch('ragtune.cli.visualize.console') as mock_console:
            show_diff(config, config)
            # Check that "No changes" message was printed
            calls = mock_console.print.call_args_list
            assert any("No changes" in str(call) for call in calls)

    def test_with_changes(self):
        """Test diff with changes shows additions and removals."""
        original = {"pipeline": {"name": "original"}}
        modified = {"pipeline": {"name": "modified"}}

        with patch('ragtune.cli.visualize.console') as mock_console:
            show_diff(original, modified)
            # Should have been called with colored output
            assert mock_console.print.called


class TestEditComponent:
    """Tests for edit_component function (non-interactive aspects)."""

    def test_creates_missing_component(self):
        """Test that missing components are created."""
        config = {"pipeline": {"components": {}}}

        # Mock the prompts to return default values without interaction
        with patch('ragtune.cli.visualize.Confirm.ask', return_value=False):
            result = edit_component(config, "retriever")

        assert "retriever" in result["pipeline"]["components"]

    def test_preserves_existing_type(self):
        """Test that existing type is preserved when not changed."""
        config = {
            "pipeline": {
                "components": {
                    "retriever": {"type": "bm25", "params": {"k": 10}}
                }
            }
        }

        with patch('ragtune.cli.visualize.Confirm.ask', return_value=False):
            result = edit_component(config, "retriever")

        assert result["pipeline"]["components"]["retriever"]["type"] == "bm25"
        assert result["pipeline"]["components"]["retriever"]["params"] == {"k": 10}


class TestEditBudget:
    """Tests for edit_budget function (non-interactive aspects)."""

    def test_creates_budget_structure(self):
        """Test that budget structure is created if missing."""
        config = {"pipeline": {}}

        # Mock prompts to exit immediately
        with patch('ragtune.cli.visualize.Prompt.ask', return_value="d"):
            result = edit_budget(config)

        assert "budget" in result["pipeline"]
        assert "limits" in result["pipeline"]["budget"]
