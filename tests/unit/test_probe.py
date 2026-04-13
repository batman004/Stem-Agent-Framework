"""Tests for the TP3 Environment Probe integration."""

import json
from unittest.mock import patch, MagicMock
import pytest

from stem_agent.core.state import StemAgentState, AgentPhase
from stem_agent.core.graph import environment_probe

MOCK_DOMAIN_JSON = {
    "sub_problems": [
        {
            "name": "mock_inventory",
            "description": "Inventory stuff",
            "expert_workflow": ["count", "order"],
            "required_tool_capabilities": ["math", "search"],
            "quality_rubric": "Is accurate."
        }
    ],
    "overall_eval_criteria": "Done."
}

@pytest.fixture
def mock_state():
    return {
        "agent": StemAgentState(
            task_class="mock_restaurant",
            task_class_description="A test domain description",
            phase=AgentPhase.PROBING
        )
    }

@patch("stem_agent.core.graph.SkillStore")
@patch("stem_agent.core.graph.get_openai_client")
@patch("stem_agent.core.graph.web_search")
def test_environment_probe_cache_miss(mock_search, mock_get_client, mock_store_cls, mock_state):
    # Setup mocks
    mock_store = MagicMock()
    mock_store.get_domain.return_value = None
    mock_store_cls.return_value = mock_store
    
    mock_search.invoke.return_value = "Mock search results"
    
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps(MOCK_DOMAIN_JSON)
    mock_client.chat.completions.create.return_value = mock_response
    mock_get_client.return_value = mock_client

    # Execute
    new_state = environment_probe(mock_state)
    agent = new_state["agent"]

    # Verify
    assert agent.phase == AgentPhase.ARCHITECTING
    assert "mock_inventory" in agent.domain_model["sub_problems"][0]["name"]
    
    # Check that search was called
    assert mock_search.invoke.call_count > 0
    
    # Check that it stored the result
    mock_store.store_domain.assert_called_once_with("mock_restaurant", "A test domain description", MOCK_DOMAIN_JSON)


@patch("stem_agent.core.graph.SkillStore")
@patch("stem_agent.core.graph.get_openai_client")
@patch("stem_agent.core.graph.web_search")
def test_environment_probe_cache_hit(mock_search, mock_get_client, mock_store_cls, mock_state):
    # Setup mocks
    mock_store = MagicMock()
    mock_store.get_domain.return_value = MOCK_DOMAIN_JSON
    mock_store_cls.return_value = mock_store

    # Execute
    new_state = environment_probe(mock_state)
    agent = new_state["agent"]

    # Verify transition
    assert agent.phase == AgentPhase.ARCHITECTING
    assert agent.domain_model == MOCK_DOMAIN_JSON
    
    # Verify no API calls were made due to cache hit
    mock_search.invoke.assert_not_called()
    mock_get_client.assert_not_called()
