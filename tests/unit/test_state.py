import pytest
from stem_agent.core.state import (
    AgentPhase,
    SubProblemState,
    StemAgentState,
    TaskRecord,
    GraphState,
)


class TestAgentPhase:
    def test_all_phases_exist(self):
        expected = {"probing", "architecting", "executing", "evaluating", "branching", "complete"}
        actual = {p.value for p in AgentPhase}
        assert actual == expected

    def test_phase_is_string_enum(self):
        assert AgentPhase.PROBING == "probing"
        assert isinstance(AgentPhase.PROBING, str)


class TestTaskRecord:
    def test_default_values(self):
        record = TaskRecord()
        assert record.task_id == ""
        assert record.score == 0.0
        assert record.timestamp

    def test_with_data(self):
        record = TaskRecord(
            task_id="t1",
            task={"query": "test"},
            output="result",
            score=0.85,
            iteration=3,
        )
        assert record.task_id == "t1"
        assert record.score == 0.85
        assert record.iteration == 3


class TestSubProblemState:
    def test_minimal_creation(self):
        sp = SubProblemState(name="test_sp", description="A test sub-problem")
        assert sp.name == "test_sp"
        assert sp.competence_score == 0.0
        assert sp.agent_pattern == "react"
        assert sp.is_branched is False

    def test_is_branched_property(self):
        sp = SubProblemState(name="sp1", description="test")
        assert sp.is_branched is False
        sp.specialist_id = "specialist_123"
        assert sp.is_branched is True

    def test_full_creation(self):
        sp = SubProblemState(
            name="inventory_tracking",
            description="Track restaurant inventory levels",
            expert_workflow=["count stock", "compare to pars", "generate order"],
            required_capabilities=["search", "compute"],
            quality_rubric="Inventory counts must be accurate",
            tools_acquired=["web_search", "python_repl"],
            prompt_templates={"base": "You are an inventory expert."},
            agent_pattern="plan_execute",
            competence_score=0.82,
        )
        assert sp.agent_pattern == "plan_execute"
        assert len(sp.tools_acquired) == 2
        assert sp.competence_score == 0.82

    def test_serialization_roundtrip(self):
        sp = SubProblemState(name="test", description="roundtrip test", competence_score=0.5)
        data = sp.model_dump()
        restored = SubProblemState(**data)
        assert restored.name == sp.name
        assert restored.competence_score == sp.competence_score


class TestStemAgentState:
    def test_default_state(self):
        state = StemAgentState()
        assert state.phase == AgentPhase.PROBING
        assert state.iteration == 0
        assert state.max_iterations == 20
        assert state.session_id
        assert len(state.sub_problems) == 0
        assert len(state.specialists) == 0

    def test_with_task_class(self):
        state = StemAgentState(
            task_class="restaurant_ops",
            task_class_description="Operational management for restaurants",
        )
        assert state.task_class == "restaurant_ops"
        assert state.phase == AgentPhase.PROBING

    def test_all_branched_empty(self):
        state = StemAgentState()
        assert state.all_branched is False

    def test_all_branched_partial(self):
        state = StemAgentState()
        state.sub_problems = {
            "sp1": SubProblemState(name="sp1", description="test", specialist_id="s1"),
            "sp2": SubProblemState(name="sp2", description="test"),
        }
        assert state.all_branched is False

    def test_all_branched_complete(self):
        state = StemAgentState()
        state.sub_problems = {
            "sp1": SubProblemState(name="sp1", description="test", specialist_id="s1"),
            "sp2": SubProblemState(name="sp2", description="test", specialist_id="s2"),
        }
        assert state.all_branched is True

    def test_any_ready_to_branch(self):
        state = StemAgentState(branch_threshold=0.75)
        state.sub_problems = {
            "sp1": SubProblemState(name="sp1", description="test", competence_score=0.80),
            "sp2": SubProblemState(name="sp2", description="test", competence_score=0.40),
        }
        assert state.any_ready_to_branch is True

    def test_any_ready_to_branch_none_ready(self):
        state = StemAgentState(branch_threshold=0.75)
        state.sub_problems = {
            "sp1": SubProblemState(name="sp1", description="test", competence_score=0.50),
        }
        assert state.any_ready_to_branch is False

    def test_any_ready_to_branch_already_branched(self):
        state = StemAgentState(branch_threshold=0.75)
        state.sub_problems = {
            "sp1": SubProblemState(
                name="sp1", description="test",
                competence_score=0.90, specialist_id="s1",
            ),
        }
        assert state.any_ready_to_branch is False

    def test_add_checkpoint(self):
        state = StemAgentState(max_checkpoints=3)
        for i in range(5):
            state.iteration = i
            state.add_checkpoint()
        assert len(state.checkpoints) == 3
        assert state.checkpoints[-1]["iteration"] == 4

    def test_checkpoint_excludes_nested_checkpoints(self):
        state = StemAgentState()
        state.add_checkpoint()
        assert "checkpoints" not in state.checkpoints[0]

    def test_add_log(self):
        state = StemAgentState()
        state.add_log("Test message")
        assert len(state.log) == 1
        assert "Test message" in state.log[0]

    def test_full_serialization_roundtrip(self):
        state = StemAgentState(
            task_class="test",
            task_class_description="test domain",
            phase=AgentPhase.EXECUTING,
            iteration=5,
        )
        state.sub_problems = {
            "sp1": SubProblemState(name="sp1", description="test", competence_score=0.6),
        }
        state.add_checkpoint()
        state.add_log("roundtrip test")

        data = state.model_dump()
        restored = StemAgentState(**data)

        assert restored.task_class == state.task_class
        assert restored.phase == state.phase
        assert restored.iteration == state.iteration
        assert len(restored.sub_problems) == 1
        assert restored.sub_problems["sp1"].competence_score == 0.6
        assert len(restored.checkpoints) == 1
        assert len(restored.log) == 1


class TestGraphState:
    def test_wraps_stem_agent_state(self):
        agent = StemAgentState(task_class="test", task_class_description="test")
        gs: GraphState = {"agent": agent}
        assert gs["agent"].task_class == "test"
        assert isinstance(gs["agent"], StemAgentState)
