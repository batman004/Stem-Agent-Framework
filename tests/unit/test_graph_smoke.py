import pytest
from stem_agent.core.state import AgentPhase, StemAgentState, SubProblemState, GraphState
from stem_agent.core.graph import (
    build_graph,
    environment_probe,
    architect_planner,
    execution_loop,
    competence_tracker,
    branching_mechanism,
    complete,
    route_after_competence,
    route_after_branching,
)


class TestEnvironmentProbe:
    def test_produces_domain_model(self):
        state: GraphState = {
            "agent": StemAgentState(task_class="test", task_class_description="test domain")
        }
        result = environment_probe(state)
        agent = result["agent"]
        assert agent.phase == AgentPhase.ARCHITECTING
        assert "sub_problems" in agent.domain_model
        assert len(agent.domain_model["sub_problems"]) == 2


class TestArchitectPlanner:
    def test_creates_sub_problem_states(self):
        state: GraphState = {
            "agent": StemAgentState(
                task_class="test",
                task_class_description="test",
                phase=AgentPhase.ARCHITECTING,
                domain_model={
                    "sub_problems": [
                        {
                            "name": "sp1",
                            "description": "test sp1",
                            "expert_workflow": ["step1"],
                            "required_tool_capabilities": ["search"],
                            "quality_rubric": "Must be good.",
                        }
                    ]
                },
            )
        }
        result = architect_planner(state)
        agent = result["agent"]
        assert agent.phase == AgentPhase.EXECUTING
        assert "sp1" in agent.sub_problems
        assert agent.sub_problems["sp1"].agent_pattern == "react"
        assert len(agent.sub_problems["sp1"].tools_acquired) > 0


class TestExecutionLoop:
    def test_increments_iteration(self):
        state: GraphState = {
            "agent": StemAgentState(
                task_class="test",
                task_class_description="test",
                phase=AgentPhase.EXECUTING,
            )
        }
        state["agent"].sub_problems = {
            "sp1": SubProblemState(name="sp1", description="test")
        }
        result = execution_loop(state)
        agent = result["agent"]
        assert agent.iteration == 1
        assert agent.phase == AgentPhase.EVALUATING
        assert len(agent.checkpoints) == 1
        assert agent.sub_problems["sp1"].competence_score > 0


class TestCompetenceTracker:
    def test_routes_to_branching_when_ready(self):
        state: GraphState = {
            "agent": StemAgentState(
                task_class="test", task_class_description="test", branch_threshold=0.75,
            )
        }
        state["agent"].sub_problems = {
            "sp1": SubProblemState(name="sp1", description="test", competence_score=0.80)
        }
        result = competence_tracker(state)
        assert result["agent"].phase == AgentPhase.BRANCHING

    def test_routes_to_execution_when_not_ready(self):
        state: GraphState = {
            "agent": StemAgentState(
                task_class="test", task_class_description="test", branch_threshold=0.75,
            )
        }
        state["agent"].sub_problems = {
            "sp1": SubProblemState(name="sp1", description="test", competence_score=0.40)
        }
        result = competence_tracker(state)
        assert result["agent"].phase == AgentPhase.EXECUTING


class TestBranchingMechanism:
    def test_branches_ready_sub_problems(self):
        state: GraphState = {
            "agent": StemAgentState(
                task_class="test", task_class_description="test", branch_threshold=0.75,
            )
        }
        state["agent"].sub_problems = {
            "sp1": SubProblemState(name="sp1", description="test", competence_score=0.80)
        }
        result = branching_mechanism(state)
        agent = result["agent"]
        assert agent.sub_problems["sp1"].is_branched
        assert "sp1" in agent.specialists
        assert agent.phase == AgentPhase.COMPLETE

    def test_partial_branching(self):
        state: GraphState = {
            "agent": StemAgentState(
                task_class="test", task_class_description="test", branch_threshold=0.75,
            )
        }
        state["agent"].sub_problems = {
            "sp1": SubProblemState(name="sp1", description="test", competence_score=0.80),
            "sp2": SubProblemState(name="sp2", description="test", competence_score=0.40),
        }
        result = branching_mechanism(state)
        agent = result["agent"]
        assert agent.sub_problems["sp1"].is_branched
        assert not agent.sub_problems["sp2"].is_branched
        assert agent.phase == AgentPhase.EXECUTING


class TestRouting:
    def test_route_after_competence_to_branching(self):
        state: GraphState = {"agent": StemAgentState(phase=AgentPhase.BRANCHING)}
        assert route_after_competence(state) == "branching_mechanism"

    def test_route_after_competence_to_execution(self):
        state: GraphState = {"agent": StemAgentState(phase=AgentPhase.EXECUTING)}
        assert route_after_competence(state) == "execution_loop"

    def test_route_after_branching_to_complete(self):
        state: GraphState = {"agent": StemAgentState(phase=AgentPhase.COMPLETE)}
        assert route_after_branching(state) == "complete"

    def test_route_after_branching_to_execution(self):
        state: GraphState = {"agent": StemAgentState(phase=AgentPhase.EXECUTING)}
        assert route_after_branching(state) == "execution_loop"


class TestFullGraph:
    def test_end_to_end_smoke(self):
        app = build_graph()
        initial_state: GraphState = {
            "agent": StemAgentState(
                task_class="test_domain",
                task_class_description="A test domain for graph smoke testing.",
                max_iterations=20,
            )
        }
        final_state = app.invoke(
            initial_state,
            config={"configurable": {"thread_id": "test-smoke-1"}},
        )
        agent = final_state["agent"]
        assert agent.phase == AgentPhase.COMPLETE
        assert agent.iteration > 0
        assert len(agent.sub_problems) == 2
        assert agent.all_branched
        assert len(agent.specialists) == 2
        assert len(agent.log) > 0
        assert len(agent.checkpoints) > 0

    def test_respects_max_iterations(self):
        app = build_graph()
        initial_state: GraphState = {
            "agent": StemAgentState(
                task_class="test",
                task_class_description="test",
                max_iterations=2,
                branch_threshold=0.99,
            )
        }
        final_state = app.invoke(
            initial_state,
            config={"configurable": {"thread_id": "test-max-iter"}},
        )
        agent = final_state["agent"]
        assert agent.iteration <= 2

    def test_state_has_valid_session_id(self):
        app = build_graph()
        initial_state: GraphState = {
            "agent": StemAgentState(task_class="test", task_class_description="test")
        }
        final_state = app.invoke(
            initial_state,
            config={"configurable": {"thread_id": "test-session-id"}},
        )
        assert final_state["agent"].session_id
        assert len(final_state["agent"].session_id) == 36
