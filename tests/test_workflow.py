

import pytest
import logging
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from langgraph.graph import StateGraph, START, END

from workflow import (
    build_workflow_graph,
    should_run_walk_forward_and_simulate,
    should_simulate,
    invoke_workflow,
    simulation_node,
    performance_node,
    create_initial_state
)
from output import render_recommendations, render_simulation
from data.models import State
from config.constants import DEFAULT_STOCKS
from agents import (
    data_fetcher_agent,
    technical_analysis_agent,
    fundamental_analysis_agent,
    sentiment_analysis_agent,
    risk_assessment_agent
)
from recommendation.final_recommendation import final_recommendation_agent
from simulation.simulation_runner import run_trading_simulation
from analysis import PerformanceAnalyzer

SAMPLE_STOCKS = ["AAPL"]
SAMPLE_DATE = datetime.now()
SAMPLE_DF = pd.DataFrame({
    'Open': [100.0] * 252,
    'High': [105.0] * 252,
    'Low': [95.0] * 252,
    'Close': [100.0] * 252,
    'Volume': [1000000] * 252
}, index=pd.date_range(start=SAMPLE_DATE - timedelta(days=251), periods=252, freq='B'))

SAMPLE_TECH = {"AAPL": {"RSI": "buy", "MACD": "buy", "SMA": "hold"}}
SAMPLE_FUND = {"AAPL": {"PE": 25.0, "valuations": "fairly valued"}}
SAMPLE_SENT = {"AAPL": {"compound": 0.1, "positive": 0.2}}
SAMPLE_RISK = {"AAPL": {"volatility": 0.2, "risk_ok": True}}
SAMPLE_RECOMMENDATION_BUY_INNER = {"action": "BUY", "confidence": 0.8, "reasoning": "Overall score: 2.5/4"}
SAMPLE_RECOMMENDATION_HOLD_INNER = {"action": "HOLD", "confidence": 0.5, "reasoning": "Neutral signals"}
SAMPLE_SIM_INNER = {"final_portfolio_value": 1100000.0, "total_return": 0.1}
SAMPLE_PERF_INNER = {"performance_rating": "Good", "insights": ["Strong returns"]}

logging.getLogger().setLevel(logging.WARNING)


@pytest.fixture
def initial_state() -> State:
    
    return create_initial_state()


@pytest.fixture
def sample_state() -> State:
    
    state = create_initial_state()
    state["stock_data"] = {"AAPL": SAMPLE_DF}
    return state


def test_create_initial_state():
    
    state = create_initial_state()
    expected_keys = [
        "stock_data", "technical_signals", "fundamental_analysis",
        "sentiment_scores", "risk_metrics", "macro_scores", "final_recommendation",
        "simulation_results", "performance_analysis", "failed_stocks",
        "data_valid", "validation_errors"
    ]
    assert isinstance(state, dict)
    assert set(state.keys()) == set(expected_keys)
    for key in expected_keys:
        assert state[key] == {}


class TestBuildWorkflowGraph:


    @patch('workflow.data_fetcher_agent')
    @patch('workflow.technical_analysis_agent')
    @patch('workflow.fundamental_analysis_agent')
    @patch('workflow.sentiment_analysis_agent')
    @patch('workflow.risk_assessment_agent')
    @patch('workflow.final_recommendation_agent')
    @patch('workflow.simulation_node')
    @patch('workflow.performance_node')
    def test_build_workflow_graph_structure(self, mock_perf, mock_sim, mock_final, mock_risk,
                                            mock_sent, mock_fund, mock_tech, mock_data):
        
        # Mock node functions to return dict updates
        mock_data.return_value = {"stock_data": {}}
        mock_tech.return_value = {"technical_signals": {}}
        mock_fund.return_value = {"fundamental_analysis": {}}
        mock_sent.return_value = {"sentiment_scores": {}}
        mock_risk.return_value = {"risk_metrics": {}}
        mock_final.return_value = {"final_recommendation": {}}
        mock_sim.return_value = {"simulation_results": {}}
        mock_perf.return_value = {"performance_analysis": {}}

        graph = build_workflow_graph(SAMPLE_STOCKS)

        # Compiled graph ready for execution
        assert hasattr(graph, "get_graph")

        # Check nodes via graph inspection
        expected_nodes = [
            "data_fetcher", "validation", "analyses_hub", "technical_analysis", "fundamental_analysis",
            "sentiment_analysis", "macro_analysis", "risk_assessment", "final_recommendation",
            "simulation", "performance", "walk_forward"
        ]
        graph_obj = graph.get_graph()
        assert set(graph_obj.nodes) == set(expected_nodes)

        # Check edges
        expected_edges = [
            (START, "data_fetcher"),
            ("data_fetcher", "validation"),
            ("analyses_hub", "technical_analysis"),
            ("analyses_hub", "fundamental_analysis"),
            ("analyses_hub", "sentiment_analysis"),
            ("analyses_hub", "macro_analysis"),
            ("technical_analysis", "risk_assessment"),
            ("fundamental_analysis", "risk_assessment"),
            ("sentiment_analysis", "risk_assessment"),
            ("macro_analysis", "risk_assessment"),
            ("risk_assessment", "final_recommendation"),
            ("walk_forward", "simulation"),
            ("simulation", "performance"),
            ("performance", END)
        ]
        graph_edges = graph_obj.edges
        for src, dst in expected_edges:
            assert (src, dst) in graph_edges

        # Check conditional edges
        conditional_edges = graph_obj.conditionals
        assert "final_recommendation" in conditional_edges
        cond_final = conditional_edges["final_recommendation"]
        assert cond_final.when == should_run_walk_forward_and_simulate
        assert cond_final.path_map == {"walk_forward": "walk_forward", END: END}

        assert "validation" in conditional_edges
        cond_val = conditional_edges["validation"]
        assert cond_val.when == should_proceed_to_analyses
        assert cond_val.path_map == {"analyses_hub": "analyses_hub", END: END}

        # Test with default stocks
        default_graph = build_workflow_graph()
        assert isinstance(default_graph, CompiledStateGraph)

    def test_build_workflow_graph_error(self):

        with patch('workflow.StateGraph.compile', side_effect=Exception("Compilation failed")):
            with pytest.raises(ValueError, match="Workflow graph compilation failed"):
                build_workflow_graph(SAMPLE_STOCKS)


class TestNodeFunctions:
    

    def test_simulation_node_happy_path(self, sample_state: State):

        with patch('workflow.run_trading_simulation', return_value=SAMPLE_SIM_INNER):
            result = simulation_node(sample_state)
            assert result == {"simulation_results": SAMPLE_SIM_INNER}

    def test_simulation_node_error(self, sample_state: State, caplog):

        with caplog.at_level(logging.ERROR):
            with patch('workflow.run_trading_simulation', side_effect=Exception("Sim error")):
                result = simulation_node(sample_state)
                assert result == {"simulation_results": {"error": "Sim error"}}
                assert "SimulationActor failed: Sim error" in caplog.text

    def test_performance_node_happy_path(self, sample_state: State):
        
        sample_state["simulation_results"] = SAMPLE_SIM_INNER
        with patch.object(PerformanceAnalyzer, 'analyze_strategy_performance', return_value=SAMPLE_PERF_INNER):
            result = performance_node(sample_state)
            assert result == {"performance_analysis": SAMPLE_PERF_INNER}

    def test_performance_node_sim_error(self, sample_state: State):
        
        sample_state["simulation_results"] = {"error": "Sim failed"}
        result = performance_node(sample_state)
        assert result == {"performance_analysis": {"error": "No simulation results"}}

    def test_performance_node_analysis_error(self, sample_state: State, caplog):
        
        sample_state["simulation_results"] = SAMPLE_SIM_INNER
        with caplog.at_level(logging.ERROR):
            with patch.object(PerformanceAnalyzer, 'analyze_strategy_performance', side_effect=Exception("Analysis error")):
                result = performance_node(sample_state)
                assert result == {"performance_analysis": {"error": "Analysis error"}}
                assert "Performance analysis failed: Analysis error" in caplog.text


class TestShouldSimulate:
    

    @pytest.mark.parametrize("recommendations, expected", [
        (SAMPLE_RECOMMENDATION_BUY_INNER, "simulation"),
        ({"action": "SELL"}, "simulation"),
        (SAMPLE_RECOMMENDATION_HOLD_INNER, END),
        ({}, END),
        ({"action": "HOLD", "other": "data"}, END),
    ])
    def test_should_simulate(self, recommendations: Dict[str, Any], expected: str, initial_state: State):
        
        state = initial_state.copy()
        state["final_recommendation"] = {"AAPL": recommendations}
        result = should_simulate(state)
        assert result == expected

    def test_should_simulate_invalid_rec(self, initial_state: State):
        
        state = initial_state.copy()
        state["final_recommendation"] = {"AAPL": "invalid"}
        result = should_simulate(state)
        assert result == END


class TestRunAnalysisAndSimulation:


    @patch('workflow.build_workflow_graph')
    def test_run_analysis_happy_path(self, mock_build, capsys):

        # Mocks
        mock_graph = MagicMock()
        mock_build.return_value = mock_graph
        final_state = {
            "data_valid": True,
            "final_recommendation": {"AAPL": SAMPLE_RECOMMENDATION_BUY_INNER},
            "simulation_results": SAMPLE_SIM_INNER,
            "performance_analysis": SAMPLE_PERF_INNER
        }
        mock_graph.invoke.return_value = final_state

        result = invoke_workflow(SAMPLE_STOCKS)

        # Verify calls
        mock_build.assert_called_once_with(SAMPLE_STOCKS, period="5y")
        mock_graph.invoke.assert_called_once_with(create_initial_state())

        # Verify output after rendering
        render_recommendations(result)
        render_simulation(result)
        captured = capsys.readouterr()
        assert "Top 10 Trading Recommendations" in captured.out
        assert "AAPL: BUY" in captured.out
        assert "Portfolio Simulation Results" in captured.out

        # Verify return
        assert result == final_state
        assert result["simulation_results"] == SAMPLE_SIM_INNER
        assert result["performance_analysis"] == SAMPLE_PERF_INNER

    @patch('workflow.build_workflow_graph')
    def test_run_analysis_no_simulation(self, mock_build, capsys):

        mock_graph = MagicMock()
        mock_build.return_value = mock_graph
        final_state = {
            "stock_data": {"AAPL": pd.DataFrame()},
            "final_recommendation": {"AAPL": SAMPLE_RECOMMENDATION_HOLD_INNER}
        }
        mock_graph.invoke.return_value = final_state

        result = invoke_workflow(SAMPLE_STOCKS)

        render_recommendations(result)
        captured = capsys.readouterr()
        assert "AAPL: HOLD" in captured.out
        assert result.get("simulation_results") is None
        assert result.get("performance_analysis") is None

    @patch('workflow.build_workflow_graph')
    def test_run_analysis_graph_error(self, mock_build):

        mock_build.side_effect = ValueError("Graph error")
        with pytest.raises(RuntimeError, match="Trading analysis failed"):
            invoke_workflow(SAMPLE_STOCKS)

    @patch('workflow.build_workflow_graph')
    def test_run_analysis_empty_state(self, mock_build):

        mock_graph = MagicMock()
        mock_build.return_value = mock_graph
        mock_graph.invoke.return_value = None
        result = invoke_workflow(SAMPLE_STOCKS)
        assert result is None

    @patch('workflow.build_workflow_graph')
    def test_run_analysis_sim_error(self, mock_build, capsys):

        mock_graph = MagicMock()
        mock_build.return_value = mock_graph
        final_state = {
            "stock_data": {"AAPL": pd.DataFrame()},
            "final_recommendation": {"AAPL": SAMPLE_RECOMMENDATION_BUY_INNER},
            "simulation_results": {"error": "Sim failed"}
        }
        mock_graph.invoke.return_value = final_state

        result = invoke_workflow(SAMPLE_STOCKS)

        render_recommendations(result)
        render_simulation(result)
        captured = capsys.readouterr()
        assert "Simulation failed: Sim failed" in captured.out
        assert result["simulation_results"] == {"error": "Sim failed"}


class TestPrintFunctions:


    def test_print_recommendations_happy(self, capsys):

        sample_state = {"final_recommendation": {"AAPL": SAMPLE_RECOMMENDATION_BUY_INNER}}
        render_recommendations(sample_state)
        captured = capsys.readouterr()
        assert "Top 10 Trading Recommendations (ranked by confidence):" in captured.out
        assert "AAPL: BUY (Confidence: 0.8%)" in captured.out
        assert "Reasoning: Overall score" in captured.out

    def test_print_recommendations_empty(self, capsys):

        render_recommendations({"final_recommendation": {}})
        captured = capsys.readouterr()
        assert "No recommendations generated." in captured.out

    def test_print_simulation_results_happy(self, capsys):

        render_simulation({"simulation_results": SAMPLE_SIM_INNER, "performance_analysis": SAMPLE_PERF_INNER})
        captured = capsys.readouterr()
        assert "Portfolio Simulation Results" in captured.out
        assert "Final Portfolio Value: ₹1,100,000" in captured.out
        assert "Performance Rating: Good" in captured.out
        assert "Key Insights:" in captured.out

    def test_print_simulation_results_error(self, capsys):

        render_simulation({"simulation_results": {"error": "Sim error"}})
        captured = capsys.readouterr()
        assert "Simulation failed: Sim error" in captured.out

    def test_print_simulation_results_no_results(self, capsys):

        render_simulation({})
        captured = capsys.readouterr()
        assert "Simulation failed: No results" in captured.out


@patch('yfinance.download')
def test_data_fetcher_external_error(mock_yf):
    
    mock_yf.side_effect = Exception("API error")
    initial_state = create_initial_state()
    result = data_fetcher_agent(initial_state, SAMPLE_STOCKS)
    assert "stock_data" in result
    assert len(result["stock_data"]) == len(SAMPLE_STOCKS)
    for symbol in SAMPLE_STOCKS:
        df = result["stock_data"][symbol]
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

@patch('config.config.ALPHA_VANTAGE_API_KEY', 'dummy_key')
@patch('requests.get')
def test_fundamental_external_error(mock_requests, mock_key):
    
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_requests.return_value = mock_resp
    initial_state = create_initial_state()
    initial_state["stock_data"] = {"AAPL": SAMPLE_DF}
    result = fundamental_analysis_agent(initial_state)
    assert "fundamental_analysis" in result
    assert result["fundamental_analysis"]["AAPL"] == {"error": "Fundamental analysis failed"}


