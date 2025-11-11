import numpy as np
from config.trading_config import TOP_N_RECOMMENDATIONS

def render_recommendations(final_state):
    recommendations = final_state.get("final_recommendation", {})
    failed_stocks = final_state.get("failed_stocks", [])

    if not recommendations:
        print("No recommendations generated.")
        if failed_stocks:
            print("\nThe following stocks could not be analyzed due to data unavailability:")
            for failed in failed_stocks:
                symbol = failed.get("symbol", "Unknown")
                error = failed.get("error", "Unknown error")
                print(f"  • {symbol}: {error}")
        return

    print(f"\nTop {TOP_N_RECOMMENDATIONS} Trading Recommendations (ranked by confidence):")
    print("=" * 50)

    buy_ranking = final_state.get("buy_ranking")
    ranking_reasoning = final_state.get("ranking_reasoning")
    all_recommendations = final_state.get("all_recommendations", {})

    if buy_ranking and len(buy_ranking) > 0:
        print("\n" + "="*80)
        print("MULTI-STOCK ANALYSIS RESULTS")
        print("="*80)

        if ranking_reasoning:
            print(f"\nRanking Summary: {ranking_reasoning}")

        top_candidate = final_state.get("top_buy_candidate")
        if top_candidate:
            print(f"\nTOP BUY CANDIDATE:")
            print("-" * 40)
            for symbol, rec in top_candidate.items():
                action = rec.get("action", "UNKNOWN")
                confidence = rec.get("confidence", 0)
                llm_reasoning = rec.get("llm_reasoning")
                reasoning = llm_reasoning if llm_reasoning else rec.get("reasoning", "No reasoning provided")
                reasoning_label = "LLM Reasoning: " if llm_reasoning else "Reasoning: "
                print(f" {symbol}: {action} (Confidence: {confidence:.1%})")
                print(f"   {reasoning_label}{reasoning}")
                print()

        print("\nBUY RANKING TABLE:")
        print("-" * 70)
        print(f"{'Rank':<5} {'Symbol':<12} {'Confidence':<12} {'Composite Score':<15}")
        print("-" * 70)

        rank = 1
        for item in buy_ranking:
            symbol = item.get("symbol", "UNKNOWN")
            confidence = item.get("confidence", 0)
            composite_score = item.get("composite_score", 0)
            marker = f"{rank}."
            print(f"{marker:<5} {symbol:<12} {confidence:<12.1%} {composite_score:<15.2f}")
            rank += 1

        if all_recommendations:
            print(f"\nALL INDIVIDUAL RECOMMENDATIONS ({len(all_recommendations)} stocks):")
            print("-" * 70)
            for symbol, rec in all_recommendations.items():
                action = rec.get("action", "UNKNOWN")
                confidence = rec.get("confidence", 0)
                llm_reasoning = rec.get("llm_reasoning")
                reasoning = llm_reasoning if llm_reasoning else rec.get("reasoning", "No reasoning provided")
                reasoning_label = "LLM " if llm_reasoning else ""
                print(f"{symbol:<12} {action:<6} {confidence:<12.1%} {reasoning_label}Reasoning: {reasoning}")

    else:
        for symbol, rec in recommendations.items():
            action = rec.get("action", "UNKNOWN")
            llm_reasoning = rec.get("llm_reasoning")
            reasoning = llm_reasoning if llm_reasoning else rec.get("reasoning", "No reasoning provided")
            confidence = rec.get("confidence", 0)
            reasoning_label = "LLM Reasoning: " if llm_reasoning else "Reasoning: "
            print(f"{symbol}: {action} (Confidence: {confidence:.1%})")
            print(f"  {reasoning_label}{reasoning}")

def render_simulation(state):
    simulation_results = state.get("simulation_results")
    performance_analysis = state.get("performance_analysis")
    
    if not simulation_results or "error" in simulation_results:
        error_msg = simulation_results.get("error", "Unknown error") if simulation_results else "No results"
        print(f"\nSimulation failed: {error_msg}")
        return

    print("\nPortfolio Simulation Results:")
    print("=" * 50)

    final_value = float(simulation_results.get('final_portfolio_value', 0) or 0)
    total_return = float(simulation_results.get('total_return', 0) or 0)
    max_drawdown = float(simulation_results.get('max_drawdown', 0) or 0)

    print(f"Final Portfolio Value: ₹{final_value:,.0f}")
    print(f"Total Return: {total_return:.2%}")
    print(f"Maximum Drawdown: {max_drawdown:.2%}")

    rating = performance_analysis.get('performance_rating', 'Unknown') or 'Unknown'
    print(f"\nPerformance Rating: {rating}")

    insights = performance_analysis.get("insights", []) or []
    if insights:
        print("\nKey Insights:")
        for insight in insights[:3]:
            print(f"• {insight}")

def render_walk_forward(final_state):
    results = final_state.get("walk_forward_results", {})
    if not results or "error" in results:
        if isinstance(results, dict) and "error" in results:
            print(f"\nWalk-Forward Optimization failed: {results['error']}")
        return
    
    print("\nWalk-Forward Optimization Results:")
    print("=" * 80)
    
    target_symbols = ["RELIANCE.NS", "TATAMOTORS.NS"]
    for symbol in target_symbols:
        if symbol in results:
            wf_res = results[symbol]
            oos = wf_res.get('aggregated_oos', {})
            if 'avg_sharpe' in oos:
                avg_sharpe = oos.get('avg_sharpe', 0.0) or 0.0
                std_sharpe = oos.get('std_sharpe', 0.0) or 0.0
                avg_win_rate = oos.get('avg_win_rate', 0.0) or 0.0
                avg_drawdown = oos.get('avg_drawdown', 0.0) or 0.0
                avg_returns = oos.get('avg_returns', 0.0) or 0.0
                total_oos_trades = oos.get('total_oos_trades', 0) or 0
                target_met = oos.get('oos_win_rate_target_met', False)
                print(f"\n{symbol} ({wf_res['num_periods']} periods):")
                print(f"  OOS Avg Sharpe: {avg_sharpe:.2f} (Std: {std_sharpe:.2f})")
                print(f"  OOS Avg Win Rate: {avg_win_rate:.1%}")
                print(f"  OOS Avg Max Drawdown: {avg_drawdown:.1%}")
                print(f"  OOS Avg Return: {avg_returns:.1%}")
                print(f"  Total OOS Trades: {total_oos_trades}")
                print(f"  OOS Win Rate Target (>50%): {'Yes' if target_met else 'No'}")
                
                periods = wf_res.get('periods', [])
                if periods:
                    print("  Sample Periods:")
                    for p in periods[:2]:
                        is_sharpe = p['is_metrics'].get('sharpe', 0.0) or 0.0
                        oos_sharpe = p['oos_metrics'].get('sharpe', 0.0) or 0.0
                        print(f"    Period {p['period_id']}: IS {p['is_dates']} | OOS {p['oos_dates']} | "
                              f"IS Sharpe={is_sharpe:.2f} | OOS Sharpe={oos_sharpe:.2f}")
            else:
                print(f"{symbol}: No OOS metrics available")
        else:
            print(f"{symbol}: No results available")
    
    all_symbols = [s for s in results if s in target_symbols and 'aggregated_oos' in results[s]]
    if all_symbols:
        avg_oos_sharpe = np.mean([results[s]['aggregated_oos']['avg_sharpe'] for s in all_symbols])
        avg_oos_win = np.mean([results[s]['aggregated_oos']['avg_win_rate'] for s in all_symbols])
        print(f"\nOverall OOS (across {len(all_symbols)} symbols):")
        print(f"  Avg Sharpe: {avg_oos_sharpe:.2f}, Avg Win Rate: {avg_oos_win:.1%}")