import logging
import concurrent.futures
from typing import Dict, Any, Callable

from data.models import State

logger = logging.getLogger(__name__)

def run_analyses_in_parallel(state: State, analyses: Dict[str, Callable[[State], Dict]]) -> Dict[str, Any]:
    """
    Runs multiple analysis functions in parallel using a thread pool.

    Args:
        state: The current state dictionary.
        analyses: A dictionary where keys are the names of the analyses (and the corresponding keys in the state to update)
                  and values are the analysis functions (agents) to run.

    Returns:
        A dictionary containing the combined results of all analyses.
    """
    results = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_analysis = {executor.submit(func, state): name for name, func in analyses.items()}
        for future in concurrent.futures.as_completed(future_to_analysis):
            analysis_name = future_to_analysis[future]
            try:
                result = future.result()
                results.update(result)
                logger.info(f"Successfully completed parallel analysis: {analysis_name}")
            except Exception as e:
                logger.error(f"Parallel analysis '{analysis_name}' failed: {e}", exc_info=True)
                # Update state with error for this specific analysis
                results[analysis_name] = {"error": str(e)}
    return results