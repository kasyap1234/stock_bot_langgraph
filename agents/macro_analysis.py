

import logging
from typing import Dict
from data.models import State
from data.apis import get_macro_data

logger = logging.getLogger(__name__)


def macro_analysis_agent(state: State) -> State:
    
    logging.info("Starting macro analysis agent")

    macro_data = {}
    try:
        macro_data = get_macro_data()
    except Exception as e:
        logging.error(f"Failed to fetch macro data: {e}")
        macro_data = {}

    if not macro_data:
        state["macro_scores"] = {"error": "No macro data available"}
        logging.warning("No macro data available for analysis")
        return state

    macro_scores = {'RBI_REPO': 0.0, 'INDIA_UNRATE': 0.0, 'INDIA_GDP': 0.0}

    try:
        repo_data = macro_data.get('RBI Repo Rate', {})
        repo_rate = repo_data.get('value', 6.5)
        is_default = repo_data.get('date') == 'default'
        if is_default:
            logging.warning(f"Using default RBI Repo Rate: {repo_rate}% (API data unavailable)")
        if repo_rate > 7.0:
            repo_score = -1.0
        elif repo_rate > 6.5:
            repo_score = -0.5
        elif repo_rate > 5.5:
            repo_score = 0.0
        else:
            repo_score = 0.5
        macro_scores['RBI_REPO'] = repo_score
        logging.info(f"RBI Repo Rate score: {repo_score} (rate: {repo_rate}%) - Contribution to composite: {repo_score * 0.3:.2f}")
    except Exception as e:
        logger.error(f"Error calculating RBI Repo Rate score: {e}")
        macro_scores['RBI_REPO'] = 0.0
        logging.info("RBI Repo Rate score: 0.0 (default due to error)")

    try:
        unrate_data = macro_data.get('Unemployment Rate', {})
        unrate = unrate_data.get('value', 7.0)
        is_default = unrate_data.get('date') == 'default'
        if is_default:
            logging.warning(f"Failed to fetch Unemployment Rate from FRED: Bad Request. Series tried: LRUN64TTINQ, INDUNEMP, UNRATE. Using default: {unrate}%")
        if unrate > 8.0:
            unrate_score = -1.0
        elif unrate > 6.5:
            unrate_score = -0.5
        elif unrate > 5.0:
            unrate_score = 0.0
        else:
            unrate_score = 0.5
        macro_scores['INDIA_UNRATE'] = unrate_score
        logging.info(f"India Unemployment Rate score: {unrate_score} (rate: {unrate}%) - Contribution to composite: {unrate_score * 0.3:.2f}")
    except Exception as e:
        logger.error(f"Error calculating India Unemployment Rate score: {e}")
        macro_scores['INDIA_UNRATE'] = 0.0
        logging.info("India Unemployment Rate score: 0.0 (default due to error)")

    try:
        gdp_data = macro_data.get('GDP Growth Rate YoY', {})
        gdp_growth = gdp_data.get('value', 6.5)
        is_default = gdp_data.get('date') == 'default'
        if is_default:
            logging.warning(f"Using default GDP Growth Rate: {gdp_growth}% (API data unavailable)")
        if gdp_growth > 8.0:
            gdp_score = 1.0
        elif gdp_growth > 7.0:
            gdp_score = 0.5
        elif gdp_growth > 5.0:
            gdp_score = 0.0
        elif gdp_growth > 4.0:
            gdp_score = -0.5
        else:
            gdp_score = -1.0
        macro_scores['INDIA_GDP'] = gdp_score
        logging.info(f"India GDP Growth score: {gdp_score} (growth: {gdp_growth}%) - Contribution to composite: {gdp_score * 0.4:.2f}")
    except Exception as e:
        logger.error(f"Error calculating India GDP Growth score: {e}")
        macro_scores['INDIA_GDP'] = 0.0
        logging.info("India GDP Growth score: 0.0 (default due to error)")

    weights = {'INDIA_GDP': 0.4, 'RBI_REPO': 0.3, 'INDIA_UNRATE': 0.3}
    composite_macro = sum(macro_scores[key] * weights[key] for key in weights)
    macro_scores['composite'] = composite_macro

    state["macro_scores"] = macro_scores
    logging.info(f"Computed weighted composite Indian macro score: {composite_macro:.2f} (GDP: {macro_scores['INDIA_GDP']:.2f}, RBI: {macro_scores['RBI_REPO']:.2f}, Unemployment: {macro_scores['INDIA_UNRATE']:.2f})")

    return state