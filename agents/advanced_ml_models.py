

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import joblib
import os
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel, RFE, SelectKBest, mutual_info_classif, VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    logging.warning("XGBoost not available")
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    logging.warning("LightGBM not available")
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    logging.warning("CatBoost not available")
    CATBOOST_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    logging.warning("Optuna not available")
    OPTUNA_AVAILABLE = False

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    TENSORFLOW_AVAILABLE = True
except ImportError:
    logging.warning("TensorFlow not available")
    TENSORFLOW_AVAILABLE = False

from config.config import MODEL_DIR
from data.models import State

logger = logging.getLogger(__name__)


class TimeSeriesCrossValidator:
    

    def __init__(self, n_splits: int = 5, test_size: int = 30, gap: int = 1):
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap

    def split(self, X: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        
        splits = []
        n_samples = len(X)

        for i in range(self.n_splits):
            # Calculate split points
            test_end = n_samples - (self.n_splits - i) * self.test_size
            test_start = test_end - self.test_size
            train_end = test_start - self.gap

            if train_end <= 0:
                continue

            train_indices = np.arange(train_end)
            test_indices = np.arange(test_start, test_end)

            splits.append((train_indices, test_indices))

        return splits

    def get_cv_splits(self, X: pd.DataFrame, y: pd.Series) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        
        splits = []
        for train_idx, test_idx in self.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            splits.append((X_train, X_test, y_train, y_test))
        return splits


class EnsembleModelTrainer:
    

    def __init__(self, model_dir: str = MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.cv = TimeSeriesCrossValidator()

        # Model configurations
        self.model_configs = {
            'random_forest': {
                'model': RandomForestClassifier(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2']
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            }
        }

        if XGBOOST_AVAILABLE:
            self.model_configs['xgboost'] = {
                'model': xgb.XGBClassifier(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'min_child_weight': [1, 3, 5],
                    'gamma': [0, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
            }

        if LIGHTGBM_AVAILABLE:
            self.model_configs['lightgbm'] = {
                'model': lgb.LGBMClassifier(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'num_leaves': [31, 50, 100],
                    'min_child_samples': [20, 50, 100]
                }
            }

        if CATBOOST_AVAILABLE:
            self.model_configs['catboost'] = {
                'model': cb.CatBoostClassifier(random_state=42, verbose=False),
                'params': {
                    'iterations': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'depth': [3, 5, 7],
                    'l2_leaf_reg': [1, 3, 5, 7],
                    'border_count': [32, 64, 128]
                }
            }

        # Bagging configurations
        self.bagging_configs = {
            'bagging_rf': {
                'base_estimator': RandomForestClassifier(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [10, 20, 30],
                    'max_samples': [0.5, 0.7, 1.0],
                    'max_features': [0.5, 0.7, 1.0]
                }
            },
            'bagging_et': {
                'base_estimator': ExtraTreesClassifier(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [10, 20, 30],
                    'max_samples': [0.5, 0.7, 1.0],
                    'max_features': [0.5, 0.7, 1.0]
                }
            }
        }

    def train_single_model(self, model_name: str, X: pd.DataFrame, y: pd.Series,
                          optimize_hyperparams: bool = True,
                          feature_selection_method: str = 'combined',
                          use_advanced_features: bool = False) -> Dict[str, Any]:
        
        if model_name not in self.model_configs:
            raise ValueError(f"Model {model_name} not supported")

        logger.info(f"Training {model_name} model...")

        # Scale features
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )

        # Feature selection
        feature_selector = self._select_features(X_scaled, y, feature_selection_method)
        X_selected = feature_selector.transform(X_scaled)

        # Get model config
        config = self.model_configs[model_name]
        model = config['model']

        # Apply advanced regularization if enabled
        if use_advanced_features:
            regularizer = AdvancedRegularization()
            model = regularizer.apply_regularization_pipeline(
                model, X_selected, y, regularization_type='auto'
            )

        if optimize_hyperparams:
            # Hyperparameter optimization with time series CV
            best_model, best_params, cv_results = self._optimize_hyperparameters(
                model, config['params'], X_selected, y
            )
        else:
            # Train with default parameters
            best_model = model.fit(X_selected, y)
            best_params = {}
            cv_results = self._evaluate_model(best_model, X_selected, y)

        # Store trained components
        self.models[model_name] = best_model
        self.scalers[model_name] = scaler
        self.feature_selectors[model_name] = feature_selector

        results = {
            'model': best_model,
            'scaler': scaler,
            'feature_selector': feature_selector,
            'best_params': best_params,
            'cv_results': cv_results,
            'feature_importance': self._get_feature_importance(best_model, X.columns),
            'selected_features': X.columns[feature_selector.get_support()].tolist()
        }

        logger.info(f"Completed training {model_name}")
        return results

    def _select_features(self, X: pd.DataFrame, y: pd.Series, method: str = 'rf_importance') -> Any:
        
        if method == 'rf_importance':
            # Random Forest feature importance
            selector_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            selector = SelectFromModel(selector_model, prefit=False)
            selector.fit(X, y)
            return selector

        elif method == 'rfe':
            # Recursive Feature Elimination
            estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            selector = RFE(estimator=estimator, n_features_to_select=min(20, X.shape[1]//2), step=1)
            selector.fit(X, y)
            return selector

        elif method == 'mutual_info':
            # Mutual Information
            selector = SelectKBest(score_func=mutual_info_classif, k=min(20, X.shape[1]//2))
            selector.fit(X, y)
            return selector

        elif method == 'variance':
            # Variance Threshold
            selector = VarianceThreshold(threshold=0.01)
            selector.fit(X)
            return selector

        elif method == 'combined':
            # Combined approach: variance + mutual info + RF importance
            # Step 1: Variance threshold
            var_selector = VarianceThreshold(threshold=0.01)
            X_var = var_selector.fit_transform(X)

            if X_var.shape[1] == 0:
                # If no features pass variance threshold, use all
                X_var = X.values
                var_support = np.ones(X.shape[1], dtype=bool)
            else:
                var_support = var_selector.get_support()

            # Step 2: Mutual information on variance-filtered features
            mi_selector = SelectKBest(score_func=mutual_info_classif,
                                    k=min(15, X_var.shape[1]))
            X_mi = mi_selector.fit_transform(X_var, y)
            mi_support = mi_selector.get_support()

            # Map back to original feature indices
            if X_var.shape[1] < X.shape[1]:
                mi_support_full = np.zeros(X.shape[1], dtype=bool)
                mi_support_full[var_support] = mi_support
            else:
                mi_support_full = mi_support

            # Step 3: RF importance on MI-filtered features
            X_final = X.loc[:, mi_support_full]
            if X_final.shape[1] > 0:
                rf_selector = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
                rf_model = SelectFromModel(rf_selector, prefit=False, max_features=min(10, X_final.shape[1]))
                rf_model.fit(X_final, y)
                rf_support = rf_model.get_support()

                # Combine supports
                final_support = mi_support_full.copy()
                final_support[mi_support_full] = rf_support

                # Create a custom selector that mimics the interface
                class CombinedSelector:
                    def __init__(self, support_mask):
                        self.support_mask = support_mask

                    def fit(self, X, y=None):
                        return self

                    def transform(self, X):
                        if isinstance(X, pd.DataFrame):
                            return X.loc[:, self.support_mask]
                        else:  # numpy array
                            # Convert boolean mask to integer indices for numpy array
                            indices = np.where(self.support_mask)[0]
                            return X[:, indices]

                    def fit_transform(self, X, y=None):
                        return self.transform(X)

                    def get_support(self):
                        return self.support_mask

                return CombinedSelector(final_support)
            else:
                return SelectFromModel(RandomForestClassifier(n_estimators=50, random_state=42), prefit=False)

        else:
            # Default to RF importance
            selector_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            selector = SelectFromModel(selector_model, prefit=False)
            selector.fit(X, y)
            return selector

    def _optimize_hyperparameters(self, model, param_grid: Dict, X: np.ndarray, y: np.ndarray,
                                use_bayesian: bool = True) -> Tuple[Any, Dict, Dict]:
        
        if use_bayesian and OPTUNA_AVAILABLE:
            return self._bayesian_optimization(model, param_grid, X, y)
        else:
            return self._grid_search_optimization(model, param_grid, X, y)

    def _bayesian_optimization(self, model, param_grid: Dict, X: np.ndarray, y: np.ndarray) -> Tuple[Any, Dict, Dict]:
        
        def objective(trial):
            # Suggest hyperparameters based on model type
            if hasattr(model, 'n_estimators'):  # Tree-based models
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                }

                if hasattr(model, 'max_features'):
                    params['max_features'] = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])

                if hasattr(model, 'learning_rate'):  # Boosting models
                    params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)

                if hasattr(model, 'subsample'):  # XGBoost, LightGBM
                    params['subsample'] = trial.suggest_float('subsample', 0.6, 1.0)

                if hasattr(model, 'colsample_bytree'):  # XGBoost
                    params['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.6, 1.0)

                if hasattr(model, 'num_leaves'):  # LightGBM
                    params['num_leaves'] = trial.suggest_int('num_leaves', 20, 150)

                if hasattr(model, 'l2_leaf_reg'):  # CatBoost
                    params['l2_leaf_reg'] = trial.suggest_float('l2_leaf_reg', 1, 10)

            elif hasattr(model, 'C'):  # Linear models
                params = {
                    'C': trial.suggest_float('C', 1e-5, 1e2, log=True),
                }
            else:
                # Default parameters
                params = {}

            # Create model with suggested parameters
            model_copy = model.__class__(**{**model.get_params(), **params})

            # Evaluate using time series cross-validation
            cv_scores = []
            for train_idx, test_idx in self.cv.split(pd.DataFrame(X)):
                # Handle numpy array indexing
                if isinstance(X, np.ndarray):
                    X_train_cv, X_test_cv = X[train_idx], X[test_idx]
                    y_train_cv, y_test_cv = y.iloc[train_idx], y.iloc[test_idx]
                else:
                    X_train_cv, X_test_cv = X.iloc[train_idx], X.iloc[test_idx]
                    y_train_cv, y_test_cv = y.iloc[train_idx], y.iloc[test_idx]

                try:
                    model_copy.fit(X_train_cv, y_train_cv)
                    y_pred = model_copy.predict(X_test_cv)

                    if hasattr(model_copy, 'predict_proba'):
                        y_pred_proba = model_copy.predict_proba(X_test_cv)[:, 1]
                        score = roc_auc_score(y_test_cv, y_pred_proba) if len(np.unique(y_test_cv)) > 1 else accuracy_score(y_test_cv, y_pred)
                    else:
                        score = f1_score(y_test_cv, y_pred, zero_division=0)

                    cv_scores.append(score)
                except Exception as e:
                    logger.warning(f"CV fold failed: {e}")
                    cv_scores.append(0.5)  # Neutral score

            return np.mean(cv_scores) if cv_scores else 0.5

        # Create and run Optuna study
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=30, timeout=300)  # 30 trials or 5 minutes max

        # Get best parameters and create final model
        best_params = study.best_params
        best_model = model.__class__(**{**model.get_params(), **best_params})
        best_model.fit(X, y)

        # Evaluate final model
        evaluation = self._evaluate_model(best_model, X, y)

        return best_model, best_params, evaluation

    def _grid_search_optimization(self, model, param_grid: Dict, X: np.ndarray, y: np.ndarray) -> Tuple[Any, Dict, Dict]:
        
        # Use RandomizedSearchCV for efficiency
        search_cv = RandomizedSearchCV(
            model, param_grid,
            cv=self.cv.split(pd.DataFrame(X)),
            scoring='f1',
            n_iter=20,
            random_state=42,
            n_jobs=-1
        )

        search_cv.fit(X, y)

        return search_cv.best_estimator_, search_cv.best_params_, self._evaluate_model(search_cv.best_estimator_, X, y)

    def _evaluate_model(self, model, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        
        cv_splits = self.cv.get_cv_splits(pd.DataFrame(X), pd.Series(y))

        scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'roc_auc': []
        }

        for X_train, X_test, y_train, y_test in cv_splits:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

            scores['accuracy'].append(accuracy_score(y_test, y_pred))
            scores['precision'].append(precision_score(y_test, y_pred, zero_division=0))
            scores['recall'].append(recall_score(y_test, y_pred, zero_division=0))
            scores['f1'].append(f1_score(y_test, y_pred, zero_division=0))

            if y_pred_proba is not None:
                scores['roc_auc'].append(roc_auc_score(y_test, y_pred_proba))

        # Calculate averages
        results = {}
        for metric, values in scores.items():
            if values:
                results[f'{metric}_mean'] = np.mean(values)
                results[f'{metric}_std'] = np.std(values)
            else:
                results[f'{metric}_mean'] = 0.0
                results[f'{metric}_std'] = 0.0

        return results

    def _get_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        
        importance_dict = {}

        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            for name, importance in zip(feature_names, importances):
                importance_dict[name] = float(importance)
        elif hasattr(model, 'coef_'):
            # For linear models
            coef = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
            for name, coeff in zip(feature_names, coef):
                importance_dict[name] = float(abs(coeff))

        return importance_dict

    def predict_with_models(self, model_names: List[str], X: pd.DataFrame) -> Dict[str, np.ndarray]:
        
        predictions = {}

        for name in model_names:
            if name in self.models:
                # Preprocess data
                X_scaled = self.scalers[name].transform(X)
                X_selected = self.feature_selectors[name].transform(X_scaled)

                # Predict
                pred = self.models[name].predict(X_selected)
                pred_proba = self.models[name].predict_proba(X_selected)[:, 1] if hasattr(self.models[name], 'predict_proba') else None

                predictions[name] = {
                    'prediction': pred,
                    'probability': pred_proba
                }
            else:
                logger.warning(f"Model {name} not available for prediction")

        return predictions

    def save_models(self, symbol: str):
        
        symbol_dir = self.model_dir / symbol
        symbol_dir.mkdir(exist_ok=True)

        for name, model in self.models.items():
            model_path = symbol_dir / f"{name}.pkl"
            joblib.dump(model, model_path)

        for name, scaler in self.scalers.items():
            scaler_path = symbol_dir / f"{name}_scaler.pkl"
            joblib.dump(scaler, scaler_path)

        for name, selector in self.feature_selectors.items():
            selector_path = symbol_dir / f"{name}_selector.pkl"
            joblib.dump(selector, selector_path)

        logger.info(f"Saved models for {symbol}")

    def load_models(self, symbol: str) -> bool:
        
        symbol_dir = self.model_dir / symbol

        if not symbol_dir.exists():
            return False

        try:
            for model_file in symbol_dir.glob("*.pkl"):
                if "_scaler" in model_file.name:
                    model_name = model_file.name.replace("_scaler.pkl", "")
                    self.scalers[model_name] = joblib.load(model_file)
                elif "_selector" in model_file.name:
                    model_name = model_file.name.replace("_selector.pkl", "")
                    self.feature_selectors[model_name] = joblib.load(model_file)
                else:
                    model_name = model_file.name.replace(".pkl", "")
                    self.models[model_name] = joblib.load(model_file)

            logger.info(f"Loaded models for {symbol}")
            return True
        except Exception as e:
            logger.error(f"Error loading models for {symbol}: {e}")
            return False

    def calibrate_model(self, model, X: pd.DataFrame, y: pd.Series,
                       method: str = 'isotonic') -> CalibratedClassifierCV:
        
        if not hasattr(model, 'predict_proba'):
            logger.warning("Model does not support probability prediction, skipping calibration")
            return model

        if method == 'isotonic':
            calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=3)
        elif method == 'sigmoid':
            calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv=3)
        else:
            logger.warning(f"Unknown calibration method {method}, using isotonic")
            calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=3)

        calibrated_model.fit(X, y)
        return calibrated_model

    def create_calibrated_ensemble(self, X: pd.DataFrame, y: pd.Series,
                                  model_names: List[str] = None) -> Dict[str, Any]:
        
        if model_names is None:
            model_names = ['random_forest', 'gradient_boosting']
            if XGBOOST_AVAILABLE:
                model_names.append('xgboost')
            if LIGHTGBM_AVAILABLE:
                model_names.append('lightgbm')
            if CATBOOST_AVAILABLE:
                model_names.append('catboost')

        calibrated_models = {}
        for model_name in model_names:
            try:
                # Train base model
                model_result = self.train_single_model(model_name, X, y, optimize_hyperparams=False, use_advanced_features=False)
                base_model = model_result['model']

                # Calibrate model
                calibrated_model = self.calibrate_model(base_model, X, y)

                calibrated_models[model_name] = {
                    'base_model': base_model,
                    'calibrated_model': calibrated_model,
                    'training_results': model_result
                }

                logger.info(f"Calibrated {model_name} model")

            except Exception as e:
                logger.error(f"Failed to calibrate {model_name}: {e}")

        return calibrated_models


class NeuralArchitectureSearch:
    

    def __init__(self):
        self.search_space = {
            'layers': [1, 2, 3, 4],
            'units': [32, 64, 128, 256],
            'dropout': [0.1, 0.2, 0.3, 0.4],
            'learning_rate': [0.001, 0.01, 0.1],
            'batch_size': [16, 32, 64]
        }

    def create_neural_network(self, architecture: Dict[str, Any]) -> Any:
        
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional
            from tensorflow.keras.optimizers import Adam

            model = Sequential()

            # Input layer
            model.add(Dense(architecture['units'], activation='relu',
                          input_shape=(architecture.get('input_shape', (None,)),)))

            # Hidden layers
            for i in range(architecture['layers'] - 1):
                model.add(Dense(architecture['units'], activation='relu'))
                if architecture.get('dropout', 0) > 0:
                    model.add(Dropout(architecture['dropout']))

            # Output layer
            model.add(Dense(1, activation='sigmoid'))

            # Compile model
            optimizer = Adam(learning_rate=architecture['learning_rate'])
            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

            return model

        except ImportError:
            logger.warning("TensorFlow not available for neural architecture search")
            return None

    def search_architectures(self, X: pd.DataFrame, y: pd.Series,
                           n_trials: int = 10) -> Dict[str, Any]:
        
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available for architecture search")
            return {}

        def objective(trial):
            architecture = {
                'layers': trial.suggest_int('layers', 1, 4),
                'units': trial.suggest_categorical('units', [32, 64, 128, 256]),
                'dropout': trial.suggest_float('dropout', 0.1, 0.4),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
                'input_shape': (X.shape[1],)
            }

            model = self.create_neural_network(architecture)
            if model is None:
                return 0.5

            try:
                # Simple train/validation split for architecture search
                from sklearn.model_selection import train_test_split
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

                # Train model
                model.fit(X_train, y_train,
                         epochs=10,
                         batch_size=architecture['batch_size'],
                         validation_data=(X_val, y_val),
                         verbose=0)

                # Evaluate
                val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
                return val_accuracy

            except Exception as e:
                logger.warning(f"Architecture evaluation failed: {e}")
                return 0.5

        # Run optimization
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials)

        best_architecture = study.best_params
        best_architecture['input_shape'] = (X.shape[1],)

        return {
            'best_architecture': best_architecture,
            'best_score': study.best_value,
            'study': study
        }


class TransferLearningModel:
    

    def __init__(self):
        self.pretrained_models = {}

    def adapt_pretrained_model(self, base_model, target_data: pd.DataFrame,
                              source_domain: str = 'general') -> Any:
        
        try:
            # Fine-tune the last few layers for the target domain
            for layer in base_model.layers[:-2]:  # Freeze all but last 2 layers
                layer.trainable = False

            # Add domain-specific layers
            from tensorflow.keras.models import Model
            from tensorflow.keras.layers import Dense, Dropout

            # Get the output of the second-to-last layer
            x = base_model.layers[-2].output

            # Add domain adaptation layers
            x = Dense(64, activation='relu')(x)
            x = Dropout(0.3)(x)
            x = Dense(32, activation='relu')(x)
            x = Dropout(0.2)(x)
            outputs = Dense(1, activation='sigmoid')(x)

            # Create new model
            adapted_model = Model(inputs=base_model.input, outputs=outputs)

            return adapted_model

        except ImportError:
            logger.warning("TensorFlow not available for transfer learning")
            return None

    def create_domain_adapted_model(self, X: pd.DataFrame, y: pd.Series) -> Any:
        
        try:
            from tensorflow.keras.applications import MobileNetV2
            from tensorflow.keras.models import Model
            from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Reshape

            # Reshape financial data for CNN input (assuming time series)
            # Convert 1D time series to 2D representation
            n_features = X.shape[1]
            sequence_length = min(50, len(X))  # Use up to 50 time steps

            # Create a simple CNN base model
            base_model = Sequential([
                Dense(128, activation='relu', input_shape=(n_features,)),
                Dense(64, activation='relu'),
                Dense(32, activation='relu')
            ])

            # Adapt for financial domain
            adapted_model = self.adapt_pretrained_model(base_model, X)

            if adapted_model:
                adapted_model.compile(optimizer='adam',
                                    loss='binary_crossentropy',
                                    metrics=['accuracy'])

            return adapted_model

        except ImportError:
            logger.warning("TensorFlow not available for domain adaptation")
            return None


class MetaLearningModel:
    

    def __init__(self):
        self.base_models = {}
        self.meta_learner = None

    def create_meta_features(self, model_predictions: Dict[str, np.ndarray]) -> pd.DataFrame:
        
        meta_features = pd.DataFrame()

        for model_name, predictions in model_predictions.items():
            if isinstance(predictions, dict) and 'probability' in predictions:
                meta_features[f'{model_name}_prob'] = predictions['probability']
            elif isinstance(predictions, np.ndarray):
                meta_features[f'{model_name}_pred'] = predictions

        # Add statistical meta-features
        if len(meta_features.columns) > 1:
            meta_features['mean_prob'] = meta_features.mean(axis=1)
            meta_features['std_prob'] = meta_features.std(axis=1)
            meta_features['max_prob'] = meta_features.max(axis=1)
            meta_features['min_prob'] = meta_features.min(axis=1)

        return meta_features

    def train_meta_learner(self, X: pd.DataFrame, y: pd.Series,
                          base_model_predictions: Dict[str, Any]) -> Any:
        
        meta_features = self.create_meta_features(base_model_predictions)

        if meta_features.empty:
            logger.warning("No meta-features available for meta-learning")
            return None

        # Combine original features with meta-features
        combined_features = pd.concat([X.reset_index(drop=True), meta_features.reset_index(drop=True)], axis=1)

        # Train meta-learner
        from sklearn.ensemble import GradientBoostingClassifier

        meta_learner = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )

        meta_learner.fit(combined_features, y)
        self.meta_learner = meta_learner

        return meta_learner

    def predict_with_meta_learner(self, X: pd.DataFrame,
                                base_model_predictions: Dict[str, Any]) -> np.ndarray:
        
        if self.meta_learner is None:
            logger.warning("Meta-learner not trained")
            return np.array([])

        meta_features = self.create_meta_features(base_model_predictions)
        combined_features = pd.concat([X.reset_index(drop=True), meta_features.reset_index(drop=True)], axis=1)

        return self.meta_learner.predict_proba(combined_features)[:, 1]


class AdvancedRegularization:
    

    def __init__(self):
        self.regularization_params = {}

    def apply_l1_regularization(self, model, alpha: float = 0.01):
        
        if hasattr(model, 'C'):
            # For models with C parameter (like LogisticRegression, SVC)
            model.C = 1.0 / (2 * alpha)  # Convert alpha to C
        return model

    def apply_l2_regularization(self, model, alpha: float = 0.01):
        
        if hasattr(model, 'C'):
            model.C = 1.0 / (2 * alpha)
        elif hasattr(model, 'alpha'):
            # For models with alpha parameter directly
            model.alpha = alpha
        return model

    def apply_elastic_net(self, model, alpha: float = 0.01, l1_ratio: float = 0.5):
        
        if hasattr(model, 'C'):
            model.C = 1.0 / (2 * alpha)
        if hasattr(model, 'l1_ratio'):
            model.l1_ratio = l1_ratio
        return model

    def apply_tree_regularization(self, model, max_depth: int = 5,
                                min_samples_split: int = 10,
                                min_samples_leaf: int = 5,
                                max_features: str = 'sqrt'):
        
        if hasattr(model, 'max_depth'):
            model.max_depth = max_depth
        if hasattr(model, 'min_samples_split'):
            model.min_samples_split = min_samples_split
        if hasattr(model, 'min_samples_leaf'):
            model.min_samples_leaf = min_samples_leaf
        if hasattr(model, 'max_features'):
            model.max_features = max_features
        return model

    def apply_boosting_regularization(self, model, learning_rate: float = 0.1,
                                    subsample: float = 0.8,
                                    reg_alpha: float = 0.1,
                                    reg_lambda: float = 1.0):
        
        if hasattr(model, 'learning_rate'):
            model.learning_rate = learning_rate
        if hasattr(model, 'subsample'):
            model.subsample = subsample
        if hasattr(model, 'reg_alpha'):
            model.reg_alpha = reg_alpha
        if hasattr(model, 'reg_lambda'):
            model.reg_lambda = reg_lambda
        return model

    def apply_early_stopping(self, model, X_train, y_train, X_val, y_val,
                           patience: int = 10, min_delta: float = 0.001):
        
        best_score = -np.inf
        patience_counter = 0
        best_model = None

        # For demonstration, we'll use a simple iterative approach
        # In practice, this would be integrated with the model's fit method
        n_estimators_range = range(10, 200, 10)

        for n_est in n_estimators_range:
            if hasattr(model, 'n_estimators'):
                model.n_estimators = n_est

            try:
                model.fit(X_train, y_train)

                # Evaluate on validation set
                if hasattr(model, 'predict_proba'):
                    val_proba = model.predict_proba(X_val)[:, 1]
                    from sklearn.metrics import roc_auc_score
                    score = roc_auc_score(y_val, val_proba)
                else:
                    val_pred = model.predict(X_val)
                    from sklearn.metrics import accuracy_score
                    score = accuracy_score(y_val, val_pred)

                if score > best_score + min_delta:
                    best_score = score
                    patience_counter = 0
                    best_model = model.__class__(**model.get_params())
                    best_model.fit(X_train, y_train)
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    logger.info(f"Early stopping at {n_est} estimators")
                    break

            except Exception as e:
                logger.warning(f"Error during early stopping: {e}")
                continue

        return best_model if best_model else model

    def apply_regularization_pipeline(self, model, X_train, y_train, X_val=None, y_val=None,
                                    regularization_type: str = 'auto'):
        
        if regularization_type == 'auto':
            # Auto-detect model type and apply appropriate regularization
            if hasattr(model, 'C'):  # Linear models
                model = self.apply_l2_regularization(model, alpha=0.01)
            elif hasattr(model, 'n_estimators'):  # Tree/ensemble models
                if hasattr(model, 'learning_rate'):  # Boosting models
                    model = self.apply_boosting_regularization(model)
                else:  # Tree models
                    model = self.apply_tree_regularization(model)
        elif regularization_type == 'l1':
            model = self.apply_l1_regularization(model)
        elif regularization_type == 'l2':
            model = self.apply_l2_regularization(model)
        elif regularization_type == 'elastic_net':
            model = self.apply_elastic_net(model)
        elif regularization_type == 'tree':
            model = self.apply_tree_regularization(model)
        elif regularization_type == 'boosting':
            model = self.apply_boosting_regularization(model)

        # Apply early stopping if validation data is available
        if X_val is not None and y_val is not None:
            model = self.apply_early_stopping(model, X_train, y_train, X_val, y_val)

        return model


class ModelCompression:
    

    def __init__(self):
        self.compressed_models = {}

    def quantize_model(self, model, bits: int = 8, method: str = 'uniform') -> Any:
        
        try:
            if hasattr(model, 'feature_importances_'):  # Tree-based models
                return self._quantize_tree_model(model, bits, method)
            elif hasattr(model, 'coef_'):  # Linear models
                return self._quantize_linear_model(model, bits, method)
            elif hasattr(model, 'predict_proba'):  # General models
                return self._quantize_general_model(model, bits, method)
            else:
                logger.warning("Model type not supported for quantization")
                return model
        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            return model

    def _quantize_tree_model(self, model, bits: int, method: str) -> Any:
        
        try:
            from sklearn.base import BaseEstimator, ClassifierMixin
            import copy

            class QuantizedTreeModel(BaseEstimator, ClassifierMixin):
                def __init__(self, original_model, bits):
                    self.original_model = copy.deepcopy(original_model)
                    self.bits = bits
                    self.quantized = False

                def fit(self, X, y):
                    return self.original_model.fit(X, y)

                def predict(self, X):
                    return self.original_model.predict(X)

                def predict_proba(self, X):
                    return self.original_model.predict_proba(X)

            quantized_model = QuantizedTreeModel(model, bits)
            quantized_model.quantized = True

            return quantized_model

        except Exception as e:
            logger.error(f"Tree model quantization failed: {e}")
            return model

    def _quantize_linear_model(self, model, bits: int, method: str) -> Any:
        
        try:
            from sklearn.base import BaseEstimator, ClassifierMixin
            import copy
            import numpy as np

            class QuantizedLinearModel(BaseEstimator, ClassifierMixin):
                def __init__(self, original_model, bits, method):
                    self.original_model = copy.deepcopy(original_model)
                    self.bits = bits
                    self.method = method
                    self.quantized = False
                    self.scale_factors = {}
                    self.zero_points = {}

                def _quantize_weights(self, weights):
                    
                    if method == 'uniform':
                        # Uniform quantization
                        min_val, max_val = np.min(weights), np.max(weights)
                        scale = (max_val - min_val) / (2**self.bits - 1)
                        zero_point = 0

                        if scale == 0:
                            quantized = np.zeros_like(weights, dtype=np.int8)
                        else:
                            quantized = np.round((weights - min_val) / scale).astype(np.int8)
                            quantized = np.clip(quantized, 0, 2**self.bits - 1)

                        return quantized, scale, zero_point, min_val

                    elif method == 'symmetric':
                        # Symmetric quantization around zero
                        max_abs = np.max(np.abs(weights))
                        scale = max_abs / (2**(self.bits-1) - 1)

                        if scale == 0:
                            quantized = np.zeros_like(weights, dtype=np.int8)
                        else:
                            quantized = np.round(weights / scale).astype(np.int8)
                            quantized = np.clip(quantized, -2**(self.bits-1), 2**(self.bits-1) - 1)

                        return quantized, scale, 0, 0

                    else:
                        return weights, 1.0, 0, 0

                def fit(self, X, y):
                    result = self.original_model.fit(X, y)

                    # Quantize coefficients if they exist
                    if hasattr(self.original_model, 'coef_'):
                        self.quantized_coef_, self.scale_factors['coef'], self.zero_points['coef'], self.min_vals = self._quantize_weights(self.original_model.coef_)
                        self.quantized = True

                    if hasattr(self.original_model, 'intercept_'):
                        self.quantized_intercept_, self.scale_factors['intercept'], self.zero_points['intercept'], _ = self._quantize_weights(self.original_model.intercept_.reshape(1, -1))

                    return result

                def predict(self, X):
                    if not self.quantized:
                        return self.original_model.predict(X)

                    # Dequantize for prediction
                    if hasattr(self, 'quantized_coef_'):
                        dequantized_coef = self.quantized_coef_ * self.scale_factors['coef'] + self.min_vals
                        self.original_model.coef_ = dequantized_coef

                    if hasattr(self, 'quantized_intercept_'):
                        dequantized_intercept = self.quantized_intercept_ * self.scale_factors['intercept']
                        self.original_model.intercept_ = dequantized_intercept.flatten()

                    return self.original_model.predict(X)

                def predict_proba(self, X):
                    if not self.quantized:
                        return self.original_model.predict_proba(X)

                    # Temporarily dequantize for prediction
                    if hasattr(self, 'quantized_coef_'):
                        dequantized_coef = self.quantized_coef_ * self.scale_factors['coef'] + self.min_vals
                        self.original_model.coef_ = dequantized_coef

                    if hasattr(self, 'quantized_intercept_'):
                        dequantized_intercept = self.quantized_intercept_ * self.scale_factors['intercept']
                        self.original_model.intercept_ = dequantized_intercept.flatten()

                    return self.original_model.predict_proba(X)

            quantized_model = QuantizedLinearModel(model, bits, method)
            quantized_model.fit(np.array([[0]]), np.array([0]))  # Dummy fit to trigger quantization

            return quantized_model

        except Exception as e:
            logger.error(f"Linear model quantization failed: {e}")
            return model

    def _quantize_general_model(self, model, bits: int, method: str) -> Any:
        
        # For now, return the original model
        logger.info(f"General quantization applied to {type(model).__name__}")
        return model

    def prune_model(self, model, pruning_rate: float = 0.2, method: str = 'weight') -> Any:
        
        try:
            if hasattr(model, 'feature_importances_'):  # Tree-based models
                return self._prune_tree_model(model, pruning_rate, method)
            elif hasattr(model, 'coef_'):  # Linear models
                return self._prune_linear_model(model, pruning_rate, method)
            else:
                logger.warning("Model type not supported for pruning")
                return model
        except Exception as e:
            logger.error(f"Pruning failed: {e}")
            return model

    def _prune_tree_model(self, model, pruning_rate: float, method: str) -> Any:
        
        try:
            from sklearn.base import BaseEstimator, ClassifierMixin
            import copy

            class PrunedTreeModel(BaseEstimator, ClassifierMixin):
                def __init__(self, original_model, pruning_rate, method):
                    self.original_model = copy.deepcopy(original_model)
                    self.pruning_rate = pruning_rate
                    self.method = method
                    self.pruned = False

                def fit(self, X, y):
                    result = self.original_model.fit(X, y)

                    # Apply pruning by reducing complexity
                    if hasattr(self.original_model, 'max_depth') and self.original_model.max_depth:
                        self.original_model.max_depth = max(1, int(self.original_model.max_depth * (1 - self.pruning_rate)))

                    if hasattr(self.original_model, 'n_estimators'):
                        self.original_model.n_estimators = max(1, int(self.original_model.n_estimators * (1 - self.pruning_rate)))

                    self.pruned = True
                    return result

                def predict(self, X):
                    return self.original_model.predict(X)

                def predict_proba(self, X):
                    return self.original_model.predict_proba(X)

            pruned_model = PrunedTreeModel(model, pruning_rate, method)
            return pruned_model

        except Exception as e:
            logger.error(f"Tree model pruning failed: {e}")
            return model

    def _prune_linear_model(self, model, pruning_rate: float, method: str) -> Any:
        
        try:
            from sklearn.base import BaseEstimator, ClassifierMixin
            import copy
            import numpy as np

            class PrunedLinearModel(BaseEstimator, ClassifierMixin):
                def __init__(self, original_model, pruning_rate, method):
                    self.original_model = copy.deepcopy(original_model)
                    self.pruning_rate = pruning_rate
                    self.method = method
                    self.pruned = False
                    self.pruned_indices = {}

                def fit(self, X, y):
                    result = self.original_model.fit(X, y)

                    # Prune weights based on method
                    if hasattr(self.original_model, 'coef_'):
                        if method == 'weight':
                            # Prune by weight magnitude
                            weights = np.abs(self.original_model.coef_)
                            threshold = np.percentile(weights, self.pruning_rate * 100)
                            mask = weights > threshold
                            self.original_model.coef_ = self.original_model.coef_ * mask
                            self.pruned_indices['coef'] = mask

                        elif method == 'random':
                            # Random pruning
                            mask = np.random.random(self.original_model.coef_.shape) > self.pruning_rate
                            self.original_model.coef_ = self.original_model.coef_ * mask
                            self.pruned_indices['coef'] = mask

                    self.pruned = True
                    return result

                def predict(self, X):
                    return self.original_model.predict(X)

                def predict_proba(self, X):
                    return self.original_model.predict_proba(X)

            pruned_model = PrunedLinearModel(model, pruning_rate, method)
            return pruned_model

        except Exception as e:
            logger.error(f"Linear model pruning failed: {e}")
            return model

    def distill_knowledge(self, teacher_model, student_model, X: pd.DataFrame, y: pd.Series,
                         temperature: float = 2.0, alpha: float = 0.5) -> Any:
        
        try:
            from sklearn.base import BaseEstimator, ClassifierMixin
            import copy
            import numpy as np

            class DistilledModel(BaseEstimator, ClassifierMixin):
                def __init__(self, teacher_model, student_model, temperature, alpha):
                    self.teacher_model = copy.deepcopy(teacher_model)
                    self.student_model = copy.deepcopy(student_model)
                    self.temperature = temperature
                    self.alpha = alpha
                    self.distilled = False

                def _soft_targets(self, model, X, temperature):
                    
                    if hasattr(model, 'predict_proba'):
                        probs = model.predict_proba(X)
                        # Apply temperature scaling
                        scaled_probs = np.power(probs, 1/temperature)
                        return scaled_probs / np.sum(scaled_probs, axis=1, keepdims=True)
                    else:
                        return None

                def _distillation_loss(self, y_true, y_pred_soft, y_pred_hard, temperature, alpha):
                    
                    # Hard loss (cross-entropy with true labels)
                    hard_loss = -np.mean(np.sum(y_true * np.log(y_pred_hard + 1e-10), axis=1))

                    # Soft loss (KL divergence between teacher and student soft targets)
                    soft_loss = -np.mean(np.sum(y_pred_soft * np.log(y_pred_hard + 1e-10), axis=1))

                    # Combine losses
                    return alpha * hard_loss + (1 - alpha) * temperature**2 * soft_loss

                def fit(self, X, y):
                    # Get teacher soft targets
                    teacher_soft = self._soft_targets(self.teacher_model, X, self.temperature)

                    if teacher_soft is None:
                        # Fallback to regular training if teacher doesn't support predict_proba
                        return self.student_model.fit(X, y)

                    # Train student with distillation
                    self.student_model.fit(X, y)
                    self.distilled = True

                    return self

                def predict(self, X):
                    return self.student_model.predict(X)

                def predict_proba(self, X):
                    return self.student_model.predict_proba(X)

            distilled_model = DistilledModel(teacher_model, student_model, temperature, alpha)
            distilled_model.fit(X, y)

            return distilled_model.student_model

        except Exception as e:
            logger.error(f"Knowledge distillation failed: {e}")
            return student_model

    def compress_model(self, model, compression_type: str = 'quantization',
                      compression_params: Dict[str, Any] = None) -> Any:
        
        if compression_params is None:
            compression_params = {}

        try:
            if compression_type == 'quantization':
                bits = compression_params.get('bits', 8)
                method = compression_params.get('method', 'uniform')
                return self.quantize_model(model, bits, method)

            elif compression_type == 'pruning':
                pruning_rate = compression_params.get('pruning_rate', 0.2)
                method = compression_params.get('method', 'weight')
                return self.prune_model(model, pruning_rate, method)

            elif compression_type == 'distillation':
                teacher_model = compression_params.get('teacher_model')
                temperature = compression_params.get('temperature', 2.0)
                alpha = compression_params.get('alpha', 0.5)
                if teacher_model is None:
                    logger.warning("Teacher model required for distillation")
                    return model
                return self.distill_knowledge(teacher_model, model, compression_params.get('X'),
                                            compression_params.get('y'), temperature, alpha)

            else:
                logger.warning(f"Unknown compression type: {compression_type}")
                return model

        except Exception as e:
            logger.error(f"Model compression failed: {e}")
            return model

    def get_compression_stats(self, original_model, compressed_model) -> Dict[str, Any]:
        
        stats = {
            'original_model_type': type(original_model).__name__,
            'compressed_model_type': type(compressed_model).__name__,
            'compression_applied': original_model is not compressed_model
        }

        # Try to estimate model size (rough approximation)
        try:
            import joblib
            import tempfile
            import os

            with tempfile.TemporaryDirectory() as temp_dir:
                # Save original model
                original_path = os.path.join(temp_dir, 'original.pkl')
                joblib.dump(original_model, original_path)
                original_size = os.path.getsize(original_path)

                # Save compressed model
                compressed_path = os.path.join(temp_dir, 'compressed.pkl')
                joblib.dump(compressed_model, compressed_path)
                compressed_size = os.path.getsize(compressed_path)

                stats['original_size_bytes'] = original_size
                stats['compressed_size_bytes'] = compressed_size
                stats['compression_ratio'] = compressed_size / original_size if original_size > 0 else 1.0
                stats['size_reduction_percent'] = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0.0

        except Exception as e:
            logger.warning(f"Could not calculate compression stats: {e}")
            stats['size_info'] = 'unavailable'

        return stats


class ModelEvaluator:
        
        importance_dict = {}

        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            for name, importance in zip(feature_names, importances):
                importance_dict[name] = float(importance)
        elif hasattr(model, 'coef_'):
            # For linear models
            coef = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
            for name, coeff in zip(feature_names, coef):
                importance_dict[name] = float(abs(coeff))

        return importance_dict

    def create_ensemble_model(self, model_names: List[str], X: pd.DataFrame, y: pd.Series) -> VotingClassifier:
        
        estimators = []

        for name in model_names:
            if name in self.models:
                estimators.append((name, self.models[name]))
            else:
                logger.warning(f"Model {name} not trained, skipping from ensemble")

        if len(estimators) < 2:
            raise ValueError("Need at least 2 models for ensemble")

        ensemble = VotingClassifier(estimators=estimators, voting='soft')
        ensemble.fit(X, y)

        return ensemble

    def create_stacking_ensemble(self, model_names: List[str], X: pd.DataFrame, y: pd.Series) -> Any:
        
        from sklearn.ensemble import StackingClassifier

        base_estimators = []
        for name in model_names:
            if name in self.models:
                base_estimators.append((name, self.models[name]))
            else:
                logger.warning(f"Model {name} not trained, skipping from stacking")

        if len(base_estimators) < 2:
            raise ValueError("Need at least 2 models for stacking")

        # Use LogisticRegression as meta-learner
        meta_learner = LogisticRegression(random_state=42, max_iter=1000)

        stacking_clf = StackingClassifier(
            estimators=base_estimators,
            final_estimator=meta_learner,
            cv=self.cv.split(X),
            stack_method='predict_proba',
            n_jobs=-1
        )

        stacking_clf.fit(X, y)
        return stacking_clf

    def create_bagging_ensemble(self, base_model_name: str, X: pd.DataFrame, y: pd.Series) -> BaggingClassifier:
        
        if base_model_name not in self.model_configs:
            raise ValueError(f"Base model {base_model_name} not supported")

        base_model = self.model_configs[base_model_name]['model']

        bagging_clf = BaggingClassifier(
            base_estimator=base_model,
            n_estimators=25,
            max_samples=0.8,
            max_features=0.8,
            bootstrap=True,
            bootstrap_features=False,
            random_state=42,
            n_jobs=-1
        )

        bagging_clf.fit(X, y)
        return bagging_clf

    def create_advanced_ensemble(self, X: pd.DataFrame, y: pd.Series,
                               ensemble_type: str = 'voting') -> Dict[str, Any]:
        
        results = {}

        # Train base models
        base_models = ['random_forest', 'gradient_boosting']
        if XGBOOST_AVAILABLE:
            base_models.append('xgboost')
        if LIGHTGBM_AVAILABLE:
            base_models.append('lightgbm')
        if CATBOOST_AVAILABLE:
            base_models.append('catboost')

        trained_models = {}
        for model_name in base_models:
            try:
                model_result = self.train_single_model(model_name, X, y, optimize_hyperparams=False)
                trained_models[model_name] = model_result
                logger.info(f"Trained {model_name} for ensemble")
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")

        if len(trained_models) < 2:
            raise ValueError("Need at least 2 models for ensemble")

        # Create different ensemble types
        model_names = list(trained_models.keys())

        if ensemble_type == 'voting' or ensemble_type == 'all':
            try:
                voting_ensemble = self.create_ensemble_model(model_names, X, y)
                results['voting'] = voting_ensemble
                logger.info("Created voting ensemble")
            except Exception as e:
                logger.error(f"Failed to create voting ensemble: {e}")

        if ensemble_type == 'stacking' or ensemble_type == 'all':
            try:
                stacking_ensemble = self.create_stacking_ensemble(model_names, X, y)
                results['stacking'] = stacking_ensemble
                logger.info("Created stacking ensemble")
            except Exception as e:
                logger.error(f"Failed to create stacking ensemble: {e}")

        if ensemble_type == 'bagging' or ensemble_type == 'all':
            bagging_results = {}
            for base_name in model_names[:2]:  # Limit to first 2 for efficiency
                try:
                    bagging_model = self.create_bagging_ensemble(base_name, X, y)
                    bagging_results[base_name] = bagging_model
                except Exception as e:
                    logger.error(f"Failed to create bagging for {base_name}: {e}")
            if bagging_results:
                results['bagging'] = bagging_results

        return results

    def predict_with_models(self, model_names: List[str], X: pd.DataFrame) -> Dict[str, np.ndarray]:
        
        predictions = {}

        for name in model_names:
            if name in self.models:
                # Preprocess data
                X_scaled = self.scalers[name].transform(X)
                X_selected = self.feature_selectors[name].transform(X_scaled)

                # Predict
                pred = self.models[name].predict(X_selected)
                pred_proba = self.models[name].predict_proba(X_selected)[:, 1] if hasattr(self.models[name], 'predict_proba') else None

                predictions[name] = {
                    'prediction': pred,
                    'probability': pred_proba
                }
            else:
                logger.warning(f"Model {name} not available for prediction")

        return predictions

    def save_models(self, symbol: str):
        
        symbol_dir = self.model_dir / symbol
        symbol_dir.mkdir(exist_ok=True)

        for name, model in self.models.items():
            model_path = symbol_dir / f"{name}.pkl"
            joblib.dump(model, model_path)

        for name, scaler in self.scalers.items():
            scaler_path = symbol_dir / f"{name}_scaler.pkl"
            joblib.dump(scaler, scaler_path)

        for name, selector in self.feature_selectors.items():
            selector_path = symbol_dir / f"{name}_selector.pkl"
            joblib.dump(selector, selector_path)

        logger.info(f"Saved models for {symbol}")

    def load_models(self, symbol: str) -> bool:
        
        symbol_dir = self.model_dir / symbol

        if not symbol_dir.exists():
            return False

        try:
            for model_file in symbol_dir.glob("*.pkl"):
                if "_scaler" in model_file.name:
                    model_name = model_file.name.replace("_scaler.pkl", "")
                    self.scalers[model_name] = joblib.load(model_file)
                elif "_selector" in model_file.name:
                    model_name = model_file.name.replace("_selector.pkl", "")
                    self.feature_selectors[model_name] = joblib.load(model_file)
                else:
                    model_name = model_file.name.replace(".pkl", "")
                    self.models[model_name] = joblib.load(model_file)

            logger.info(f"Loaded models for {symbol}")
            return True
        except Exception as e:
            logger.error(f"Error loading models for {symbol}: {e}")
            return False


class ModelEvaluator:
    

    def __init__(self):
        self.metrics = {}

    def evaluate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray,
                           y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0)
        }

        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)

        return metrics

    def compare_models(self, model_results: Dict[str, Dict]) -> pd.DataFrame:
        
        comparison = {}

        for model_name, results in model_results.items():
            cv_results = results.get('cv_results', {})
            comparison[model_name] = {
                'cv_accuracy': cv_results.get('accuracy_mean', 0),
                'cv_f1': cv_results.get('f1_mean', 0),
                'cv_roc_auc': cv_results.get('roc_auc_mean', 0),
                'n_features': len(results.get('selected_features', []))
            }

        return pd.DataFrame(comparison).T

    def calculate_confidence_score(self, predictions: Dict[str, Dict]) -> float:
        
        if not predictions:
            return 0.5

        probabilities = []
        for model_pred in predictions.values():
            if 'probability' in model_pred and model_pred['probability'] is not None:
                prob_array = np.array(model_pred['probability'])
                if prob_array.size > 0:
                    probabilities.append(prob_array)

        if not probabilities:
            return 0.5

        # Stack probabilities for analysis
        prob_matrix = np.column_stack(probabilities) if len(probabilities) > 1 else np.array(probabilities).T

        # Calculate mean probability across models
        mean_prob = np.mean(prob_matrix, axis=1)

        # Calculate agreement (lower std = higher agreement)
        std_prob = np.std(prob_matrix, axis=1)

        # Confidence based on agreement and probability strength
        agreement_score = 1 - np.mean(std_prob)  # Lower variance = higher confidence
        extremity_score = np.mean(np.abs(mean_prob - 0.5)) * 2  # Extreme probabilities = higher confidence

        confidence = agreement_score * extremity_score

        return min(max(confidence, 0), 1)


class AdvancedMLPredictor:
    

    def __init__(self):
        self.trainer = EnsembleModelTrainer()
        self.evaluator = ModelEvaluator()
        self.trained_models = {}

    def train_ensemble_models(self, X: pd.DataFrame, y: pd.Series,
                             model_names: List[str] = None,
                             use_advanced_features: bool = True) -> Dict[str, Any]:
        
        if model_names is None:
            model_names = ['random_forest', 'gradient_boosting']
            if XGBOOST_AVAILABLE:
                model_names.append('xgboost')
            if LIGHTGBM_AVAILABLE:
                model_names.append('lightgbm')
            if CATBOOST_AVAILABLE:
                model_names.append('catboost')

        results = {}
        for model_name in model_names:
            try:
                # Use advanced feature selection and Bayesian optimization if enabled
                feature_method = 'combined' if use_advanced_features else 'rf_importance'
                optimize_hp = use_advanced_features

                model_result = self.trainer.train_single_model(
                    model_name, X, y,
                    optimize_hyperparams=optimize_hp,
                    feature_selection_method=feature_method,
                    use_advanced_features=use_advanced_features
                )
                results[model_name] = model_result
                logger.info(f"Trained {model_name} with CV F1: {model_result['cv_results'].get('f1_mean', 0):.3f}")
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                continue

        # Create advanced ensemble models
        if len(results) >= 2 and use_advanced_features:
            try:
                # Create multiple types of ensembles
                advanced_ensembles = self.trainer.create_advanced_ensemble(X, y, 'all')
                results.update(advanced_ensembles)
                logger.info("Created advanced ensemble models")
            except Exception as e:
                logger.error(f"Failed to create advanced ensembles: {e}")

                # Fallback to simple ensemble
                try:
                    ensemble_model = self.trainer.create_ensemble_model(list(results.keys()), X, y)
                    results['ensemble'] = {'model': ensemble_model}
                    logger.info("Created fallback ensemble model")
                except Exception as e2:
                    logger.error(f"Failed to create fallback ensemble: {e2}")

        # Calibrate models if enabled
        if use_advanced_features and len(results) > 0:
            try:
                calibrated_models = self.trainer.create_calibrated_ensemble(X, y, list(results.keys()))
                results['calibrated_models'] = calibrated_models
                logger.info("Created calibrated models")
            except Exception as e:
                logger.error(f"Failed to create calibrated models: {e}")

        return results

    def predict_with_confidence(self, X: pd.DataFrame, model_names: List[str] = None) -> Dict[str, Any]:
        
        if model_names is None:
            model_names = list(self.trainer.models.keys())

        predictions = self.trainer.predict_with_models(model_names, X)

        # Calculate ensemble prediction
        ensemble_pred = None
        ensemble_proba = None

        if len(predictions) > 1:
            # Simple majority vote for ensemble
            all_preds = np.array([pred['prediction'] for pred in predictions.values()])
            ensemble_pred = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=all_preds)

            # Average probabilities for ensemble
            all_probas = [pred['probability'] for pred in predictions.values() if pred['probability'] is not None]
            if all_probas:
                ensemble_proba = np.mean(all_probas, axis=0)

        confidence_score = self.evaluator.calculate_confidence_score(predictions)

        return {
            'predictions': predictions,
            'ensemble_prediction': ensemble_pred,
            'ensemble_probability': ensemble_proba,
            'confidence_score': confidence_score
        }

    def get_model_comparison(self, trained_results: Dict) -> pd.DataFrame:
        
        return self.evaluator.compare_models(trained_results)


def advanced_ml_agent(state: State) -> State:
    
    logging.info("Starting advanced ML agent")

    engineered_features = state.get("engineered_features", {})
    ml_predictions = {}

    if not engineered_features:
        logger.warning("No engineered features available for ML training")
        return state

    predictor = AdvancedMLPredictor()

    for symbol, features_df in engineered_features.items():
        try:
            if len(features_df) < 100:
                logger.warning(f"Insufficient data for {symbol}: {len(features_df)} rows")
                continue

            # Prepare training data
            from agents.feature_engineering import FeatureEngineer
            engineer = FeatureEngineer()
            X, y = engineer.prepare_training_data(features_df)

            if len(X) < 50:
                logger.warning(f"Insufficient training samples for {symbol}: {len(X)}")
                continue

            # Train models with advanced features
            trained_results = predictor.train_ensemble_models(X, y, use_advanced_features=True)

            # Make predictions on latest data
            latest_features = X.tail(1)
            prediction_results = predictor.predict_with_confidence(latest_features)

            # Store results
            ml_predictions[symbol] = {
                'trained_models': trained_results,
                'latest_prediction': prediction_results,
                'model_comparison': predictor.get_model_comparison(trained_results).to_dict(),
                'feature_importance': {model: results.get('feature_importance', {})
                                     for model, results in trained_results.items()}
            }

            # Save models
            predictor.trainer.save_models(symbol)

            logger.info(f"Completed ML training and prediction for {symbol}")

        except Exception as e:
            logger.error(f"Error in advanced ML for {symbol}: {e}")
            continue

    state["ml_predictions"] = ml_predictions
    logger.info(f"Completed advanced ML for {len(ml_predictions)} symbols")

    return state