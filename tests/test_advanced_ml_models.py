

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from agents.advanced_ml_models import (
    EnsembleModelTrainer, TimeSeriesCrossValidator,
    ModelEvaluator, AdvancedMLPredictor
)


class TestTimeSeriesCrossValidator:
    

    @pytest.fixture
    def sample_data(self):
        
        np.random.seed(42)
        data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'target': np.random.randint(0, 2, 100)
        })
        return data

    def test_initialization(self):
        
        cv = TimeSeriesCrossValidator(n_splits=3, test_size=20)
        assert cv.n_splits == 3
        assert cv.test_size == 20

    def test_split_generation(self, sample_data):
        
        cv = TimeSeriesCrossValidator(n_splits=3, test_size=20)
        splits = cv.split(sample_data)

        assert len(splits) == 3
        for train_idx, test_idx in splits:
            assert len(test_idx) == 20
            assert len(train_idx) > 20  # Should have training data
            assert max(test_idx) < len(sample_data)  # Valid indices

    def test_get_cv_splits(self, sample_data):
        
        cv = TimeSeriesCrossValidator(n_splits=2, test_size=10)
        X = sample_data[['feature1', 'feature2']]
        y = sample_data['target']

        splits = cv.get_cv_splits(X, y)

        assert len(splits) == 2
        for X_train, X_test, y_train, y_test in splits:
            assert isinstance(X_train, pd.DataFrame)
            assert isinstance(X_test, pd.DataFrame)
            assert isinstance(y_train, pd.Series)
            assert isinstance(y_test, pd.Series)
            assert len(X_test) == 10


class TestEnsembleModelTrainer:
    

    @pytest.fixture
    def sample_training_data(self):
        
        np.random.seed(42)
        n_samples = 200
        data = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'feature3': np.random.randn(n_samples),
            'target': np.random.randint(0, 2, n_samples)
        })
        return data

    @pytest.fixture
    def trainer(self):
        
        return EnsembleModelTrainer()

    def test_initialization(self, trainer):
        
        assert trainer.models == {}
        assert trainer.scalers == {}
        assert trainer.cv is not None

    def test_train_single_model(self, trainer, sample_training_data):
        
        X = sample_training_data[['feature1', 'feature2', 'feature3']]
        y = sample_training_data['target']

        result = trainer.train_single_model('random_forest', X, y, optimize_hyperparams=False)

        assert 'model' in result
        assert 'scaler' in result
        assert 'feature_selector' in result
        assert 'cv_results' in result
        assert 'feature_importance' in result

        # Check that model is trained
        assert result['model'] is not None
        assert len(result['feature_importance']) > 0

    def test_predict_with_models(self, trainer, sample_training_data):
        
        X = sample_training_data[['feature1', 'feature2', 'feature3']]
        y = sample_training_data['target']

        # Train a model first
        trainer.train_single_model('random_forest', X, y, optimize_hyperparams=False)

        # Make predictions
        predictions = trainer.predict_with_models(['random_forest'], X.head(10))

        assert 'random_forest' in predictions
        assert 'prediction' in predictions['random_forest']
        assert len(predictions['random_forest']['prediction']) == 10

    def test_create_ensemble_model(self, trainer, sample_training_data):
        
        X = sample_training_data[['feature1', 'feature2', 'feature3']]
        y = sample_training_data['target']

        # Train multiple models
        trainer.train_single_model('random_forest', X, y, optimize_hyperparams=False)

        # Mock another model
        with patch('sklearn.ensemble.GradientBoostingClassifier') as mock_gb:
            mock_gb.return_value.fit.return_value = MagicMock()
            trainer.models['gradient_boosting'] = mock_gb.return_value

            ensemble = trainer.create_ensemble_model(['random_forest', 'gradient_boosting'], X, y)

            assert ensemble is not None
            # Should be a VotingClassifier
            assert hasattr(ensemble, 'estimators_')

    @patch('joblib.dump')
    def test_save_models(self, mock_dump, trainer, tmp_path):
        
        trainer.model_dir = tmp_path

        # Mock trained models
        trainer.models['test_model'] = MagicMock()
        trainer.scalers['test_model'] = MagicMock()
        trainer.feature_selectors['test_model'] = MagicMock()

        trainer.save_models('TEST')

        # Should have called dump multiple times
        assert mock_dump.call_count >= 3

    @patch('os.path.exists')
    @patch('joblib.load')
    def test_load_models(self, mock_load, mock_exists, trainer, tmp_path):
        
        trainer.model_dir = tmp_path
        mock_exists.return_value = True
        mock_load.return_value = MagicMock()

        result = trainer.load_models('TEST')

        assert result is True
        assert mock_load.called


class TestModelEvaluator:
    

    @pytest.fixture
    def evaluator(self):
        
        return ModelEvaluator()

    def test_evaluate_predictions(self, evaluator):
        
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        y_pred_proba = np.array([0.3, 0.7, 0.4, 0.2, 0.8])

        metrics = evaluator.evaluate_predictions(y_true, y_pred, y_pred_proba)

        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'roc_auc' in metrics

        assert metrics['accuracy'] == 0.8  # 4/5 correct
        assert 0 <= metrics['roc_auc'] <= 1

    def test_compare_models(self, evaluator):
        
        model_results = {
            'model1': {
                'cv_results': {
                    'accuracy_mean': 0.8,
                    'f1_mean': 0.75,
                    'roc_auc_mean': 0.85
                },
                'selected_features': ['feat1', 'feat2']
            },
            'model2': {
                'cv_results': {
                    'accuracy_mean': 0.75,
                    'f1_mean': 0.7,
                    'roc_auc_mean': 0.8
                },
                'selected_features': ['feat1']
            }
        }

        comparison = evaluator.compare_models(model_results)

        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 2
        assert 'cv_accuracy' in comparison.columns
        assert comparison.loc['model1', 'cv_accuracy'] == 0.8

    def test_calculate_confidence_score(self, evaluator):
        
        # High agreement predictions
        predictions = {
            'model1': {'probability': np.array([0.8, 0.7])},
            'model2': {'probability': np.array([0.75, 0.65])},
            'model3': {'probability': np.array([0.85, 0.75])}
        }

        confidence = evaluator.calculate_confidence_score(predictions)

        assert 0 <= confidence <= 1
        assert confidence > 0.4  # Should be reasonably high confidence

        # Low agreement predictions
        predictions_diverse = {
            'model1': {'probability': np.array([0.9, 0.1])},
            'model2': {'probability': np.array([0.1, 0.9])},
        }

        confidence_diverse = evaluator.calculate_confidence_score(predictions_diverse)

        assert confidence_diverse < confidence  # Should be lower confidence


class TestAdvancedMLPredictor:
    

    @pytest.fixture
    def predictor(self):
        
        return AdvancedMLPredictor()

    @pytest.fixture
    def sample_data(self):
        
        np.random.seed(42)
        n_samples = 150
        data = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'feature3': np.random.randn(n_samples),
            'target': np.random.randint(0, 2, n_samples)
        })
        return data

    def test_initialization(self, predictor):
        
        assert predictor.trainer is not None
        assert predictor.evaluator is not None
        assert predictor.trained_models == {}

    def test_train_ensemble_models(self, predictor, sample_data):
        
        X = sample_data[['feature1', 'feature2', 'feature3']]
        y = sample_data['target']

        results = predictor.train_ensemble_models(X, y, model_names=['random_forest'])

        assert 'random_forest' in results
        assert 'model' in results['random_forest']
        assert 'cv_results' in results['random_forest']

    def test_predict_with_confidence(self, predictor, sample_data):
        
        X = sample_data[['feature1', 'feature2', 'feature3']]
        y = sample_data['target']

        # Train models first
        predictor.train_ensemble_models(X, y, model_names=['random_forest'])

        # Make predictions
        predictions = predictor.predict_with_confidence(X.head(5))

        assert 'individual_predictions' in predictions
        assert 'ensemble_prediction' in predictions
        assert 'confidence_score' in predictions
        assert 'anomalies' in predictions

        assert 0 <= predictions['ensemble_prediction'] <= 1
        assert 0 <= predictions['confidence_score'] <= 1

    def test_get_model_comparison(self, predictor, sample_data):
        
        X = sample_data[['feature1', 'feature2', 'feature3']]
        y = sample_data['target']

        trained_results = predictor.train_ensemble_models(X, y, model_names=['random_forest'])

        comparison = predictor.get_model_comparison(trained_results)

        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) >= 1


if __name__ == '__main__':
    pytest.main([__file__])