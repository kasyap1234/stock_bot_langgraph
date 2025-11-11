

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import joblib
import os
from pathlib import Path

try:
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module="tensorflow")
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (
        Dense, LSTM, GRU, Bidirectional, Conv1D, MaxPooling1D,
        Flatten, Dropout, BatchNormalization, Input, Attention,
        MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D,
        TimeDistributed, RepeatVector, Reshape, UpSampling1D
    )
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2
    TENSORFLOW_AVAILABLE = True
except ImportError:
    logging.warning("TensorFlow not available, neural networks disabled")
    TENSORFLOW_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    PYTORCH_AVAILABLE = True
except ImportError:
    logging.warning("PyTorch not available")
    PYTORCH_AVAILABLE = False

from config.ml_config import LSTM_EPOCHS, LSTM_BATCH
from config.constants import MODEL_DIR
from data.models import State

logger = logging.getLogger(__name__)

if PYTORCH_AVAILABLE:
    class TimeSeriesDataset(Dataset):
        def __init__(self, X: np.ndarray, y: np.ndarray):
            self.X = torch.FloatTensor(X)
            self.y = torch.LongTensor(y)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]


class CNNTimeSeriesPredictor:
    

    def __init__(self, sequence_length: int = 60, n_features: int = 50):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.scaler = None

        if TENSORFLOW_AVAILABLE:
            self._build_model()

    def _build_model(self):
        
        model = Sequential([
            Input(shape=(self.sequence_length, self.n_features)),
            Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.2),

            Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.2),

            Conv1D(filters=256, kernel_size=3, activation='relu', padding='same'),
            BatchNormalization(),
            GlobalAveragePooling1D(),

            Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.3),
            Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )

        self.model = model
        logger.info("CNN model built")

    def prepare_sequences(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        
        from sklearn.preprocessing import StandardScaler

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Create sequences
        X_seq, y_seq = [], []

        for i in range(self.sequence_length, len(X_scaled)):
            X_seq.append(X_scaled[i-self.sequence_length:i])
            y_seq.append(y.iloc[i])

        return np.array(X_seq), np.array(y_seq)

    def train(self, X: pd.DataFrame, y: pd.Series, validation_split: float = 0.2) -> Dict[str, Any]:
        
        if not TENSORFLOW_AVAILABLE or self.model is None:
            return {'error': 'TensorFlow not available'}

        X_seq, y_seq = self.prepare_sequences(X, y)

        if len(X_seq) < 100:
            return {'error': 'Insufficient training data'}

        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        # Train model
        history = self.model.fit(
            X_seq, y_seq,
            epochs=LSTM_EPOCHS,
            batch_size=LSTM_BATCH,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=0
        )

        return {
            'model': self.model,
            'history': history.history,
            'scaler': self.scaler,
            'final_accuracy': history.history['accuracy'][-1],
            'final_val_accuracy': history.history['val_accuracy'][-1]
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        
        if self.model is None or self.scaler is None:
            return np.array([])

        X_scaled = self.scaler.transform(X.tail(self.sequence_length))
        X_seq = X_scaled.reshape(1, self.sequence_length, -1)

        prediction = self.model.predict(X_seq, verbose=0)
        return prediction.flatten()


class RNNPredictor:
    

    def __init__(self, sequence_length: int = 60, n_features: int = 50,
                 model_type: str = 'bilstm'):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model_type = model_type  # 'lstm', 'gru', 'bilstm'
        self.model = None
        self.scaler = None

        if TENSORFLOW_AVAILABLE:
            self._build_model()

    def _build_model(self):
        
        inputs = Input(shape=(self.sequence_length, self.n_features))

        if self.model_type == 'lstm':
            rnn_layer = LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
        elif self.model_type == 'gru':
            rnn_layer = GRU(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
        elif self.model_type == 'bilstm':
            rnn_layer = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        x = rnn_layer(inputs)
        x = BatchNormalization()(x)

        # Attention mechanism
        attention = Attention()([x, x])
        x = GlobalAveragePooling1D()(attention)

        x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
        x = Dropout(0.3)(x)
        x = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(x)
        x = Dropout(0.2)(x)

        outputs = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )

        self.model = model
        logger.info(f"{self.model_type.upper()} model built")

    def prepare_sequences(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        
        from sklearn.preprocessing import StandardScaler

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        X_seq, y_seq = [], []

        for i in range(self.sequence_length, len(X_scaled)):
            X_seq.append(X_scaled[i-self.sequence_length:i])
            y_seq.append(y.iloc[i])

        return np.array(X_seq), np.array(y_seq)

    def train(self, X: pd.DataFrame, y: pd.Series, validation_split: float = 0.2) -> Dict[str, Any]:
        
        if not TENSORFLOW_AVAILABLE or self.model is None:
            return {'error': 'TensorFlow not available'}

        X_seq, y_seq = self.prepare_sequences(X, y)

        if len(X_seq) < 100:
            return {'error': 'Insufficient training data'}

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        history = self.model.fit(
            X_seq, y_seq,
            epochs=LSTM_EPOCHS,
            batch_size=LSTM_BATCH,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=0
        )

        return {
            'model': self.model,
            'history': history.history,
            'scaler': self.scaler,
            'final_accuracy': history.history['accuracy'][-1],
            'final_val_accuracy': history.history['val_accuracy'][-1]
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        
        if self.model is None or self.scaler is None:
            return np.array([])

        X_scaled = self.scaler.transform(X.tail(self.sequence_length))
        X_seq = X_scaled.reshape(1, self.sequence_length, -1)

        prediction = self.model.predict(X_seq, verbose=0)
        return prediction.flatten()


class TransformerPredictor:
    

    def __init__(self, sequence_length: int = 60, n_features: int = 50,
                 num_heads: int = 8, ff_dim: int = 128, num_transformer_blocks: int = 4):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.model = None
        self.scaler = None

        if TENSORFLOW_AVAILABLE:
            self._build_model()

    def _build_model(self):
        
        inputs = Input(shape=(self.sequence_length, self.n_features))

        # Positional encoding
        positions = tf.range(start=0, limit=self.sequence_length, delta=1)
        position_embedding = tf.keras.layers.Embedding(
            input_dim=self.sequence_length, output_dim=self.n_features
        )(positions)
        x = inputs + position_embedding

        # Transformer blocks
        for _ in range(self.num_transformer_blocks):
            # Multi-head attention
            attn_output = MultiHeadAttention(
                num_heads=self.num_heads, key_dim=self.n_features
            )(x, x)
            attn_output = Dropout(0.1)(attn_output)
            x = LayerNormalization(epsilon=1e-6)(x + attn_output)

            # Feed-forward network
            ffn = tf.keras.Sequential([
                Dense(self.ff_dim, activation='relu'),
                Dense(self.n_features)
            ])
            ffn_output = ffn(x)
            ffn_output = Dropout(0.1)(ffn_output)
            x = LayerNormalization(epsilon=1e-6)(x + ffn_output)

        # Global pooling and classification
        x = GlobalAveragePooling1D()(x)
        x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
        x = Dropout(0.3)(x)
        x = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(x)
        x = Dropout(0.2)(x)

        outputs = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )

        self.model = model
        logger.info("Transformer model built")

    def prepare_sequences(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        
        from sklearn.preprocessing import StandardScaler

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        X_seq, y_seq = [], []

        for i in range(self.sequence_length, len(X_scaled)):
            X_seq.append(X_scaled[i-self.sequence_length:i])
            y_seq.append(y.iloc[i])

        return np.array(X_seq), np.array(y_seq)

    def train(self, X: pd.DataFrame, y: pd.Series, validation_split: float = 0.2) -> Dict[str, Any]:
        
        if not TENSORFLOW_AVAILABLE or self.model is None:
            return {'error': 'TensorFlow not available'}

        X_seq, y_seq = self.prepare_sequences(X, y)

        if len(X_seq) < 100:
            return {'error': 'Insufficient training data'}

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        history = self.model.fit(
            X_seq, y_seq,
            epochs=LSTM_EPOCHS,
            batch_size=LSTM_BATCH,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=0
        )

        return {
            'model': self.model,
            'history': history.history,
            'scaler': self.scaler,
            'final_accuracy': history.history['accuracy'][-1],
            'final_val_accuracy': history.history['val_accuracy'][-1]
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        
        if self.model is None or self.scaler is None:
            return np.array([])

        X_scaled = self.scaler.transform(X.tail(self.sequence_length))
        X_seq = X_scaled.reshape(1, self.sequence_length, -1)

        prediction = self.model.predict(X_seq, verbose=0)
        return prediction.flatten()


class AutoencoderAnomalyDetector:
    

    def __init__(self, sequence_length: int = 60, n_features: int = 50,
                 encoding_dim: int = 32):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.encoding_dim = encoding_dim
        self.autoencoder = None
        self.encoder = None
        self.scaler = None

        if TENSORFLOW_AVAILABLE:
            self._build_autoencoder()

    def _build_autoencoder(self):
        
        input_layer = Input(shape=(self.sequence_length, self.n_features))

        # Encoder
        encoded = Conv1D(64, 3, activation='relu', padding='same')(input_layer)
        encoded = MaxPooling1D(2)(encoded)
        encoded = Conv1D(32, 3, activation='relu', padding='same')(encoded)
        encoded = MaxPooling1D(2)(encoded)
        encoded = Flatten()(encoded)
        encoded = Dense(self.encoding_dim, activation='relu')(encoded)

        # Decoder
        decoded = Dense(self.sequence_length * self.n_features // 4, activation='relu')(encoded)
        decoded = Reshape((self.sequence_length // 4, self.n_features))(decoded)
        decoded = Conv1D(32, 3, activation='relu', padding='same')(decoded)
        decoded = UpSampling1D(2)(decoded)
        decoded = Conv1D(64, 3, activation='relu', padding='same')(decoded)
        decoded = UpSampling1D(2)(decoded)
        decoded = Conv1D(self.n_features, 3, activation='sigmoid', padding='same')(decoded)

        # Autoencoder model
        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')

        # Encoder model for anomaly scoring
        encoder = Model(input_layer, encoded)

        self.autoencoder = autoencoder
        self.encoder = encoder
        logger.info("Autoencoder model built")

    def prepare_sequences(self, X: pd.DataFrame) -> np.ndarray:
        
        from sklearn.preprocessing import MinMaxScaler

        self.scaler = MinMaxScaler()
        X_scaled = self.scaler.fit_transform(X)

        X_seq = []
        for i in range(self.sequence_length, len(X_scaled)):
            X_seq.append(X_scaled[i-self.sequence_length:i])

        return np.array(X_seq)

    def train(self, X: pd.DataFrame, epochs: int = 50, batch_size: int = 32) -> Dict[str, Any]:
        
        if not TENSORFLOW_AVAILABLE or self.autoencoder is None:
            return {'error': 'TensorFlow not available'}

        X_seq = self.prepare_sequences(X)

        if len(X_seq) < 100:
            return {'error': 'Insufficient training data'}

        early_stopping = EarlyStopping(
            monitor='loss',
            patience=10,
            restore_best_weights=True
        )

        history = self.autoencoder.fit(
            X_seq, X_seq,  # Input and target are the same for autoencoder
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )

        # Calculate reconstruction errors for threshold setting
        reconstructions = self.autoencoder.predict(X_seq, verbose=0)
        mse = np.mean(np.power(X_seq - reconstructions, 2), axis=(1, 2))
        threshold = np.percentile(mse, 95)  # 95th percentile as anomaly threshold

        return {
            'autoencoder': self.autoencoder,
            'encoder': self.encoder,
            'scaler': self.scaler,
            'history': history.history,
            'threshold': threshold,
            'reconstruction_errors': mse
        }

    def detect_anomalies(self, X: pd.DataFrame) -> Dict[str, Any]:
        
        if self.autoencoder is None or self.scaler is None:
            return {'error': 'Model not trained'}

        X_scaled = self.scaler.transform(X)
        X_seq = []

        # Create sequences for the entire dataset
        for i in range(self.sequence_length, len(X_scaled)):
            X_seq.append(X_scaled[i-self.sequence_length:i])

        if not X_seq:
            return {'error': 'Insufficient data for anomaly detection'}

        X_seq = np.array(X_seq)
        reconstructions = self.autoencoder.predict(X_seq, verbose=0)
        mse = np.mean(np.power(X_seq - reconstructions, 2), axis=(1, 2))

        # Get threshold from training if available, otherwise use percentile
        threshold = getattr(self, 'threshold', np.percentile(mse, 95))

        anomalies = mse > threshold
        anomaly_indices = np.where(anomalies)[0] + self.sequence_length

        return {
            'anomaly_indices': anomaly_indices.tolist(),
            'reconstruction_errors': mse.tolist(),
            'threshold': threshold,
            'anomaly_scores': anomalies.tolist(),
            'n_anomalies': int(np.sum(anomalies))
        }


class NeuralNetworkEnsemble:
    

    def __init__(self, sequence_length: int = 60):
        self.sequence_length = sequence_length
        self.models = {
            'cnn': CNNTimeSeriesPredictor(sequence_length=sequence_length),
            'bilstm': RNNPredictor(sequence_length=sequence_length, model_type='bilstm'),
            'gru': RNNPredictor(sequence_length=sequence_length, model_type='gru'),
            'transformer': TransformerPredictor(sequence_length=sequence_length),
            'autoencoder': AutoencoderAnomalyDetector(sequence_length=sequence_length)
        }
        self.trained_models = {}

    def train_all_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        
        results = {}

        for name, model in self.models.items():
            if name == 'autoencoder':
                # Autoencoder doesn't need y
                result = model.train(X)
            else:
                result = model.train(X, y)

            if 'error' not in result:
                self.trained_models[name] = model
                results[name] = result
                logger.info(f"Trained {name} model successfully")
            else:
                logger.warning(f"Failed to train {name}: {result['error']}")

        return results

    def predict_with_ensemble(self, X: pd.DataFrame) -> Dict[str, Any]:
        
        predictions = {}
        probabilities = []

        for name, model in self.trained_models.items():
            if name != 'autoencoder':
                try:
                    pred = model.predict(X)
                    if len(pred) > 0:
                        predictions[name] = pred[0]
                        probabilities.append(pred[0])
                except Exception as e:
                    logger.warning(f"Prediction failed for {name}: {e}")

        # Ensemble prediction
        if probabilities:
            ensemble_pred = np.mean(probabilities)
            ensemble_std = np.std(probabilities)
            confidence = 1 - ensemble_std  # Lower variance = higher confidence
        else:
            ensemble_pred = 0.5
            ensemble_std = 0.0
            confidence = 0.0

        # Anomaly detection
        anomaly_result = {}
        if 'autoencoder' in self.trained_models:
            try:
                anomaly_result = self.trained_models['autoencoder'].detect_anomalies(X)
            except Exception as e:
                logger.warning(f"Anomaly detection failed: {e}")

        return {
            'individual_predictions': predictions,
            'ensemble_prediction': ensemble_pred,
            'ensemble_std': ensemble_std,
            'confidence': confidence,
            'anomalies': anomaly_result
        }

    def save_models(self, symbol: str, model_dir: str = MODEL_DIR):
        
        symbol_dir = Path(model_dir) / f"nn_{symbol}"
        symbol_dir.mkdir(exist_ok=True)

        for name, model in self.trained_models.items():
            try:
                if hasattr(model, 'model') and model.model is not None:
                    model_path = symbol_dir / f"{name}.h5"
                    model.model.save(model_path)
                if hasattr(model, 'autoencoder') and model.autoencoder is not None:
                    ae_path = symbol_dir / f"{name}_autoencoder.h5"
                    model.autoencoder.save(ae_path)
                if hasattr(model, 'scaler') and model.scaler is not None:
                    scaler_path = symbol_dir / f"{name}_scaler.pkl"
                    joblib.dump(model.scaler, scaler_path)
            except Exception as e:
                logger.warning(f"Failed to save {name} model: {e}")

        logger.info(f"Saved neural network models for {symbol}")

    def load_models(self, symbol: str, model_dir: str = MODEL_DIR) -> bool:
        
        symbol_dir = Path(model_dir) / f"nn_{symbol}"

        if not symbol_dir.exists():
            return False

        try:
            for model_file in symbol_dir.glob("*.h5"):
                model_name = model_file.name.replace('.h5', '').replace('_autoencoder', '')

                if model_name in self.models:
                    if TENSORFLOW_AVAILABLE:
                        from tensorflow.keras.models import load_model
                        if '_autoencoder' in model_file.name:
                            self.models[model_name].autoencoder = load_model(model_file)
                        else:
                            self.models[model_name].model = load_model(model_file)

            for scaler_file in symbol_dir.glob("*_scaler.pkl"):
                model_name = scaler_file.name.replace('_scaler.pkl', '')
                if model_name in self.models:
                    self.models[model_name].scaler = joblib.load(scaler_file)
                    self.trained_models[model_name] = self.models[model_name]

            logger.info(f"Loaded neural network models for {symbol}")
            return True
        except Exception as e:
            logger.error(f"Error loading neural network models for {symbol}: {e}")
            return False


def neural_network_agent(state: State) -> State:
    
    logging.info("Starting neural network agent")

    engineered_features = state.get("engineered_features", {})
    nn_predictions = {}

    if not engineered_features:
        logger.warning("No engineered features available for neural network training")
        return state

    if not TENSORFLOW_AVAILABLE:
        logger.warning("TensorFlow not available, skipping neural network models")
        return state

    for symbol, features_df in engineered_features.items():
        try:
            if len(features_df) < 120:  # Need more data for NN training
                logger.warning(f"Insufficient data for {symbol}: {len(features_df)} rows")
                continue

            # Prepare training data
            from agents.feature_engineering import FeatureEngineer
            engineer = FeatureEngineer()
            X, y = engineer.prepare_training_data(features_df)

            if len(X) < 100:
                logger.warning(f"Insufficient training samples for {symbol}: {len(X)}")
                continue

            # Train neural network ensemble
            nn_ensemble = NeuralNetworkEnsemble()
            training_results = nn_ensemble.train_all_models(X, y)

            # Make predictions
            prediction_results = nn_ensemble.predict_with_ensemble(X)

            # Store results
            nn_predictions[symbol] = {
                'training_results': training_results,
                'predictions': prediction_results
            }

            logger.info(f"Completed neural network training and prediction for {symbol}")

        except Exception as e:
            logger.error(f"Error in neural network processing for {symbol}: {e}")
            continue

    state["nn_predictions"] = nn_predictions
    logger.info(f"Completed neural network processing for {len(nn_predictions)} symbols")

    return state