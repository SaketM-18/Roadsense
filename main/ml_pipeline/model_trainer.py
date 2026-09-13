import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

class ModelTrainer:
    """
    ModelTrainer handles the splitting of data, handling of class imbalances via SMOTE,
    training the Random Forest model, evaluating its performance, and saving it to disk.
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = RandomForestClassifier(n_estimators=100, random_state=self.random_state, class_weight='balanced')
        
    def prepare_data(self, df):
        """
        Prepares the feature matrix X and target vector y.
        Drops non-feature columns like start_time and end_time.
        
        Args:
            df (pd.DataFrame): DataFrame containing features and labels.
            
        Returns:
            X, y
        """
        if 'label' not in df.columns:
            raise ValueError("Dataframe must contain a 'label' column for training.")
            
        # Drop metadata columns that shouldn't be used as features
        drop_cols = ['label', 'start_time', 'end_time']
        features = [col for col in df.columns if col not in drop_cols]
        
        X = df[features]
        y = df['label']
        
        return X, y

    def train(self, df, save_path=None):
        """
        Trains the Machine Learning model using the provided dataframe.
        Applies SMOTE to handle class imbalance (e.g., fewer potholes than normal road).
        
        Args:
            df (pd.DataFrame): The extracted features and labels.
            save_path (str, optional): Path to save the trained model (.pkl).
        """
        X, y = self.prepare_data(df)
        
        # Split into Train and Test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        print(f"\nOriginal Training Set Distribution:\n{y_train.value_counts()}")
        
        # Handle Class Imbalance using SMOTE
        # SMOTE generates synthetic samples for minority classes to balance the dataset.
        # This prevents the model from just always predicting "Normal".
        smote = SMOTE(random_state=self.random_state)
        try:
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            print(f"\nResampled Training Set Distribution (SMOTE):\n{y_train_resampled.value_counts()}")
        except ValueError as e:
            # In case there are too few samples for SMOTE (e.g. k_neighbors condition fails)
            print(f"Warning: SMOTE failed ({e}). Proceeding without SMOTE.")
            X_train_resampled, y_train_resampled = X_train, y_train

        # Train the Random Forest
        print("\nTraining Random Forest Classifier...")
        self.model.fit(X_train_resampled, y_train_resampled)
        
        # Evaluate
        print("\nEvaluating Model on Test Set...")
        y_pred = self.model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # Save model if path provided
        if save_path:
            joblib.dump(self.model, save_path)
            
            # Save the feature names so we can ensure alignment during inference
            feature_names_path = save_path.replace('.pkl', '_features.pkl')
            joblib.dump(list(X.columns), feature_names_path)
            print(f"\nModel saved to {save_path}")
            
    def load_model(self, model_path):
        """
        Loads a pre-trained model from disk.
        """
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            print(f"Model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")

    def predict(self, X):
        """
        Predicts labels for given features.
        
        Args:
            X (pd.DataFrame): Features to predict on.
            
        Returns:
            np.array: Predicted labels.
        """
        return self.model.predict(X)
