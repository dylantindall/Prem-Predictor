"""
This module trains the Premier League classifier and exposes a global `clf`
that is imported by run_simulation.py (via `from model import clf`).

It is not a generic usage example; the block at the bottom performs the
specific training routine required for the simulation workflow and should
remain present so run_simulation.py can import a ready-trained classifier.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

features =  ['home_win_form',
            'home_draw_form',
            'home_loss_form',
            'away_win_form',
            'away_draw_form',
            'away_loss_form',
            'home_position',
            'away_position',
            'home_points',
            'away_points',
            'position_gap',
            'points_gap',
            'home_team_label',
            'away_team_label',
            'h2h_home_wins',
            'h2h_draws',
            'h2h_away_wins',
            'home_avg_h2h_goals',
            'away_avg_h2h_goals',
            'home_team_goals_for',
            'home_team_goals_against',
            'home_team_goal_diff',
            'away_team_goals_for',
            'away_team_goals_against',
            'away_team_goal_diff']

class PremierLeagueClassifier:
    """
    Premier League match outcome classifier using Random Forest with hybrid temporal approach.
    Combines historical seasons with recent matchweek form.
    """
    
    def __init__(self, 
                 n_historical_seasons=3, 
                 recent_matchweeks=10, 
                 recent_weight=1.5,
                 verbose: bool = False):
        """
        Parameters:
        -----------
        n_historical_seasons : int
            Number of complete previous seasons to include in training
        recent_matchweeks : int
            Number of recent matchweeks from current season to include
        recent_weight : float
            Weight multiplier for recent matchweek data (1.0 = equal weight)
        """
        self.n_historical_seasons = n_historical_seasons
        self.recent_matchweeks = recent_matchweeks
        self.recent_weight = recent_weight
        self.model = None
        self.best_params = None
        self.feature_columns = None
        self.label_encoder = LabelEncoder()
        self.verbose = verbose
        
    def load_and_prepare_data(self, filepath = 'prem_results.csv', 
                             date_col='date',
                             matchweek_col='match_week',
                             season_col='season',
                             target_col='result',
                             feature_cols=None):
        """
        Load and prepare the dataset.
        
        Parameters:
        -----------
        filepath : str
            Path to CSV file
        date_col : str
            Name of date column
        matchweek_col : str
            Name of matchweek column
        season_col : str
            Name of season column
        target_col : str
            Name of target column (match result: H/D/A)
        feature_cols : list
            List of column names to use as features (excluding date, matchweek, season, target)
        """
        # Load data
        self.df = pd.read_csv(filepath)
        
        # Rename columns for consistency
        self.df = self.df.rename(columns={
            date_col: 'Date',
            matchweek_col: 'Matchweek',
            season_col: 'Season',
            target_col: 'FTR'
        })
        
        # Convert date to datetime
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        
        # Sort by date
        self.df = self.df.sort_values('Date').reset_index(drop=True)
        
        # Store feature columns
        if feature_cols is None:
            # Use all columns except metadata, identifiers, and labels
            self.feature_columns = [col for col in self.df.columns 
                                   if col not in ['Date', 'Matchweek', 'Season', 'FTR', 'home_team', 'away_team']]
        else:
            self.feature_columns = feature_cols
            
        if self.verbose:
            print(f"Dataset loaded: {len(self.df)} matches")
            print(f"Features to use ({len(self.feature_columns)}): {self.feature_columns}")
            print(f"Seasons available: {sorted(self.df['Season'].unique())}")
        
        return self.df
    
    def create_hybrid_train_test(self, current_season, current_matchweek):
        """
        Create training and test sets using hybrid temporal approach.
        
        Parameters:
        -----------
        current_season : str
            Current season identifier (e.g., '2024/25')
        current_matchweek : int
            Matchweek to predict
            
        Returns:
        --------
        X_train, y_train, X_test, y_test, sample_weights
        """
        # Get all available seasons
        all_seasons = sorted(self.df['Season'].unique())
        
        # Find index of current season
        try:
            current_season_idx = all_seasons.index(current_season)
        except ValueError:
            raise ValueError(f"Season {current_season} not found in dataset")
        
        # Get previous N complete seasons
        start_idx = max(0, current_season_idx - self.n_historical_seasons)
        previous_seasons = all_seasons[start_idx:current_season_idx]
        
        if len(previous_seasons) == 0:
            raise ValueError("Not enough historical seasons available")
        
        # Historical training data
        historical = self.df[self.df['Season'].isin(previous_seasons)].copy()
        historical['sample_weight'] = 1.0
        
        # Recent matchweeks from current season
        recent = self.df[
            (self.df['Season'] == current_season) & 
            (self.df['Matchweek'] >= current_matchweek - self.recent_matchweeks) &
            (self.df['Matchweek'] < current_matchweek)
        ].copy()
        recent['sample_weight'] = self.recent_weight
        
        # Combine training data
        train = pd.concat([historical, recent], ignore_index=True)
        
        # Test data (current matchweek)
        test = self.df[
            (self.df['Season'] == current_season) & 
            (self.df['Matchweek'] == current_matchweek)
        ]
        
        if len(test) == 0:
            raise ValueError(f"No data found for {current_season}, Matchweek {current_matchweek}")
        
        # Prepare features and target
        X_train = train[self.feature_columns]
        y_train = train['FTR']
        sample_weights = train['sample_weight'].values
        
        X_test = test[self.feature_columns]
        y_test = test['FTR']
        
        if self.verbose:
            print(f"\nTraining set: {len(train)} matches")
            print(f"  - Historical ({len(previous_seasons)} seasons): {len(historical)} matches")
            print(f"  - Recent (last {self.recent_matchweeks} matchweeks): {len(recent)} matches")
            print(f"Test set: {len(test)} matches (Matchweek {current_matchweek})")
        
        return X_train, y_train, X_test, y_test, sample_weights
    
    def tune_hyperparameters(self, X_train, y_train, sample_weights, n_iter=50):
        """
        Perform hyperparameter tuning using RandomizedSearchCV with TimeSeriesSplit.
        
        Parameters:
        -----------
        X_train : DataFrame
            Training features
        y_train : Series
            Training target
        sample_weights : array
            Sample weights for training
        n_iter : int
            Number of parameter settings sampled
        """
        if self.verbose:
            print("\n" + "="*60)
            print("HYPERPARAMETER TUNING")
            print("="*60)
        
        # Define parameter distribution
        param_dist = {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [10, 15, 20, 25, 30, None],
            'min_samples_split': [2, 5, 10, 15],
            'min_samples_leaf': [1, 2, 4, 6],
            'max_features': ['sqrt', 'log2', None],
            'class_weight': ['balanced', 'balanced_subsample', None],
            'bootstrap': [True, False]
        }
        
        # Base model
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Randomized search
        random_search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=tscv,
            scoring='accuracy',
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        # Fit (note: sample_weights not used in CV, but will be used in final fit)
        random_search.fit(X_train, y_train)
        
        self.best_params = random_search.best_params_
        print(f"Best params: {self.best_params}")
        print(f"Best CV accuracy: {random_search.best_score_:.4f}")
        
        
        return self.best_params
    
    def train(self, X_train, y_train, sample_weights, params=None):
        """
        Train the Random Forest model with given parameters.
        
        Parameters:
        -----------
        X_train : DataFrame
            Training features
        y_train : Series
            Training target
        sample_weights : array
            Sample weights for training
        params : dict
            Model parameters (if None, uses best_params from tuning)
        """
        if params is None:
            if self.best_params is None:
                print("No parameters provided. Using default parameters.")
                params = {'n_estimators': 200, 'random_state': 42}
            else:
                params = self.best_params.copy()
                params['random_state'] = 42
        
        if self.verbose:
            print("\n" + "="*60)
            print("TRAINING MODEL")
            print("="*60)
            print(f"Parameters: {params}")
        
        self.model = RandomForestClassifier(**params)
        self.model.fit(X_train, y_train, sample_weight=sample_weights)
        
        print("Model training complete!")
        
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Parameters:
        -----------
        X_test : DataFrame
            Test features
        y_test : Series
            Test target
            
        Returns:
        --------
        predictions, accuracy
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        if self.verbose:
            print("\n" + "="*60)
            print("MODEL EVALUATION")
            print("="*60)
        
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        print(f"Accuracy: {accuracy:.4f}")
        if self.verbose:
            print(f"\nClassification Report:")
            print(classification_report(y_test, predictions))
            # Confusion Matrix
            cm = confusion_matrix(y_test, predictions, labels=['H', 'D', 'A'])
            print("\nConfusion Matrix:")
            print("              Predicted")
            print("              H    D    A")
            for i, label in enumerate(['H', 'D', 'A']):
                print(f"Actual {label}:    {cm[i][0]:3d}  {cm[i][1]:3d}  {cm[i][2]:3d}")
        
        return predictions, accuracy
    
    def plot_feature_importance(self, top_n=20):
        """
        Plot feature importance from the trained model.
        
        Parameters:
        -----------
        top_n : int
            Number of top features to display
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Get feature importances
        importances = self.model.feature_importances_
        # Ensure we do not request more than available features
        top_n_effective = min(top_n, len(importances))
        indices = np.argsort(importances)[::-1][:top_n_effective]
        
        # Create DataFrame for easy viewing
        feature_importance_df = pd.DataFrame({
            'Feature': [self.feature_columns[i] for i in indices],
            'Importance': importances[indices]
        })
        
        print("\n" + "="*60)
        print(f"TOP {top_n_effective} MOST IMPORTANT FEATURES")
        print("="*60)
        print(feature_importance_df.to_string(index=False))
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.barh(range(top_n_effective), importances[indices])
        plt.yticks(range(top_n_effective), [self.feature_columns[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n_effective} Most Important Features')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
        
        return feature_importance_df
    
    def walk_forward_validation(self, current_season, start_matchweek=None, end_matchweek=None):
        """
        Perform walk-forward validation for the current season.
        
        Parameters:
        -----------
        current_season : str
            Season to validate on
        start_matchweek : int
            Starting matchweek (default: recent_matchweeks + 1)
        end_matchweek : int
            Ending matchweek (default: max available)
            
        Returns:
        --------
        DataFrame with results for each matchweek
        """
        if self.verbose:
            print("\n" + "="*60)
            print("WALK-FORWARD VALIDATION")
            print("="*60)
        
        # Get matchweeks for current season
        season_data = self.df[self.df['Season'] == current_season]
        available_matchweeks = sorted(season_data['Matchweek'].unique())
        
        if start_matchweek is None:
            start_matchweek = self.recent_matchweeks + 1
        if end_matchweek is None:
            end_matchweek = max(available_matchweeks)
        
        matchweeks_to_test = [mw for mw in available_matchweeks 
                             if start_matchweek <= mw <= end_matchweek]
        
        results = []
        all_predictions = []
        all_actuals = []
        
        for mw in matchweeks_to_test:
            if self.verbose:
                print(f"\nPredicting Matchweek {mw}...")
            
            try:
                # Create train/test split
                X_train, y_train, X_test, y_test, weights = self.create_hybrid_train_test(
                    current_season, mw
                )
                
                # Train model (use best params if available, otherwise defaults)
                params = self.best_params.copy() if self.best_params else {}
                params['random_state'] = 42
                if 'random_state' not in params:
                    params['n_estimators'] = 200
                
                model = RandomForestClassifier(**params)
                model.fit(X_train, y_train, sample_weight=weights)
                
                # Predict
                predictions = model.predict(X_test)
                accuracy = accuracy_score(y_test, predictions)
                
                # Store results
                results.append({
                    'Matchweek': mw,
                    'Accuracy': accuracy,
                    'Matches': len(y_test)
                })
                
                all_predictions.extend(predictions)
                all_actuals.extend(y_test.values)
                
                if self.verbose:
                    print(f"  Accuracy: {accuracy:.4f}")
                
            except Exception as e:
                if self.verbose:
                    print(f"  Error: {str(e)}")
                continue
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Calculate overall statistics
        overall_accuracy = accuracy_score(all_actuals, all_predictions)
        
        print("Walk-forward complete.")
        print(f"Overall Accuracy: {overall_accuracy:.4f}")
        print(f"Average Accuracy: {results_df['Accuracy'].mean():.4f}")
        print(f"Std Dev: {results_df['Accuracy'].std():.4f}")
        
        # Plot results
        plt.figure(figsize=(12, 6))
        plt.plot(results_df['Matchweek'], results_df['Accuracy'], marker='o', linewidth=2)
        plt.axhline(y=overall_accuracy, color='r', linestyle='--', 
                   label=f'Overall Accuracy: {overall_accuracy:.4f}')
        plt.xlabel('Matchweek')
        plt.ylabel('Accuracy')
        plt.title(f'Walk-Forward Validation Results - {current_season}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return results_df

    def save_predictions_csv(self, current_season, start_matchweek=None, end_matchweek=None, output_path='predictions.csv'):
        """
        Generate predictions for each matchweek in range and save CSV with team names.
        Output columns: home_team, away_team, predicted_result, actual_result
        """
        # Determine available matchweeks
        season_data = self.df[self.df['Season'] == current_season]
        available_matchweeks = sorted(season_data['Matchweek'].unique())
        if start_matchweek is None:
            start_matchweek = self.recent_matchweeks + 1
        if end_matchweek is None:
            end_matchweek = max(available_matchweeks)

        matchweeks_to_process = [mw for mw in available_matchweeks 
                                 if start_matchweek <= mw <= end_matchweek]

        rows = []
        for mw in matchweeks_to_process:
            try:
                X_train, y_train, X_test, y_test, weights = self.create_hybrid_train_test(current_season, mw)

                params = self.best_params.copy() if self.best_params else {}
                params['random_state'] = 42
                if 'n_estimators' not in params:
                    params['n_estimators'] = 200

                model = RandomForestClassifier(**params)
                model.fit(X_train, y_train, sample_weight=weights)

                preds = model.predict(X_test)

                # Attach team names and actuals from the original df slice
                test_slice = self.df[(self.df['Season'] == current_season) & (self.df['Matchweek'] == mw)]

                for (_, row_df), pred, actual in zip(test_slice.iterrows(), preds, y_test.values):
                    rows.append({
                        'home_team': row_df['home_team'],
                        'away_team': row_df['away_team'],
                        'predicted_result': pred,
                        'actual_result': actual
                    })
            except Exception as e:
                print(f"Error creating predictions for matchweek {mw}: {e}")
                continue

        pd.DataFrame(rows).to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")


# ============================================================================
# MODEL TRAINING FOR run_simulation.py
# ----------------------------------------------------------------------------
# This section trains the classifier and defines the global `clf` used by
# run_simulation.py. If removed, `from model import clf` in run_simulation.py
# will fail. Keep this block to ensure the simulation has a trained model.
# ============================================================================

# Initialize classifier
clf = PremierLeagueClassifier(
    n_historical_seasons= 4,    # Change this value
    recent_matchweeks= 38,      # Train only on the last 4 complete seasons (no current-season recent weeks)
    recent_weight= 1.35         # Change this value
)

# Load data - SPECIFY YOUR COLUMNS HERE
df = clf.load_and_prepare_data(
    filepath='prem_results.csv',
    date_col='date',           # Your date column name
    matchweek_col='match_week', # Your matchweek column name
    season_col='season',        # Your season column name
    target_col='result',          # Your target column name (H/D/A)
    feature_cols=features          # Specify features: ['col1', 'col2', ...] or None for all
)

# ==========================
# Train model on history
# ==========================
# Pick the most recent season in prem_results as the anchor; training will use the 4 seasons before it.
current_season = sorted(df['Season'].unique())[-1]
anchor_matchweek = int(df[df['Season'] == current_season]['Matchweek'].max())

X_train, y_train, X_test, y_test, weights = clf.create_hybrid_train_test(
    current_season=current_season,
    current_matchweek=anchor_matchweek
)

# Hyperparameter tuning (optional)
best_params = clf.tune_hyperparameters(X_train, y_train, weights, n_iter=50)

# Final train
clf.train(X_train, y_train, weights)