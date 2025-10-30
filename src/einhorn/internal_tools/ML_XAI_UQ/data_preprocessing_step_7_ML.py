#!/usr/bin/env python3
"""
BDE Machine Learning Pipeline with Conformal Prediction

Implements an XGBoost-based machine learning pipeline for predicting processing times
with uncertainty quantification using split conformal prediction.

This script:
1. Loads train/validation/test datasets from step 6
2. Preprocesses data for machine learning
3. Performs hyperparameter tuning for XGBoost based on MAE
4. Implements split conformal prediction for uncertainty quantification
5. Evaluates model performance with comprehensive metrics
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List, Any
import os
import sys
import warnings
import joblib
from datetime import datetime
import scipy.stats as stats
import gc  # For garbage collection to manage memory

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class BDEMLPipeline:
    
    def __init__(self, data_folder="bde_analysis_output/ml_data_splits"):
        self.data_folder = data_folder
        self.output_folder = "bde_analysis_output"
        self.ml_output_folder = os.path.join(self.output_folder, "ml_results")
        
        # Data containers
        self.train_df = None
        self.val_df = None
        self.test_df = None
        
        # Column definitions
        self.predictor_columns = []
        self.target_column = 'processing_time_minutes'
        self.additional_columns = []
        
        # Preprocessors
        self.label_encoders = {}
        self.scaler = None
        
        # Model and results
        self.best_model = None
        self.feature_importance = None
        self.predictions = {}
        self.conformal_scores = None
        self.alpha = 0.1  # For 90% prediction intervals
        
        self.create_output_folder()
    
    def create_output_folder(self):
        """Create output folder structure for ML results."""
        if not os.path.exists(self.ml_output_folder):
            os.makedirs(self.ml_output_folder)
    
    def load_datasets(self):
        """Load the train/validation/test datasets created in step 6."""
        print("=" * 70)
        print("BDE MACHINE LEARNING PIPELINE")
        print("=" * 70)
        
        print("Step 1: Loading datasets...")
        
        # Define file paths
        train_file = os.path.join(self.data_folder, "bde_train_dataset.csv")
        val_file = os.path.join(self.data_folder, "bde_validation_dataset.csv")
        test_file = os.path.join(self.data_folder, "bde_test_dataset.csv")
        column_def_file = os.path.join(self.data_folder, "column_definitions.csv")
        
        # Check if files exist
        for file_path in [train_file, val_file, test_file, column_def_file]:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required file not found: {file_path}")
        
        # Load datasets
        self.train_df = pd.read_csv(train_file, low_memory=False)
        self.val_df = pd.read_csv(val_file, low_memory=False)
        self.test_df = pd.read_csv(test_file, low_memory=False)
        
        # Load column definitions
        column_defs = pd.read_csv(column_def_file)
        self.predictor_columns = [col for col in column_defs['predictor_columns'].dropna() if col != '']
        self.additional_columns = [col for col in column_defs['additional_columns'].dropna() if col != '']
        
        print(f"✓ Datasets loaded successfully:")
        print(f"  - Training: {len(self.train_df):,} events")
        print(f"  - Validation: {len(self.val_df):,} events") 
        print(f"  - Test: {len(self.test_df):,} events")
        print(f"  - Predictor columns: {len(self.predictor_columns)}")
        print(f"  - Target column: {self.target_column}")
        
        return True
    
    def preprocess_data(self):
        """Preprocess data for machine learning."""
        print("\nStep 2: Preprocessing data for machine learning...")
        
        # Check that datasets are loaded
        if self.train_df is None or self.val_df is None or self.test_df is None:
            raise RuntimeError("Datasets must be loaded first. Call load_datasets() before preprocess_data().")
        
        # Combine all datasets for consistent preprocessing
        all_data = pd.concat([self.train_df, self.val_df, self.test_df], ignore_index=True)
        
        # Identify categorical and numerical columns
        categorical_columns = []
        numerical_columns = []
        
        for col in self.predictor_columns:
            if col in all_data.columns:
                if all_data[col].dtype == 'object' or col in ['activity', 'machine_identifier', 'product_type', 
                                                             'material_extracted', 'standard_extracted', 'work_type',
                                                             'previous_activity', 'previous_machine', 'previous_activity_2', 
                                                             'previous_machine_2', 'following_activity', 'following_machine']:
                    categorical_columns.append(col)
                else:
                    numerical_columns.append(col)
        
        print(f"  - Categorical columns: {len(categorical_columns)}")
        print(f"  - Numerical columns: {len(numerical_columns)}")
        
        # Handle categorical variables with label encoding
        for col in categorical_columns:
            if col in all_data.columns:
                le = LabelEncoder()
                # Convert all values to string first, then handle missing values
                all_data[col] = all_data[col].astype(str).fillna('unknown')
                # Replace 'nan' strings that might result from conversion
                all_data[col] = all_data[col].replace('nan', 'unknown')
                le.fit(all_data[col])
                self.label_encoders[col] = le
                
                # Transform each dataset
                for df in [self.train_df, self.val_df, self.test_df]:
                    if col in df.columns:
                        df[col] = df[col].astype(str).fillna('unknown')
                        df[col] = df[col].replace('nan', 'unknown')
                        df[col] = le.transform(df[col])
        
        # Handle numerical variables - fill missing values with median
        for col in numerical_columns:
            if col in all_data.columns:
                median_val = all_data[col].median()
                for df in [self.train_df, self.val_df, self.test_df]:
                    if col in df.columns:
                        df[col] = df[col].fillna(median_val)
        
        # Remove any remaining columns that have issues
        valid_predictors = []
        for col in self.predictor_columns:
            if col in self.train_df.columns and col in self.val_df.columns and col in self.test_df.columns:
                # Check if column has valid data
                if (not self.train_df[col].isna().all() and 
                    not self.val_df[col].isna().all() and 
                    not self.test_df[col].isna().all()):
                    valid_predictors.append(col)
        
        self.predictor_columns = valid_predictors
        print(f"  ✓ Data preprocessing completed")
        print(f"  - Valid predictor columns: {len(self.predictor_columns)}")
        
        return True
    
    def prepare_ml_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare X and y arrays for machine learning."""
        print("\nStep 3: Preparing data arrays...")
        
        # Extract features and targets, ensuring numeric conversion
        X_train = self.train_df[self.predictor_columns].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(float)
        y_train = pd.to_numeric(self.train_df[self.target_column], errors='coerce').fillna(0).values.astype(float)
        
        X_val = self.val_df[self.predictor_columns].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(float)
        y_val = pd.to_numeric(self.val_df[self.target_column], errors='coerce').fillna(0).values.astype(float)
        
        X_test = self.test_df[self.predictor_columns].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(float)
        y_test = pd.to_numeric(self.test_df[self.target_column], errors='coerce').fillna(0).values.astype(float)
        
        # Final check for any remaining issues
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
        y_val = np.nan_to_num(y_val, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
        y_test = np.nan_to_num(y_test, nan=0.0, posinf=0.0, neginf=0.0)
        
        print(f"  ✓ Data arrays prepared:")
        print(f"    Training: X{X_train.shape}, y{y_train.shape}")
        print(f"    Validation: X{X_val.shape}, y{y_val.shape}")
        print(f"    Test: X{X_test.shape}, y{y_test.shape}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def tune_xgboost_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray, 
                                   X_val: np.ndarray, y_val: np.ndarray) -> xgb.XGBRegressor:
        """Tune XGBoost hyperparameters with comprehensive analysis and sensitivity plots."""
        print("\nStep 4: Tuning XGBoost hyperparameters with sensitivity analysis...")
        
        # Define parameter space for tuning
        param_space = {
            'n_estimators': [100, 200, 300, 500, 800],
            'max_depth': [3, 4, 5, 6, 7, 8],
            'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'min_child_weight': [1, 3, 5, 7],
            'gamma': [0, 0.1, 0.2, 0.3],
            'reg_alpha': [0, 0.1, 0.5, 1.0],
            'reg_lambda': [0, 0.1, 0.5, 1.0]
        }
        
        # Create base model
        base_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
        )
        
        # Custom scoring function for MAE
        def mae_scorer(estimator, X, y):
            y_pred = estimator.predict(X)
            return -mean_absolute_error(y, y_pred)  # Negative because sklearn maximizes
        
        # Randomized search with conservative settings for large datasets
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_space,
            n_iter=100,  # Number of parameter combinations to try
            scoring=mae_scorer,
            cv=3,
            random_state=42,
            n_jobs=2,  # Conservative parallelization for large datasets
            verbose=2,   # Show progress without interfering libraries
            return_train_score=True  # Keep training scores for analysis
        )
        
        print("  - Running randomized search with 100 parameter combinations...")
        print("  - This may take 15-30 minutes with the large dataset...")
        print("  - Progress will be shown in batches...")
        
        # Fit the search
        random_search.fit(X_train, y_train)
        
        # Get best model
        best_model = random_search.best_estimator_
        
        # Comprehensive analysis of all hyperparameter combinations
        print("  - Analyzing all hyperparameter combinations...")
        self.analyze_hyperparameter_results(random_search, X_val, y_val)
        
        # Create sensitivity analysis plots
        print("  - Creating sensitivity analysis plots...")
        self.create_sensitivity_plots(random_search, X_val, y_val)
        
        # Evaluate best model on validation set
        y_val_pred = best_model.predict(X_val)
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        val_r2 = r2_score(y_val, y_val_pred)
        
        print(f"  ✓ Best hyperparameters found:")
        for param, value in random_search.best_params_.items():
            print(f"    {param}: {value}")
        
        print(f"  ✓ Best model validation performance:")
        print(f"    MAE: {val_mae:.3f}")
        print(f"    RMSE: {val_rmse:.3f}")
        print(f"    R²: {val_r2:.3f}")
        
        self.best_model = best_model
        self.hyperparameter_results = random_search
        return best_model
    
    def analyze_hyperparameter_results(self, search_results, X_val: np.ndarray, y_val: np.ndarray):
        """Analyze all hyperparameter combinations and their metrics."""
        
        # Extract results from RandomizedSearchCV
        results_df = pd.DataFrame(search_results.cv_results_)
        
        # Calculate detailed metrics for each hyperparameter combination
        detailed_results = []
        
        for i, params in enumerate(results_df['params']):
            # Create model with these parameters
            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                random_state=42,
                n_jobs=-1,
                **params
            )
            
            # Train on full training set
            model.fit(X_val[:int(len(X_val)*0.8)], y_val[:int(len(X_val)*0.8)])  # Use part of validation for training
            
            # Predict on remaining validation set
            X_test_subset = X_val[int(len(X_val)*0.8):]
            y_test_subset = y_val[int(len(X_val)*0.8):]
            y_pred = model.predict(X_test_subset)
            
            # Calculate conformal prediction intervals (simplified version)
            residuals = np.abs(y_test_subset - y_pred)
            conformal_score = np.quantile(residuals, 0.9)  # 90% prediction intervals
            
            lower_bounds = y_pred - conformal_score
            upper_bounds = y_pred + conformal_score
            
            # Calculate metrics
            mae = mean_absolute_error(y_test_subset, y_pred)
            mpiw = np.mean(upper_bounds - lower_bounds)  # Mean Prediction Interval Width
            picp = np.mean((y_test_subset >= lower_bounds) & (y_test_subset <= upper_bounds))  # Coverage
            
            # Add to results
            result_row = params.copy()
            result_row.update({
                'MAE': mae,
                'MPIW': mpiw,
                'PICP': picp,
                'CV_MAE': -results_df.iloc[i]['mean_test_score'],  # Convert back from negative
                'CV_MAE_std': results_df.iloc[i]['std_test_score']
            })
            detailed_results.append(result_row)
        
        # Create comprehensive results DataFrame
        self.detailed_results_df = pd.DataFrame(detailed_results)
        
        # Save detailed results
        results_file = os.path.join(self.ml_output_folder, "hyperparameter_analysis_detailed.csv")
        self.detailed_results_df.to_csv(results_file, index=False)
        
        # Print summary of best performers
        print(f"  ✓ Detailed analysis complete - {len(detailed_results)} combinations analyzed")
        
        # Hierarchical selection of best hyperparameters
        print(f"  - Applying hierarchical selection criteria...")
        best_params = self.select_best_hyperparameters(self.detailed_results_df)
        
        # Top 5 by MAE
        top_mae = self.detailed_results_df.nsmallest(5, 'MAE')
        print(f"  - Top 5 combinations by MAE:")
        for i, (_, row) in enumerate(top_mae.iterrows(), 1):
            print(f"    {i}. MAE: {row['MAE']:.3f}, MPIW: {row['MPIW']:.3f}, PICP: {row['PICP']:.3f}")
        
        # Best coverage (closest to 0.9)
        self.detailed_results_df['PICP_diff'] = np.abs(self.detailed_results_df['PICP'] - 0.9)
        top_coverage = self.detailed_results_df.nsmallest(3, 'PICP_diff')
        print(f"  - Top 3 combinations by coverage (target: 0.9):")
        for i, (_, row) in enumerate(top_coverage.iterrows(), 1):
            print(f"    {i}. PICP: {row['PICP']:.3f}, MAE: {row['MAE']:.3f}, MPIW: {row['MPIW']:.3f}")
        
        return self.detailed_results_df
    
    def select_best_hyperparameters(self, detailed_results_df: pd.DataFrame) -> Dict[str, Any]:
        """Select best hyperparameters using hierarchical criteria."""
        
        print("    - Applying hierarchical selection criteria:")
        print("      1. Lowest MAE first")
        print("      2. Then lowest MPIW") 
        print("      3. Then lowest CV_MAE")
        print("      4. Then lowest CV_MAE_std")
        
        # Create a copy for sorting
        df_sorted = detailed_results_df.copy()
        
        # Sort by hierarchical criteria: MAE (asc), MPIW (asc), CV_MAE (asc), CV_MAE_std (asc)
        df_sorted = df_sorted.sort_values([
            'MAE',           # Primary: Lowest MAE
            'MPIW',          # Secondary: Lowest MPIW  
            'CV_MAE',        # Tertiary: Lowest CV_MAE
            'CV_MAE_std'     # Quaternary: Lowest CV_MAE_std
        ], ascending=[True, True, True, True])
        
        # Get the best combination (first row after sorting)
        best_row = df_sorted.iloc[0]
        
        # Extract hyperparameters (exclude metric columns)
        metric_columns = ['MAE', 'MPIW', 'PICP', 'CV_MAE', 'CV_MAE_std', 'PICP_diff']
        param_columns = [col for col in df_sorted.columns if col not in metric_columns]
        
        best_hyperparams = {}
        for col in param_columns:
            best_hyperparams[col] = best_row[col]
        
        # Store the best parameters and their performance
        self.best_hyperparams_hierarchical = best_hyperparams
        self.best_performance = {
            'MAE': best_row['MAE'],
            'MPIW': best_row['MPIW'], 
            'PICP': best_row['PICP'],
            'CV_MAE': best_row['CV_MAE'],
            'CV_MAE_std': best_row['CV_MAE_std']
        }
        
        print(f"    ✓ Best hyperparameters selected (hierarchical):")
        for param, value in best_hyperparams.items():
            print(f"      {param}: {value}")
        
        print(f"    ✓ Performance of best combination:")
        print(f"      MAE: {best_row['MAE']:.3f}")
        print(f"      MPIW: {best_row['MPIW']:.3f}")
        print(f"      PICP: {best_row['PICP']:.3f}")
        print(f"      CV_MAE: {best_row['CV_MAE']:.3f}")
        print(f"      CV_MAE_std: {best_row['CV_MAE_std']:.3f}")
        
        # Save best hyperparameters for final model fitting
        best_params_file = os.path.join(self.ml_output_folder, "best_hyperparameters_hierarchical.csv")
        pd.DataFrame([best_hyperparams]).to_csv(best_params_file, index=False)
        
        # Save performance metrics
        performance_file = os.path.join(self.ml_output_folder, "best_hyperparameters_performance.csv")
        pd.DataFrame([self.best_performance]).to_csv(performance_file, index=False)
        
        print(f"    ✓ Best hyperparameters saved for final model: {best_params_file}")
        print(f"    ✓ Performance metrics saved: {performance_file}")
        
        return best_hyperparams
    
    def create_sensitivity_plots(self, search_results, X_val: np.ndarray, y_val: np.ndarray):
        """Create sensitivity analysis plots for each hyperparameter."""
        
        if not hasattr(self, 'detailed_results_df'):
            print("  ⚠ Warning: No detailed results available for sensitivity plots")
            return
        
        # Get all hyperparameters that were tuned
        param_columns = [col for col in self.detailed_results_df.columns 
                        if col not in ['MAE', 'MPIW', 'PICP', 'CV_MAE', 'CV_MAE_std', 'PICP_diff']]
        
        # Create sensitivity plots for each hyperparameter
        n_params = len(param_columns)
        n_cols = 3  # MAE, MPIW, PICP
        
        # Create separate figure for each hyperparameter
        for param in param_columns:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            fig.suptitle(f'Sensitivity Analysis: {param}', fontsize=16, fontweight='bold')
            
            # Get unique values for this parameter
            param_values = sorted(self.detailed_results_df[param].unique())
            
            # Plot MAE
            mae_means = []
            mae_stds = []
            for val in param_values:
                subset = self.detailed_results_df[self.detailed_results_df[param] == val]
                mae_means.append(subset['MAE'].mean())
                mae_stds.append(subset['MAE'].std())
            
            axes[0].errorbar(param_values, mae_means, yerr=mae_stds, 
                           marker='o', capsize=5, capthick=2, linewidth=2)
            axes[0].set_xlabel(param)
            axes[0].set_ylabel('MAE')
            axes[0].set_title('Mean Absolute Error')
            axes[0].grid(True, alpha=0.3)
            
            # Plot MPIW
            mpiw_means = []
            mpiw_stds = []
            for val in param_values:
                subset = self.detailed_results_df[self.detailed_results_df[param] == val]
                mpiw_means.append(subset['MPIW'].mean())
                mpiw_stds.append(subset['MPIW'].std())
            
            axes[1].errorbar(param_values, mpiw_means, yerr=mpiw_stds, 
                           marker='s', capsize=5, capthick=2, linewidth=2, color='orange')
            axes[1].set_xlabel(param)
            axes[1].set_ylabel('MPIW')
            axes[1].set_title('Mean Prediction Interval Width')
            axes[1].grid(True, alpha=0.3)
            
            # Plot PICP
            picp_means = []
            picp_stds = []
            for val in param_values:
                subset = self.detailed_results_df[self.detailed_results_df[param] == val]
                picp_means.append(subset['PICP'].mean())
                picp_stds.append(subset['PICP'].std())
            
            axes[2].errorbar(param_values, picp_means, yerr=picp_stds, 
                           marker='^', capsize=5, capthick=2, linewidth=2, color='green')
            axes[2].axhline(y=0.9, color='red', linestyle='--', alpha=0.7, label='Target (0.9)')
            axes[2].set_xlabel(param)
            axes[2].set_ylabel('PICP')
            axes[2].set_title('Prediction Interval Coverage Probability')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plot_file = os.path.join(self.ml_output_folder, f"sensitivity_analysis_{param}.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()  # Close to save memory
            
            print(f"    - Sensitivity plot saved: {param}")
        
        # Create overall correlation heatmap
        self.create_hyperparameter_correlation_plot()
    
    def create_hyperparameter_correlation_plot(self):
        """Create correlation heatmap between hyperparameters and metrics."""
        
        # Prepare data for correlation analysis
        numeric_cols = []
        for col in self.detailed_results_df.columns:
            if self.detailed_results_df[col].dtype in ['int64', 'float64']:
                numeric_cols.append(col)
        
        corr_data = self.detailed_results_df[numeric_cols].copy()
        
        # Calculate correlation matrix
        correlation_matrix = corr_data.corr()
        
        # Focus on correlations with metrics
        metrics_corr = correlation_matrix[['MAE', 'MPIW', 'PICP']].drop(['MAE', 'MPIW', 'PICP'])
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(metrics_corr, annot=True, cmap='RdBu_r', center=0, 
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Hyperparameter-Metric Correlations', fontsize=14, fontweight='bold')
        plt.xlabel('Metrics')
        plt.ylabel('Hyperparameters')
        plt.tight_layout()
        
        # Save correlation plot
        corr_file = os.path.join(self.ml_output_folder, "hyperparameter_correlations.png")
        plt.savefig(corr_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Correlation analysis saved: {corr_file}")
        
        # Print strongest correlations
        print(f"  - Strongest correlations with metrics:")
        for metric in ['MAE', 'MPIW', 'PICP']:
            abs_corr = metrics_corr[metric].abs().sort_values(ascending=False)
            top_param = abs_corr.index[0]
            corr_val = metrics_corr.loc[top_param, metric]
            print(f"    {metric}: {top_param} (r = {corr_val:.3f})")
    
    def implement_split_conformal_prediction(self, X_train: np.ndarray, y_train: np.ndarray,
                                           X_val: np.ndarray, y_val: np.ndarray) -> np.ndarray:
        """Implement split conformal prediction using validation set as calibration set."""
        print("\nStep 5: Implementing split conformal prediction...")
        
        # Train model on training set (already done in hyperparameter tuning)
        # Use validation set as calibration set
        
        # Get predictions on calibration (validation) set
        y_val_pred = self.best_model.predict(X_val)
        
        # Calculate nonconformity scores (absolute residuals)
        nonconformity_scores = np.abs(y_val - y_val_pred)
        
        # Calculate quantile for prediction intervals
        # For alpha=0.1 (90% coverage), we need the (1-alpha)*(n+1)/n quantile
        n_cal = len(nonconformity_scores)
        quantile_level = (1 - self.alpha) * (n_cal + 1) / n_cal
        
        # Handle case where quantile_level > 1
        if quantile_level > 1.0:
            quantile_level = 1.0
        
        # Get the conformal score threshold
        conformal_score = np.quantile(nonconformity_scores, quantile_level)
        
        print(f"  ✓ Conformal prediction calibrated:")
        print(f"    Calibration set size: {n_cal}")
        print(f"    Alpha (miscoverage rate): {self.alpha}")
        print(f"    Target coverage: {(1-self.alpha)*100:.1f}%")
        print(f"    Conformal score threshold: {conformal_score:.3f}")
        
        self.conformal_scores = nonconformity_scores
        return conformal_score
    
    def make_conformal_predictions(self, X_test: np.ndarray, conformal_score: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Make predictions with conformal prediction intervals on test set."""
        print("\nStep 6: Making predictions with conformal intervals...")
        
        # Get point predictions
        y_test_pred = self.best_model.predict(X_test)
        
        # Create prediction intervals
        lower_bounds = y_test_pred - conformal_score
        upper_bounds = y_test_pred + conformal_score
        
        print(f"  ✓ Predictions generated for {len(X_test)} test samples")
        print(f"    Mean prediction interval width: {conformal_score * 2:.3f}")
        
        return y_test_pred, lower_bounds, upper_bounds
    
    def evaluate_predictions(self, y_test: np.ndarray, y_test_pred: np.ndarray, 
                           lower_bounds: np.ndarray, upper_bounds: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance with comprehensive metrics."""
        print("\nStep 7: Evaluating model performance...")
        
        # Point prediction metrics
        mae = mean_absolute_error(y_test, y_test_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        r2 = r2_score(y_test, y_test_pred)
        
        # Prediction interval metrics
        # Coverage: proportion of true values within prediction intervals
        coverage = np.mean((y_test >= lower_bounds) & (y_test <= upper_bounds))
        
        # MPIW: Mean Prediction Interval Width
        interval_widths = upper_bounds - lower_bounds
        mpiw = np.mean(interval_widths)
        
        # MRPIW: Mean Relative Prediction Interval Width
        # Relative to the absolute value of true targets
        mrpiw = np.mean(interval_widths / np.abs(y_test))
        
        # Additional metrics
        median_width = np.median(interval_widths)
        std_width = np.std(interval_widths)
        
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'Coverage': coverage,
            'MPIW': mpiw,
            'MRPIW': mrpiw,
            'Median_Interval_Width': median_width,
            'Std_Interval_Width': std_width,
            'Target_Coverage': 1 - self.alpha,
            'Coverage_Gap': abs(coverage - (1 - self.alpha))
        }
        
        print(f"  ✓ Evaluation metrics computed:")
        print(f"    Point Prediction Performance:")
        print(f"      MAE: {mae:.3f}")
        print(f"      RMSE: {rmse:.3f}")
        print(f"      R²: {r2:.3f}")
        print(f"    Prediction Interval Performance:")
        print(f"      Coverage: {coverage:.3f} (target: {1-self.alpha:.3f})")
        print(f"      Coverage Gap: {abs(coverage - (1-self.alpha)):.3f}")
        print(f"      MPIW: {mpiw:.3f}")
        print(f"      MRPIW: {mrpiw:.3f}")
        print(f"      Median Width: {median_width:.3f}")
        
        return metrics
    
    def create_visualizations(self, y_test: np.ndarray, y_test_pred: np.ndarray,
                            lower_bounds: np.ndarray, upper_bounds: np.ndarray, metrics: Dict[str, float]):
        """Create comprehensive visualizations of model performance."""
        print("\nStep 8: Creating visualizations...")
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('BDE Processing Time Prediction Results', fontsize=16, fontweight='bold')
        
        # 1. Actual vs Predicted scatter plot
        ax1 = axes[0, 0]
        ax1.scatter(y_test, y_test_pred, alpha=0.6, s=20)
        min_val = min(y_test.min(), y_test_pred.min())
        max_val = max(y_test.max(), y_test_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        ax1.set_xlabel('Actual Processing Time (minutes)')
        ax1.set_ylabel('Predicted Processing Time (minutes)')
        ax1.set_title(f'Actual vs Predicted\nMAE: {metrics["MAE"]:.3f}, R²: {metrics["R2"]:.3f}')
        ax1.grid(True, alpha=0.3)
        
        # 2. Residuals plot
        ax2 = axes[0, 1]
        residuals = y_test - y_test_pred
        ax2.scatter(y_test_pred, residuals, alpha=0.6, s=20)
        ax2.axhline(y=0, color='r', linestyle='--', lw=2)
        ax2.set_xlabel('Predicted Processing Time (minutes)')
        ax2.set_ylabel('Residuals (minutes)')
        ax2.set_title('Residuals Plot')
        ax2.grid(True, alpha=0.3)
        
        # 3. Prediction intervals visualization (sample)
        ax3 = axes[1, 0]
        sample_size = min(100, len(y_test))
        indices = np.random.choice(len(y_test), sample_size, replace=False)
        sample_indices = np.sort(indices)
        
        x_pos = np.arange(len(sample_indices))
        ax3.fill_between(x_pos, lower_bounds[sample_indices], upper_bounds[sample_indices], 
                        alpha=0.3, label=f'{(1-self.alpha)*100:.0f}% Prediction Intervals')
        ax3.scatter(x_pos, y_test[sample_indices], color='red', s=20, label='Actual', alpha=0.7)
        ax3.scatter(x_pos, y_test_pred[sample_indices], color='blue', s=20, label='Predicted', alpha=0.7)
        ax3.set_xlabel('Sample Index')
        ax3.set_ylabel('Processing Time (minutes)')
        ax3.set_title(f'Prediction Intervals (Random Sample of {sample_size})\nCoverage: {metrics["Coverage"]:.3f}')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Coverage and interval width distribution
        ax4 = axes[1, 1]
        interval_widths = upper_bounds - lower_bounds
        ax4.hist(interval_widths, bins=30, alpha=0.7, edgecolor='black')
        ax4.axvline(metrics['MPIW'], color='red', linestyle='--', lw=2, label=f'Mean: {metrics["MPIW"]:.3f}')
        ax4.axvline(metrics['Median_Interval_Width'], color='orange', linestyle='--', lw=2, 
                   label=f'Median: {metrics["Median_Interval_Width"]:.3f}')
        ax4.set_xlabel('Prediction Interval Width (minutes)')
        ax4.set_ylabel('Frequency')
        ax4.set_title(f'Interval Width Distribution\nMRPIW: {metrics["MRPIW"]:.3f}')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(self.ml_output_folder, "bde_ml_prediction_results.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"  ✓ Visualizations saved: {plot_file}")
    
    def save_results(self, metrics: Dict[str, float], y_test: np.ndarray, y_test_pred: np.ndarray,
                    lower_bounds: np.ndarray, upper_bounds: np.ndarray):
        """Save model, predictions, and results."""
        print("\nStep 9: Saving results...")
        
        # Save model
        model_file = os.path.join(self.ml_output_folder, "bde_xgboost_model.joblib")
        joblib.dump(self.best_model, model_file)
        
        # Save label encoders
        encoders_file = os.path.join(self.ml_output_folder, "label_encoders.joblib")
        joblib.dump(self.label_encoders, encoders_file)
        
        # Save predictions
        predictions_df = pd.DataFrame({
            'actual': y_test,
            'predicted': y_test_pred,
            'lower_bound': lower_bounds,
            'upper_bound': upper_bounds,
            'interval_width': upper_bounds - lower_bounds,
            'covered': (y_test >= lower_bounds) & (y_test <= upper_bounds),
            'residual': y_test - y_test_pred,
            'absolute_residual': np.abs(y_test - y_test_pred)
        })
        
        predictions_file = os.path.join(self.ml_output_folder, "bde_test_predictions.csv")
        predictions_df.to_csv(predictions_file, index=False)
        
        # Save metrics
        metrics_df = pd.DataFrame([metrics]).T
        metrics_df.columns = ['Value']
        metrics_file = os.path.join(self.ml_output_folder, "bde_evaluation_metrics.csv")
        metrics_df.to_csv(metrics_file)
        
        # Save feature importance
        if hasattr(self.best_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.predictor_columns,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            importance_file = os.path.join(self.ml_output_folder, "bde_feature_importance.csv")
            importance_df.to_csv(importance_file, index=False)
            
            print(f"  - Top 10 most important features:")
            for i, row in importance_df.head(10).iterrows():
                print(f"    {row['feature']}: {row['importance']:.4f}")
        
        # Save comprehensive hyperparameter analysis summary
        if hasattr(self, 'detailed_results_df'):
            # Create summary of hyperparameter analysis
            summary_stats = {
                'Total_Combinations_Tested': len(self.detailed_results_df),
                'Best_MAE': self.detailed_results_df['MAE'].min(),
                'Worst_MAE': self.detailed_results_df['MAE'].max(),
                'Mean_MAE': self.detailed_results_df['MAE'].mean(),
                'Std_MAE': self.detailed_results_df['MAE'].std(),
                'Best_MPIW': self.detailed_results_df['MPIW'].min(),
                'Worst_MPIW': self.detailed_results_df['MPIW'].max(),
                'Mean_MPIW': self.detailed_results_df['MPIW'].mean(),
                'Std_MPIW': self.detailed_results_df['MPIW'].std(),
                'Best_PICP': self.detailed_results_df['PICP'].max(),
                'Worst_PICP': self.detailed_results_df['PICP'].min(),
                'Mean_PICP': self.detailed_results_df['PICP'].mean(),
                'Std_PICP': self.detailed_results_df['PICP'].std(),
                'Target_Coverage': 0.9,
                'Coverage_Within_5pct': ((self.detailed_results_df['PICP'] >= 0.85) & 
                                        (self.detailed_results_df['PICP'] <= 0.95)).sum(),
                'Coverage_Within_2pct': ((self.detailed_results_df['PICP'] >= 0.88) & 
                                        (self.detailed_results_df['PICP'] <= 0.92)).sum()
            }
            
            # Add hierarchical best parameters info if available
            if hasattr(self, 'best_performance'):
                summary_stats.update({
                    'Hierarchical_Best_MAE': self.best_performance['MAE'],
                    'Hierarchical_Best_MPIW': self.best_performance['MPIW'],
                    'Hierarchical_Best_PICP': self.best_performance['PICP'],
                    'Hierarchical_Best_CV_MAE': self.best_performance['CV_MAE'],
                    'Hierarchical_Best_CV_MAE_std': self.best_performance['CV_MAE_std']
                })
            
            summary_file = os.path.join(self.ml_output_folder, "hyperparameter_analysis_summary.csv")
            pd.DataFrame([summary_stats]).T.to_csv(summary_file, header=['Value'])
            
            print(f"  - Comprehensive hyperparameter analysis summary:")
            print(f"    Total combinations tested: {summary_stats['Total_Combinations_Tested']}")
            print(f"    MAE range: [{summary_stats['Best_MAE']:.3f}, {summary_stats['Worst_MAE']:.3f}]")
            print(f"    MPIW range: [{summary_stats['Best_MPIW']:.3f}, {summary_stats['Worst_MPIW']:.3f}]")
            print(f"    PICP range: [{summary_stats['Worst_PICP']:.3f}, {summary_stats['Best_PICP']:.3f}]")
            print(f"    Combinations within 5% of target coverage: {summary_stats['Coverage_Within_5pct']}")
            print(f"    Combinations within 2% of target coverage: {summary_stats['Coverage_Within_2pct']}")
            
            if hasattr(self, 'best_performance'):
                print(f"    Hierarchical best MAE: {summary_stats['Hierarchical_Best_MAE']:.3f}")
                print(f"    Hierarchical best PICP: {summary_stats['Hierarchical_Best_PICP']:.3f}")
        
        print(f"  ✓ Results saved:")
        print(f"    Model: {model_file}")
        print(f"    Predictions: {predictions_file}")
        print(f"    Metrics: {metrics_file}")
        print(f"    Label Encoders: {encoders_file}")
        if hasattr(self.best_model, 'feature_importances_'):
            print(f"    Feature Importance: {importance_file}")
        if hasattr(self, 'detailed_results_df'):
            print(f"    Hyperparameter Analysis: hyperparameter_analysis_detailed.csv")
            print(f"    Analysis Summary: hyperparameter_analysis_summary.csv")
            print(f"    Best Hyperparameters: best_hyperparameters_hierarchical.csv")
            print(f"    Best Performance: best_hyperparameters_performance.csv")
            print(f"    Sensitivity Plots: sensitivity_analysis_*.png")
            print(f"    Correlation Heatmap: hyperparameter_correlations.png")
    
    def run_complete_ml_pipeline(self):
        """Run the complete machine learning pipeline."""
        try:
            # Load datasets
            self.load_datasets()
            
            # Preprocess data
            self.preprocess_data()
            
            # Prepare ML data
            X_train, y_train, X_val, y_val, X_test, y_test = self.prepare_ml_data()
            
            # Tune hyperparameters
            self.tune_xgboost_hyperparameters(X_train, y_train, X_val, y_val)
            
            # Implement conformal prediction
            conformal_score = self.implement_split_conformal_prediction(X_train, y_train, X_val, y_val)
            
            # Make predictions with intervals
            y_test_pred, lower_bounds, upper_bounds = self.make_conformal_predictions(X_test, conformal_score)
            
            # Evaluate predictions
            metrics = self.evaluate_predictions(y_test, y_test_pred, lower_bounds, upper_bounds)
            
            # Create visualizations
            self.create_visualizations(y_test, y_test_pred, lower_bounds, upper_bounds, metrics)
            
            # Save results
            self.save_results(metrics, y_test, y_test_pred, lower_bounds, upper_bounds)
            
            # Print final summary
            self.print_final_summary(metrics)
            
            return metrics
            
        except Exception as e:
            print(f"❌ Error in ML pipeline: {str(e)}")
            raise
    
    def print_final_summary(self, metrics: Dict[str, float]):
        """Print final summary of results."""
        print("\n" + "=" * 70)
        print("BDE MACHINE LEARNING PIPELINE SUMMARY")
        print("=" * 70)
        
        print("✓ XGBoost hyperparameter optimization completed (100 combinations)")
        print("✓ Hierarchical hyperparameter selection implemented")
        print("✓ Split conformal prediction implemented for uncertainty quantification")
        print("✓ Comprehensive sensitivity analysis completed")
        print("✓ Best hyperparameters identified and saved for final model")
        
        print(f"\nFinal Performance Metrics (Test Set):")
        print(f"  Point Predictions:")
        print(f"    • Mean Absolute Error (MAE): {metrics['MAE']:.3f} minutes")
        print(f"    • Root Mean Square Error (RMSE): {metrics['RMSE']:.3f} minutes")
        print(f"    • R-squared (R²): {metrics['R2']:.3f}")
        
        print(f"  Prediction Intervals:")
        print(f"    • Coverage: {metrics['Coverage']:.1%} (target: {metrics['Target_Coverage']:.1%})")
        print(f"    • Coverage Gap: {metrics['Coverage_Gap']:.3f}")
        print(f"    • Mean Prediction Interval Width (MPIW): {metrics['MPIW']:.3f} minutes")
        print(f"    • Mean Relative Prediction Interval Width (MRPIW): {metrics['MRPIW']:.3f}")
        
        # Show hierarchical best performance if available
        if hasattr(self, 'best_performance'):
            print(f"\nHierarchical Best Hyperparameters Performance:")
            print(f"    • Optimized MAE: {self.best_performance['MAE']:.3f} minutes")
            print(f"    • Optimized MPIW: {self.best_performance['MPIW']:.3f} minutes")
            print(f"    • Optimized PICP: {self.best_performance['PICP']:.3f}")
            print(f"    • Cross-Validation MAE: {self.best_performance['CV_MAE']:.3f} minutes")
        
        print(f"\nReady for Final Model Fitting:")
        print(f"  ✓ Datasets prepared and validated")
        print(f"  ✓ Best hyperparameters selected hierarchically")
        print(f"  ✓ Sensitivity analysis completed")
        print(f"  ✓ All artifacts saved for final model implementation")
        
        print(f"\nOutput Location: {self.ml_output_folder}")
        print("🎉 Enhanced ML pipeline with hyperparameter optimization completed successfully!")
        print("   Ready to proceed with final model fitting implementation.")


def main():
    """Main function to run the complete ML pipeline."""
    try:
        # Initialize and run the ML pipeline
        ml_pipeline = BDEMLPipeline()
        metrics = ml_pipeline.run_complete_ml_pipeline()
        
        return metrics
        
    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 