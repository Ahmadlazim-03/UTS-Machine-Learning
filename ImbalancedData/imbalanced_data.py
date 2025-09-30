import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN
from sklearn.model_selection import train_test_split

class ImbalancedDataHandler:
    def __init__(self, data):
        """
        Initialize ImbalancedDataHandler with data
        
        Args:
            data: pandas DataFrame
        """
        if isinstance(data, str):
            self.df = pd.read_csv(data)
        else:
            self.df = data.copy()
        
        self.feature_columns = []
        self.target_columns = []
        self.resampler = None
        self.resampled_data = None
        self.original_distribution = {}
        self.resampled_distribution = {}
        
    def identify_features_and_targets(self):
        """
        Identify feature and target columns
        """
        # Common target columns for cervical cancer dataset
        potential_targets = ['Hinselmann', 'Schiller', 'Citology', 'Biopsy', 
                           'Dx:Cancer', 'Dx:CIN', 'Dx:HPV', 'Dx']
        
        # Identify target columns that exist in the dataset
        self.target_columns = [col for col in potential_targets if col in self.df.columns]
        
        # Feature columns are all other numeric columns
        self.feature_columns = [col for col in self.df.columns 
                              if col not in self.target_columns and 
                              self.df[col].dtype in ['int64', 'float64']]
        
        print(f"Feature columns ({len(self.feature_columns)}): {self.feature_columns}")
        print(f"Target columns ({len(self.target_columns)}): {self.target_columns}")
        
        return self.feature_columns, self.target_columns
    
    def analyze_class_distribution(self, target_col=None, output_type='terminal'):
        """
        Analyze class distribution in the target variable
        
        Args:
            target_col: target column name (if None, uses first target column)
            output_type: 'terminal' or 'web' for different output formats
        """
        if not self.target_columns:
            print("No target columns identified.")
            return None
        
        if target_col is None:
            target_col = self.target_columns[0]
        
        # Calculate class distribution
        distribution = self.df[target_col].value_counts().sort_index()
        percentages = (distribution / len(self.df) * 100).round(2)
        
        self.original_distribution[target_col] = distribution.to_dict()
        
        if output_type == 'terminal':
            print("=" * 60)
            print(f"CLASS DISTRIBUTION ANALYSIS - {target_col}")
            print("=" * 60)
            print(f"Total samples: {len(self.df)}")
            print("\nClass Distribution:")
            print("-" * 40)
            
            for class_label, count in distribution.items():
                percentage = percentages[class_label]
                print(f"Class {class_label}: {count:>6} samples ({percentage:>6.2f}%)")
            
            # Calculate imbalance ratio
            majority_class = distribution.max()
            minority_class = distribution.min()
            imbalance_ratio = majority_class / minority_class
            
            print(f"\nImbalance Ratio: {imbalance_ratio:.2f}:1")
            
            if imbalance_ratio > 1.5:
                print("⚠️  Dataset is imbalanced - resampling recommended")
            else:
                print("✅ Dataset is relatively balanced")
                
        elif output_type == 'web':
            # Create visualization plots
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Bar plot
            axes[0].bar(distribution.index.astype(str), distribution.values, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0].set_xlabel('Class')
            axes[0].set_ylabel('Number of Samples')
            axes[0].set_title(f'Class Distribution - {target_col}')
            
            # Add count labels on bars
            for i, (class_label, count) in enumerate(distribution.items()):
                axes[0].text(i, count + len(self.df)*0.01, str(count), ha='center', va='bottom')
            
            # Pie chart
            axes[1].pie(distribution.values, labels=[f'Class {i}' for i in distribution.index], 
                       autopct='%1.1f%%', startangle=90)
            axes[1].set_title(f'Class Distribution - {target_col}')
            
            plt.tight_layout()
            plt.savefig('class_distribution_original.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        return distribution
    
    def apply_resampling(self, method='ros', target_col=None, random_state=42, sampling_strategy='auto'):
        """
        Apply resampling technique to handle imbalanced data
        
        Args:
            method: 'ros' (Random Over Sampling), 'smote', 'adasyn', 'rus' (Random Under Sampling), 
                   'smote_tomek', 'smote_enn'
            target_col: target column name (if None, uses first target column)
            random_state: random seed for reproducibility
            sampling_strategy: sampling strategy ('auto', 'minority', 'majority', or dict)
        """
        if not self.target_columns:
            print("No target columns identified.")
            return None
        
        if target_col is None:
            target_col = self.target_columns[0]
        
        print(f"Applying {method.upper()} resampling...")
        
        # Prepare data
        X = self.df[self.feature_columns]
        y = self.df[target_col]
        
        # Initialize resampler based on method
        if method == 'ros':
            self.resampler = RandomOverSampler(random_state=random_state, sampling_strategy=sampling_strategy)
        elif method == 'smote':
            self.resampler = SMOTE(random_state=random_state, sampling_strategy=sampling_strategy)
        elif method == 'adasyn':
            self.resampler = ADASYN(random_state=random_state, sampling_strategy=sampling_strategy)
        elif method == 'rus':
            self.resampler = RandomUnderSampler(random_state=random_state, sampling_strategy=sampling_strategy)
        elif method == 'smote_tomek':
            self.resampler = SMOTETomek(random_state=random_state, sampling_strategy=sampling_strategy)
        elif method == 'smote_enn':
            self.resampler = SMOTEENN(random_state=random_state, sampling_strategy=sampling_strategy)
        else:
            raise ValueError("method must be one of: 'ros', 'smote', 'adasyn', 'rus', 'smote_tomek', 'smote_enn'")
        
        # Apply resampling
        try:
            X_resampled, y_resampled = self.resampler.fit_resample(X, y)
            
            # Create resampled dataframe
            resampled_df = pd.DataFrame(X_resampled, columns=self.feature_columns)
            resampled_df[target_col] = y_resampled
            
            # Add other target columns (copy from original based on indices)
            for other_target in self.target_columns:
                if other_target != target_col:
                    # For oversampled data, we need to handle the new synthetic samples
                    if len(y_resampled) > len(y):
                        # Copy original values and fill new samples with mode
                        original_values = self.df[other_target].values
                        mode_value = self.df[other_target].mode()[0] if not self.df[other_target].mode().empty else 0
                        
                        # Create array for resampled target
                        new_target_values = np.full(len(y_resampled), mode_value)
                        new_target_values[:len(original_values)] = original_values
                        
                        resampled_df[other_target] = new_target_values
                    else:
                        # For undersampled data, just use the selected indices
                        resampled_df[other_target] = self.df[other_target].iloc[:len(y_resampled)].values
            
            self.resampled_data = resampled_df
            
            # Store resampled distribution
            resampled_distribution = pd.Series(y_resampled).value_counts().sort_index()
            self.resampled_distribution[target_col] = resampled_distribution.to_dict()
            
            print(f"Resampling completed using {method.upper()}")
            print(f"Original shape: {X.shape}")
            print(f"Resampled shape: {X_resampled.shape}")
            
            return self.resampled_data
            
        except Exception as e:
            print(f"Error during resampling: {str(e)}")
            print("This might occur with highly imbalanced data or insufficient samples for SMOTE/ADASYN")
            return None
    
    def compare_distributions(self, target_col=None, output_type='terminal'):
        """
        Compare original and resampled class distributions
        
        Args:
            target_col: target column name (if None, uses first target column)
            output_type: 'terminal' or 'web' for different output formats
        """
        if target_col is None:
            target_col = self.target_columns[0]
        
        if target_col not in self.original_distribution or target_col not in self.resampled_distribution:
            print("No resampling results to compare. Please apply resampling first.")
            return
        
        original_dist = pd.Series(self.original_distribution[target_col])
        resampled_dist = pd.Series(self.resampled_distribution[target_col])
        
        if output_type == 'terminal':
            print("\n" + "=" * 60)
            print("DISTRIBUTION COMPARISON")
            print("=" * 60)
            
            comparison_df = pd.DataFrame({
                'Original_Count': original_dist,
                'Resampled_Count': resampled_dist,
                'Original_Percentage': (original_dist / original_dist.sum() * 100).round(2),
                'Resampled_Percentage': (resampled_dist / resampled_dist.sum() * 100).round(2)
            }).fillna(0)
            
            print(comparison_df)
            
            print(f"\nTotal samples:")
            print(f"Original: {original_dist.sum()}")
            print(f"Resampled: {resampled_dist.sum()}")
            print(f"Change: {resampled_dist.sum() - original_dist.sum():+d}")
            
        elif output_type == 'web':
            # Create comparison visualization
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Comparison bar plot
            x = np.arange(len(original_dist))
            width = 0.35
            
            axes[0].bar(x - width/2, original_dist.values, width, label='Original', alpha=0.7, color='lightcoral')
            axes[0].bar(x + width/2, resampled_dist.values, width, label='Resampled', alpha=0.7, color='lightblue')
            
            axes[0].set_xlabel('Class')
            axes[0].set_ylabel('Number of Samples')
            axes[0].set_title('Class Distribution Comparison')
            axes[0].set_xticks(x)
            axes[0].set_xticklabels([f'Class {i}' for i in original_dist.index])
            axes[0].legend()
            
            # Add value labels on bars
            for i, (orig, resamp) in enumerate(zip(original_dist.values, resampled_dist.values)):
                axes[0].text(i - width/2, orig + max(original_dist.values)*0.01, str(orig), ha='center', va='bottom')
                axes[0].text(i + width/2, resamp + max(resampled_dist.values)*0.01, str(resamp), ha='center', va='bottom')
            
            # Percentage comparison
            orig_pct = original_dist / original_dist.sum() * 100
            resamp_pct = resampled_dist / resampled_dist.sum() * 100
            
            axes[1].bar(x - width/2, orig_pct.values, width, label='Original', alpha=0.7, color='lightcoral')
            axes[1].bar(x + width/2, resamp_pct.values, width, label='Resampled', alpha=0.7, color='lightblue')
            
            axes[1].set_xlabel('Class')
            axes[1].set_ylabel('Percentage (%)')
            axes[1].set_title('Class Distribution Percentage Comparison')
            axes[1].set_xticks(x)
            axes[1].set_xticklabels([f'Class {i}' for i in original_dist.index])
            axes[1].legend()
            
            plt.tight_layout()
            plt.savefig('class_distribution_comparison.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def get_resampling_summary(self, target_col=None):
        """
        Get summary of resampling results
        """
        if target_col is None:
            target_col = self.target_columns[0]
        
        if self.resampled_data is None:
            return "No resampling applied yet."
        
        original_dist = pd.Series(self.original_distribution[target_col])
        resampled_dist = pd.Series(self.resampled_distribution[target_col])
        
        summary = {
            'resampling_method': type(self.resampler).__name__,
            'original_shape': self.df.shape,
            'resampled_shape': self.resampled_data.shape,
            'original_distribution': self.original_distribution[target_col],
            'resampled_distribution': self.resampled_distribution[target_col],
            'original_imbalance_ratio': original_dist.max() / original_dist.min(),
            'resampled_imbalance_ratio': resampled_dist.max() / resampled_dist.min()
        }
        
        return summary
    
    def save_resampled_data(self, output_path):
        """
        Save resampled data to CSV file
        
        Args:
            output_path: path to save the resampled data
        """
        if self.resampled_data is None:
            print("No resampled data to save. Please apply resampling first.")
            return
        
        self.resampled_data.to_csv(output_path, index=False)
        print(f"Resampled data saved to: {output_path}")
    
    def split_resampled_data(self, target_col=None, test_size=0.2, random_state=42):
        """
        Split the resampled data into training and testing sets
        
        Args:
            target_col: target column name (if None, uses first target column)
            test_size: proportion of data for testing
            random_state: random seed for reproducibility
        """
        if self.resampled_data is None:
            print("No resampled data available. Please apply resampling first.")
            return None, None, None, None
        
        if target_col is None:
            target_col = self.target_columns[0]
        
        X = self.resampled_data[self.feature_columns]
        y = self.resampled_data[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Resampled data split completed:")
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Testing set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test

# Main processing function
def process_imbalanced_data(data, output_type='terminal', method='ros', target_col=None):
    """
    Main function to process imbalanced data
    
    Args:
        data: pandas DataFrame or path to CSV
        output_type: 'terminal' or 'web'
        method: resampling method ('ros', 'smote', 'adasyn', etc.)
        target_col: target column name
    """
    print("Starting Imbalanced Data Processing...")
    
    # Initialize handler
    handler = ImbalancedDataHandler(data)
    
    # Identify features and targets
    feature_cols, target_cols = handler.identify_features_and_targets()
    
    if not target_cols:
        print("No target columns found. Cannot process imbalanced data.")
        return None, None, None
    
    # Analyze original class distribution
    original_dist = handler.analyze_class_distribution(target_col, output_type)
    
    # Apply resampling
    resampled_data = handler.apply_resampling(method, target_col)
    
    if resampled_data is not None:
        # Compare distributions
        handler.compare_distributions(target_col, output_type)
        
        # Get summary
        summary = handler.get_resampling_summary(target_col)
        
        print("\n" + "=" * 60)
        print("IMBALANCED DATA PROCESSING SUMMARY")
        print("=" * 60)
        print(f"Resampling method: {summary['resampling_method']}")
        print(f"Original shape: {summary['original_shape']}")
        print(f"Resampled shape: {summary['resampled_shape']}")
        print(f"Original imbalance ratio: {summary['original_imbalance_ratio']:.2f}:1")
        print(f"Resampled imbalance ratio: {summary['resampled_imbalance_ratio']:.2f}:1")
        
        return resampled_data, handler, summary
    else:
        print("Resampling failed. Returning original data.")
        return data, handler, None

if __name__ == "__main__":
    # Test the imbalanced data handler
    data_path = "../risk_factors_cervical_cancer.csv"
    resampled_data, handler, summary = process_imbalanced_data(
        data_path, output_type='terminal', method='ros'
    )
    print("\nImbalanced data processing completed!")
