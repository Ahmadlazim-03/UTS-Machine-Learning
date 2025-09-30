import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split

class DataTransformer:
    def __init__(self, data):
        """
        Initialize DataTransformer with data
        
        Args:
            data: pandas DataFrame
        """
        if isinstance(data, str):
            self.df = pd.read_csv(data)
        else:
            self.df = data.copy()
        
        self.scaler = None
        self.feature_columns = []
        self.target_columns = []
        self.scaled_data = None
        
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
    
    def analyze_data_distribution(self, output_type='terminal'):
        """
        Analyze data distribution before transformation
        
        Args:
            output_type: 'terminal' or 'web' for different output formats
        """
        if output_type == 'terminal':
            print("=" * 60)
            print("DATA DISTRIBUTION ANALYSIS (BEFORE TRANSFORMATION)")
            print("=" * 60)
            
            print("\nFeature Statistics:")
            print("-" * 60)
            stats = self.df[self.feature_columns].describe()
            print(stats)
            
            print("\nData Ranges:")
            print("-" * 60)
            for col in self.feature_columns:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                range_val = max_val - min_val
                print(f"{col:<35} | Min: {min_val:>8.2f} | Max: {max_val:>8.2f} | Range: {range_val:>8.2f}")
                
        elif output_type == 'web':
            # Create distribution plots
            n_features = len(self.feature_columns)
            n_cols = 4
            n_rows = (n_features + n_cols - 1) // n_cols
            
            plt.figure(figsize=(16, 4 * n_rows))
            
            for i, col in enumerate(self.feature_columns[:16]):  # Limit to first 16 features
                plt.subplot(n_rows, n_cols, i + 1)
                plt.hist(self.df[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
                plt.title(f'{col}')
                plt.xlabel('Value')
                plt.ylabel('Frequency')
            
            plt.tight_layout()
            plt.savefig('data_distribution_before.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def apply_transformation(self, scaler_type='minmax', feature_range=(0, 1)):
        """
        Apply transformation to the data
        
        Args:
            scaler_type: 'minmax', 'standard', or 'robust'
            feature_range: range for MinMaxScaler (only used for minmax)
        """
        print(f"\nApplying {scaler_type.upper()} transformation...")
        
        # Initialize scaler based on type
        if scaler_type == 'minmax':
            self.scaler = MinMaxScaler(feature_range=feature_range)
        elif scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError("scaler_type must be 'minmax', 'standard', or 'robust'")
        
        # Fit and transform the feature data
        X = self.df[self.feature_columns]
        X_scaled = self.scaler.fit_transform(X)
        
        # Create DataFrame with scaled features
        self.scaled_data = self.df.copy()
        self.scaled_data[self.feature_columns] = X_scaled
        
        print(f"Transformation completed using {scaler_type.upper()} scaler")
        
        return self.scaled_data
    
    def analyze_transformed_data(self, output_type='terminal'):
        """
        Analyze data distribution after transformation
        
        Args:
            output_type: 'terminal' or 'web' for different output formats
        """
        if self.scaled_data is None:
            print("No transformed data available. Please apply transformation first.")
            return
        
        if output_type == 'terminal':
            print("\n" + "=" * 60)
            print("DATA DISTRIBUTION ANALYSIS (AFTER TRANSFORMATION)")
            print("=" * 60)
            
            print("\nTransformed Feature Statistics:")
            print("-" * 60)
            stats = self.scaled_data[self.feature_columns].describe()
            print(stats)
            
            print("\nTransformed Data Ranges:")
            print("-" * 60)
            for col in self.feature_columns:
                min_val = self.scaled_data[col].min()
                max_val = self.scaled_data[col].max()
                range_val = max_val - min_val
                print(f"{col:<35} | Min: {min_val:>8.4f} | Max: {max_val:>8.4f} | Range: {range_val:>8.4f}")
                
        elif output_type == 'web':
            # Create before/after comparison plots
            n_features = min(8, len(self.feature_columns))  # Limit to 8 features for clarity
            
            fig, axes = plt.subplots(2, n_features, figsize=(20, 8))
            
            for i, col in enumerate(self.feature_columns[:n_features]):
                # Before transformation
                axes[0, i].hist(self.df[col].dropna(), bins=30, alpha=0.7, color='blue', edgecolor='black')
                axes[0, i].set_title(f'{col} (Original)')
                axes[0, i].set_xlabel('Value')
                axes[0, i].set_ylabel('Frequency')
                
                # After transformation
                axes[1, i].hist(self.scaled_data[col].dropna(), bins=30, alpha=0.7, color='red', edgecolor='black')
                axes[1, i].set_title(f'{col} (Transformed)')
                axes[1, i].set_xlabel('Value')
                axes[1, i].set_ylabel('Frequency')
            
            plt.tight_layout()
            plt.savefig('data_distribution_comparison.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def get_transformation_summary(self):
        """
        Get summary of transformation results
        """
        if self.scaled_data is None:
            return "No transformation applied yet."
        
        summary = {
            'scaler_type': type(self.scaler).__name__,
            'features_transformed': len(self.feature_columns),
            'original_shape': self.df.shape,
            'transformed_shape': self.scaled_data.shape,
            'feature_range_after': {
                'min': self.scaled_data[self.feature_columns].min().min(),
                'max': self.scaled_data[self.feature_columns].max().max()
            }
        }
        
        return summary
    
    def save_transformed_data(self, output_path):
        """
        Save transformed data to CSV file
        
        Args:
            output_path: path to save the transformed data
        """
        if self.scaled_data is None:
            print("No transformed data to save. Please apply transformation first.")
            return
        
        self.scaled_data.to_csv(output_path, index=False)
        print(f"Transformed data saved to: {output_path}")
    
    def split_data(self, test_size=0.2, random_state=42):
        """
        Split the transformed data into training and testing sets
        
        Args:
            test_size: proportion of data for testing
            random_state: random seed for reproducibility
        """
        if self.scaled_data is None:
            print("No transformed data available. Please apply transformation first.")
            return None, None, None, None
        
        if not self.target_columns:
            print("No target columns identified. Cannot split data.")
            return None, None, None, None
        
        X = self.scaled_data[self.feature_columns]
        # Use the first target column for splitting
        y = self.scaled_data[self.target_columns[0]]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Data split completed:")
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Testing set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test

# Main processing function
def process_data_transformation(data, output_type='terminal', scaler_type='minmax'):
    """
    Main function to process data transformation
    
    Args:
        data: pandas DataFrame or path to CSV
        output_type: 'terminal' or 'web'
        scaler_type: 'minmax', 'standard', or 'robust'
    """
    print("Starting Data Transformation...")
    
    # Initialize transformer
    transformer = DataTransformer(data)
    
    # Identify features and targets
    feature_cols, target_cols = transformer.identify_features_and_targets()
    
    # Analyze original data distribution
    transformer.analyze_data_distribution(output_type)
    
    # Apply transformation
    transformed_data = transformer.apply_transformation(scaler_type)
    
    # Analyze transformed data
    transformer.analyze_transformed_data(output_type)
    
    # Get summary
    summary = transformer.get_transformation_summary()
    
    print("\n" + "=" * 60)
    print("DATA TRANSFORMATION SUMMARY")
    print("=" * 60)
    print(f"Scaler used: {summary['scaler_type']}")
    print(f"Features transformed: {summary['features_transformed']}")
    print(f"Data shape: {summary['transformed_shape']}")
    print(f"Feature range after transformation: {summary['feature_range_after']['min']:.4f} to {summary['feature_range_after']['max']:.4f}")
    
    return transformed_data, transformer, summary

if __name__ == "__main__":
    # Test the data transformer
    data_path = "../risk_factors_cervical_cancer.csv"
    transformed_data, transformer, summary = process_data_transformation(
        data_path, output_type='terminal', scaler_type='minmax'
    )
    print("\nData transformation completed!")
