import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

class MissingValueHandler:
    def __init__(self, data):
        """
        Initialize MissingValueHandler with data
        
        Args:
            data: pandas DataFrame or path to CSV file
        """
        if isinstance(data, str):
            self.df = pd.read_csv(data)
        else:
            self.df = data.copy()
        
        self.original_shape = self.df.shape
        self.missing_info = {}
        
    def detect_missing_values(self):
        """
        Detect missing values in the dataset
        Returns information about missing values
        """
        # Replace '?' with NaN for proper missing value detection
        self.df = self.df.replace('?', np.nan)
        
        # Calculate missing value statistics
        missing_count = self.df.isnull().sum()
        missing_percentage = (self.df.isnull().sum() / len(self.df)) * 100
        
        self.missing_info = pd.DataFrame({
            'Column': self.df.columns,
            'Missing_Count': missing_count.values,
            'Missing_Percentage': missing_percentage.values
        })
        
        # Filter only columns with missing values
        self.missing_info = self.missing_info[self.missing_info['Missing_Count'] > 0]
        self.missing_info = self.missing_info.sort_values('Missing_Percentage', ascending=False)
        
        return self.missing_info
    
    def visualize_missing_values(self, output_type='terminal'):
        """
        Visualize missing values pattern
        
        Args:
            output_type: 'terminal' or 'web' for different output formats
        """
        if output_type == 'terminal':
            # Text-based visualization for terminal
            print("=" * 60)
            print("MISSING VALUES ANALYSIS")
            print("=" * 60)
            print(f"Dataset Shape: {self.original_shape}")
            print(f"Total Missing Values: {self.df.isnull().sum().sum()}")
            print("\nMissing Values by Column:")
            print("-" * 60)
            
            if len(self.missing_info) > 0:
                for _, row in self.missing_info.iterrows():
                    print(f"{row['Column']:<35} | {row['Missing_Count']:>6} | {row['Missing_Percentage']:>6.2f}%")
            else:
                print("No missing values found!")
                
        elif output_type == 'web':
            # Create visualization plots
            if len(self.missing_info) > 0:
                # Missing values heatmap
                plt.figure(figsize=(12, 8))
                
                # Create subplot for missing values heatmap
                plt.subplot(2, 1, 1)
                missing_matrix = self.df.isnull()
                sns.heatmap(missing_matrix, yticklabels=False, cbar=True, cmap='viridis')
                plt.title('Missing Values Heatmap')
                plt.xlabel('Columns')
                
                # Bar plot for missing values percentage
                plt.subplot(2, 1, 2)
                plt.barh(self.missing_info['Column'], self.missing_info['Missing_Percentage'])
                plt.xlabel('Missing Percentage (%)')
                plt.title('Missing Values Percentage by Column')
                plt.gca().invert_yaxis()
                
                plt.tight_layout()
                plt.savefig('missing_values_analysis.png', dpi=300, bbox_inches='tight')
                plt.show()
            else:
                print("No missing values to visualize!")
    
    def handle_missing_values(self, strategy='median', threshold=50):
        """
        Handle missing values using specified strategy
        
        Args:
            strategy: 'median', 'mean', 'mode', 'drop'
            threshold: percentage threshold for dropping columns (only used with 'drop' strategy)
        """
        print(f"\nHandling missing values using '{strategy}' strategy...")
        
        if strategy == 'drop':
            # Drop columns with missing percentage above threshold
            columns_to_drop = self.missing_info[
                self.missing_info['Missing_Percentage'] > threshold
            ]['Column'].tolist()
            
            if columns_to_drop:
                print(f"Dropping columns with >{threshold}% missing values: {columns_to_drop}")
                self.df = self.df.drop(columns=columns_to_drop)
            
            # Drop rows with any remaining missing values
            before_rows = len(self.df)
            self.df = self.df.dropna()
            after_rows = len(self.df)
            print(f"Dropped {before_rows - after_rows} rows with missing values")
            
        else:
            # Fill missing values
            for col in self.df.columns:
                if self.df[col].isnull().sum() > 0:
                    if self.df[col].dtype in ['int64', 'float64']:
                        # Numerical columns
                        if strategy == 'median':
                            fill_value = self.df[col].median()
                        elif strategy == 'mean':
                            fill_value = self.df[col].mean()
                        else:
                            fill_value = self.df[col].mode()[0] if not self.df[col].mode().empty else 0
                        
                        self.df[col] = self.df[col].fillna(fill_value)
                        print(f"Filled {col} with {strategy}: {fill_value}")
                    else:
                        # Categorical columns
                        mode_value = self.df[col].mode()[0] if not self.df[col].mode().empty else 'Unknown'
                        self.df[col] = self.df[col].fillna(mode_value)
                        print(f"Filled {col} with mode: {mode_value}")
        
        # Convert data types
        self.convert_data_types()
        
        return self.df
    
    def convert_data_types(self):
        """
        Convert object columns to appropriate numeric types where possible
        """
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                try:
                    # Try to convert to numeric
                    self.df[col] = pd.to_numeric(self.df[col], errors='ignore')
                except:
                    pass
    
    def get_summary(self):
        """
        Get summary of missing value handling results
        """
        current_shape = self.df.shape
        
        summary = {
            'original_shape': self.original_shape,
            'current_shape': current_shape,
            'rows_removed': self.original_shape[0] - current_shape[0],
            'columns_removed': self.original_shape[1] - current_shape[1],
            'remaining_missing_values': self.df.isnull().sum().sum()
        }
        
        return summary

# Example usage function
def process_missing_values(data_path, output_type='terminal', strategy='median'):
    """
    Main function to process missing values
    
    Args:
        data_path: path to CSV file
        output_type: 'terminal' or 'web'
        strategy: missing value handling strategy
    """
    print("Starting Missing Values Analysis...")
    
    # Initialize handler
    mv_handler = MissingValueHandler(data_path)
    
    # Detect missing values
    missing_info = mv_handler.detect_missing_values()
    
    # Visualize missing values
    mv_handler.visualize_missing_values(output_type)
    
    # Handle missing values
    cleaned_data = mv_handler.handle_missing_values(strategy)
    
    # Get summary
    summary = mv_handler.get_summary()
    
    print("\n" + "=" * 60)
    print("MISSING VALUES HANDLING SUMMARY")
    print("=" * 60)
    print(f"Original shape: {summary['original_shape']}")
    print(f"Current shape: {summary['current_shape']}")
    print(f"Rows removed: {summary['rows_removed']}")
    print(f"Columns removed: {summary['columns_removed']}")
    print(f"Remaining missing values: {summary['remaining_missing_values']}")
    
    return cleaned_data, summary

if __name__ == "__main__":
    # Test the missing value handler
    data_path = "../risk_factors_cervical_cancer.csv"
    cleaned_data, summary = process_missing_values(data_path, output_type='terminal', strategy='median')
    print("\nMissing value processing completed!")
