import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif

class FeatureSelector:
    def __init__(self, data):
        """
        Initialize FeatureSelector with data
        
        Args:
            data: pandas DataFrame or path to CSV
        """
        if isinstance(data, str):
            self.df = pd.read_csv(data)
        else:
            self.df = data.copy()
        self.feature_columns = []
        self.target_columns = []
        self.selected_features = None

    def identify_features_and_targets(self):
        """
        Identify feature and target columns
        """
        potential_targets = ['Hinselmann', 'Schiller', 'Citology', 'Biopsy', 
                             'Dx:Cancer', 'Dx:CIN', 'Dx:HPV', 'Dx']
        self.target_columns = [col for col in potential_targets if col in self.df.columns]
        self.feature_columns = [col for col in self.df.columns 
                               if col not in self.target_columns and 
                               self.df[col].dtype in ['int64', 'float64']]
        print(f"Feature columns ({len(self.feature_columns)}): {self.feature_columns}")
        print(f"Target columns ({len(self.target_columns)}): {self.target_columns}")
        return self.feature_columns, self.target_columns

    def select_features_anova(self, target_col=None, k=10):
        """
        Select top k features using ANOVA (f_classif)
        
        Args:
            target_col: target column name (if None, uses first target column)
            k: number of top features to select
        """
        if not self.target_columns:
            print("No target columns identified.")
            return None
        if target_col is None:
            target_col = self.target_columns[0]
        X = self.df[self.feature_columns]
        y = self.df[target_col]
        selector = SelectKBest(score_func=f_classif, k=k)
        selector.fit(X, y)
        feature_scores = pd.DataFrame({
            'Feature': self.feature_columns,
            'Score': selector.scores_
        }).sort_values('Score', ascending=False)
        self.selected_features = feature_scores.head(k)
        print(f"\nTop {k} features by ANOVA (f_classif):")
        print("-" * 50)
        for _, row in self.selected_features.iterrows():
            print(f"{row['Feature']:<35} | Score: {row['Score']:>8.4f}")
        return self.selected_features

# Main processing function
def process_feature_selection(data, k=10):
    """
    Main function to process feature selection using ANOVA
    
    Args:
        data: pandas DataFrame or path to CSV
        k: number of top features to select
    """
    print("Starting Feature Selection (ANOVA)...")
    selector = FeatureSelector(data)
    feature_cols, target_cols = selector.identify_features_and_targets()
    if target_cols:
        selected_features = selector.select_features_anova(k=k)
    else:
        selected_features = None
    print("\nFeature selection completed!")
    return selected_features, selector

if __name__ == "__main__":
    # Test the feature selector
    data_path = "../risk_factors_cervical_cancer.csv"
    selected_features, selector = process_feature_selection(
        data_path, k=10
    )