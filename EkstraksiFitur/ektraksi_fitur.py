import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif

class FeatureExtractor:
    def __init__(self, data):
        """
        Initialize FeatureExtractor with data
        
        Args:
            data: pandas DataFrame
        """
        if isinstance(data, str):
            self.df = pd.read_csv(data)
        else:
            self.df = data.copy()
        
        self.pca = None
        self.feature_columns = []
        self.target_columns = []
        self.pca_data = None
        self.explained_variance_ratio = None
        
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
    
    def analyze_feature_importance(self, target_col=None, method='f_classif', k=10):
        """
        Analyze feature importance using statistical methods
        
        Args:
            target_col: target column name (if None, uses first target column)
            method: 'f_classif', 'chi2', or 'mutual_info'
            k: number of top features to select
        """
        if not self.target_columns:
            print("No target columns identified.")
            return None
        
        if target_col is None:
            target_col = self.target_columns[0]
        
        X = self.df[self.feature_columns]
        y = self.df[target_col]
        
        # Select appropriate scoring function
        if method == 'f_classif':
            selector = SelectKBest(score_func=f_classif, k=k)
        elif method == 'chi2':
            # Chi2 requires non-negative features
            X = X - X.min() + 1  # Make all values positive
            selector = SelectKBest(score_func=chi2, k=k)
        elif method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
        else:
            raise ValueError("method must be 'f_classif', 'chi2', or 'mutual_info'")
        
        # Fit selector
        X_selected = selector.fit_transform(X, y)
        
        # Get feature scores
        feature_scores = pd.DataFrame({
            'Feature': self.feature_columns,
            'Score': selector.scores_
        }).sort_values('Score', ascending=False)
        
        print(f"\nTop {k} features by {method}:")
        print("-" * 50)
        for _, row in feature_scores.head(k).iterrows():
            print(f"{row['Feature']:<35} | Score: {row['Score']:>8.4f}")
        
        return feature_scores
    
    def apply_pca(self, n_components=None, variance_threshold=0.95):
        """
        Apply PCA for feature extraction
        
        Args:
            n_components: number of components (if None, determined by variance_threshold)
            variance_threshold: cumulative variance threshold for automatic component selection
        """
        print("Applying PCA for feature extraction...")
        
        # Prepare data
        X = self.df[self.feature_columns]
        
        # Standardize features before PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA
        if n_components is None:
            # First, fit PCA with all components to determine optimal number
            pca_temp = PCA()
            pca_temp.fit(X_scaled)
            
            # Find number of components that explain desired variance
            cumsum = np.cumsum(pca_temp.explained_variance_ratio_)
            n_components = np.argmax(cumsum >= variance_threshold) + 1
            
            print(f"Selected {n_components} components to explain {variance_threshold*100}% of variance")
        
        # Apply PCA with selected number of components
        self.pca = PCA(n_components=n_components)
        X_pca = self.pca.fit_transform(X_scaled)
        
        # Store results
        self.explained_variance_ratio = self.pca.explained_variance_ratio_
        
        # Create DataFrame with PCA components
        pca_columns = [f'PC{i+1}' for i in range(n_components)]
        pca_df = pd.DataFrame(X_pca, columns=pca_columns, index=self.df.index)
        
        # Combine PCA features with target columns
        self.pca_data = pd.concat([pca_df, self.df[self.target_columns]], axis=1)
        
        print(f"PCA completed with {n_components} components")
        print(f"Explained variance ratio: {self.explained_variance_ratio}")
        print(f"Total explained variance: {np.sum(self.explained_variance_ratio):.4f}")
        
        return self.pca_data
    
    def visualize_pca_results(self, output_type='terminal'):
        """
        Visualize PCA results
        
        Args:
            output_type: 'terminal' or 'web' for different output formats
        """
        if self.pca is None:
            print("PCA not applied yet. Please run apply_pca() first.")
            return
        
        if output_type == 'terminal':
            print("\n" + "=" * 60)
            print("PCA ANALYSIS RESULTS")
            print("=" * 60)
            
            print(f"\nNumber of components: {self.pca.n_components_}")
            print(f"Original features: {len(self.feature_columns)}")
            print(f"Reduced features: {self.pca.n_components_}")
            print(f"Dimensionality reduction: {len(self.feature_columns)} → {self.pca.n_components_}")
            
            print("\nExplained Variance by Component:")
            print("-" * 50)
            cumulative_variance = 0
            for i, variance in enumerate(self.explained_variance_ratio):
                cumulative_variance += variance
                print(f"PC{i+1:<3} | {variance:>8.4f} | Cumulative: {cumulative_variance:>8.4f}")
                
        elif output_type == 'web':
            # Create PCA visualization plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. Explained variance ratio
            axes[0, 0].bar(range(1, len(self.explained_variance_ratio) + 1), 
                          self.explained_variance_ratio)
            axes[0, 0].set_xlabel('Principal Component')
            axes[0, 0].set_ylabel('Explained Variance Ratio')
            axes[0, 0].set_title('Explained Variance by Component')
            
            # 2. Cumulative explained variance
            cumulative_variance = np.cumsum(self.explained_variance_ratio)
            axes[0, 1].plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-')
            axes[0, 1].axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
            axes[0, 1].set_xlabel('Number of Components')
            axes[0, 1].set_ylabel('Cumulative Explained Variance')
            axes[0, 1].set_title('Cumulative Explained Variance')
            axes[0, 1].legend()
            
            # 3. First two principal components scatter plot
            if self.pca.n_components_ >= 2 and self.target_columns:
                target_col = self.target_columns[0]
                scatter = axes[1, 0].scatter(self.pca_data['PC1'], self.pca_data['PC2'], 
                                           c=self.pca_data[target_col], cmap='viridis', alpha=0.6)
                axes[1, 0].set_xlabel('First Principal Component')
                axes[1, 0].set_ylabel('Second Principal Component')
                axes[1, 0].set_title('First Two Principal Components')
                plt.colorbar(scatter, ax=axes[1, 0])
            
            # 4. Component loadings heatmap (top features)
            if self.pca.n_components_ <= 10:  # Only for reasonable number of components
                loadings = pd.DataFrame(
                    self.pca.components_.T,
                    columns=[f'PC{i+1}' for i in range(self.pca.n_components_)],
                    index=self.feature_columns
                )
                
                # Select top features by absolute loading values
                top_features = loadings.abs().sum(axis=1).nlargest(20).index
                
                sns.heatmap(loadings.loc[top_features], annot=True, cmap='coolwarm', 
                           center=0, ax=axes[1, 1])
                axes[1, 1].set_title('PCA Loadings (Top 20 Features)')
                axes[1, 1].set_xlabel('Principal Components')
                axes[1, 1].set_ylabel('Features')
            
            plt.tight_layout()
            plt.savefig('pca_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def get_feature_loadings(self, top_n=10):
        """
        Get feature loadings for each principal component
        
        Args:
            top_n: number of top features to show for each component
        """
        if self.pca is None:
            print("PCA not applied yet. Please run apply_pca() first.")
            return None
        
        loadings_df = pd.DataFrame(
            self.pca.components_.T,
            columns=[f'PC{i+1}' for i in range(self.pca.n_components_)],
            index=self.feature_columns
        )
        
        print(f"\nTop {top_n} feature loadings for each component:")
        print("=" * 70)
        
        for pc in loadings_df.columns:
            print(f"\n{pc}:")
            print("-" * 40)
            top_features = loadings_df[pc].abs().nlargest(top_n)
            for feature, loading in top_features.items():
                actual_loading = loadings_df.loc[feature, pc]
                print(f"{feature:<30} | {actual_loading:>8.4f}")
        
        return loadings_df
    
    def get_pca_summary(self):
        """
        Get summary of PCA results
        """
        if self.pca is None:
            return "PCA not applied yet."
        
        summary = {
            'original_features': len(self.feature_columns),
            'pca_components': self.pca.n_components_,
            'total_explained_variance': np.sum(self.explained_variance_ratio),
            'individual_explained_variance': self.explained_variance_ratio.tolist(),
            'data_shape_before': (self.df.shape[0], len(self.feature_columns)),
            'data_shape_after': self.pca_data.shape
        }
        
        return summary
    
    def save_pca_data(self, output_path):
        """
        Save PCA-transformed data to CSV file
        
        Args:
            output_path: path to save the PCA data
        """
        if self.pca_data is None:
            print("No PCA data to save. Please apply PCA first.")
            return
        
        self.pca_data.to_csv(output_path, index=False)
        print(f"PCA data saved to: {output_path}")

# Main processing function
def process_feature_extraction(data, output_type='terminal', n_components=None, variance_threshold=0.95):
    """
    Main function to process feature extraction
    
    Args:
        data: pandas DataFrame or path to CSV
        output_type: 'terminal' or 'web'
        n_components: number of PCA components (if None, determined automatically)
        variance_threshold: variance threshold for automatic component selection
    """
    print("Starting Feature Extraction...")
    
    # Initialize extractor
    extractor = FeatureExtractor(data)
    
    # Identify features and targets
    feature_cols, target_cols = extractor.identify_features_and_targets()
    
    # Analyze feature importance
    if target_cols:
        extractor.analyze_feature_importance(method='f_classif', k=10)
    
    # Apply PCA
    pca_data = extractor.apply_pca(n_components, variance_threshold)
    
    # Visualize results
    extractor.visualize_pca_results(output_type)
    
    # Show feature loadings
    extractor.get_feature_loadings(top_n=5)
    
    # Get summary
    summary = extractor.get_pca_summary()
    
    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"Original features: {summary['original_features']}")
    print(f"PCA components: {summary['pca_components']}")
    print(f"Dimensionality reduction: {summary['data_shape_before'][1]} → {summary['pca_components']}")
    print(f"Total explained variance: {summary['total_explained_variance']:.4f}")
    print(f"Data shape after PCA: {summary['data_shape_after']}")
    
    return pca_data, extractor, summary

if __name__ == "__main__":
    # Test the feature extractor
    data_path = "../risk_factors_cervical_cancer.csv"
    pca_data, extractor, summary = process_feature_extraction(
        data_path, output_type='terminal', variance_threshold=0.95
    )
    print("\nFeature extraction completed!")