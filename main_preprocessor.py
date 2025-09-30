import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add subdirectories to path
sys.path.append('MissingValue')
sys.path.append('Transformasi')
sys.path.append('SeleksiFitur')
sys.path.append('ImbalancedData')

# Import custom modules
from missing_value import MissingValueHandler, process_missing_values
from transformasi import DataTransformer, process_data_transformation
from seleksi_fitur import FeatureSelector, process_feature_selection
from imbalanced_data import ImbalancedDataHandler, process_imbalanced_data

class CervicalCancerPreprocessor:
    def __init__(self, data_path):
        """
        Initialize the main preprocessor
        
        Args:
            data_path: path to the cervical cancer CSV file
        """
        self.data_path = data_path
        self.original_data = None
        self.processed_data = None
        self.results = {}
        
        # Load original data
        self.load_data()
    
    def load_data(self):
        """Load the original dataset"""
        try:
            self.original_data = pd.read_csv(self.data_path)
            print(f"✅ Data loaded successfully!")
            print(f"   Shape: {self.original_data.shape}")
            print(f"   Columns: {len(self.original_data.columns)}")
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return False
        return True
    
    def run_missing_value_analysis(self, output_type='terminal', strategy='median'):
        """
        Step 1: Handle missing values
        """
        print("\n" + "🔍 STEP 1: MISSING VALUE ANALYSIS" + "\n" + "="*50)
        
        try:
            cleaned_data, summary = process_missing_values(
                self.data_path, output_type=output_type, strategy=strategy
            )
            
            self.results['missing_values'] = {
                'cleaned_data': cleaned_data,
                'summary': summary,
                'strategy': strategy
            }
            
            # Update processed data
            self.processed_data = cleaned_data
            
            print("✅ Missing value analysis completed!")
            return True
            
        except Exception as e:
            print(f"❌ Error in missing value analysis: {str(e)}")
            return False
    
    def run_data_transformation(self, output_type='terminal', scaler_type='minmax'):
        """
        Step 2: Data transformation (MinMaxScaler)
        """
        print("\n" + "🔧 STEP 2: DATA TRANSFORMATION" + "\n" + "="*50)
        
        if self.processed_data is None:
            print("❌ No processed data available. Run missing value analysis first.")
            return False
        
        try:
            transformed_data, transformer, summary = process_data_transformation(
                self.processed_data, output_type=output_type, scaler_type=scaler_type
            )
            
            self.results['transformation'] = {
                'transformed_data': transformed_data,
                'transformer': transformer,
                'summary': summary,
                'scaler_type': scaler_type
            }
            
            # Update processed data
            self.processed_data = transformed_data
            
            print("✅ Data transformation completed!")
            return True
            
        except Exception as e:
            print(f"❌ Error in data transformation: {str(e)}")
            return False
    
    def run_feature_extraction(self, output_type='terminal', variance_threshold=0.95):
        """
        Step 3: Feature extraction (Seleksi Fitur ANOVA)
        """
        print("\n" + "📊 STEP 3: FEATURE SELECTION (ANOVA)" + "\n" + "="*50)
        
        if self.processed_data is None:
            print("❌ No processed data available. Run previous steps first.")
            return False
        
        try:
            # Only pass data and k (number of top features)
            selected_features, selector = process_feature_selection(
                self.processed_data, k=10
            )
            
            self.results['feature_extraction'] = {
                'selected_features': selected_features,
                'selector': selector,
                'summary': None
            }
            
            # Update processed data (optional: keep only selected features)
            if selected_features is not None:
                selected_cols = selected_features['Feature'].tolist()
                self.processed_data = self.processed_data[selected_cols + selector.target_columns]
            
            print("✅ Feature selection completed!")
            return True
            
        except Exception as e:
            print(f"❌ Error in feature selection: {str(e)}")
            return False
    
    def run_imbalanced_data_handling(self, output_type='terminal', method='ros'):
        """
        Step 4: Handle imbalanced data (Random Over Sampling)
        """
        print("\n" + "⚖️ STEP 4: IMBALANCED DATA HANDLING" + "\n" + "="*50)
        
        if self.processed_data is None:
            print("❌ No processed data available. Run previous steps first.")
            return False
        
        try:
            resampled_data, handler, summary = process_imbalanced_data(
                self.processed_data, output_type=output_type, method=method
            )
            
            self.results['imbalanced_data'] = {
                'resampled_data': resampled_data,
                'handler': handler,
                'summary': summary,
                'method': method
            }
            
            # Update processed data
            if resampled_data is not None:
                self.processed_data = resampled_data
            
            print("✅ Imbalanced data handling completed!")
            return True
            
        except Exception as e:
            print(f"❌ Error in imbalanced data handling: {str(e)}")
            return False
    
    def generate_final_summary(self):
        """Generate comprehensive summary of all preprocessing steps"""
        print("\n" + "📋 FINAL PREPROCESSING SUMMARY" + "\n" + "="*60)
        
        print(f"Original Dataset Shape: {self.original_data.shape}")
        print(f"Final Dataset Shape: {self.processed_data.shape}")
        print(f"Total Processing Steps: {len(self.results)}")
        
        print("\n" + "Processing Steps Summary:")
        print("-" * 60)
        
        # Step 1: Missing Values
        if 'missing_values' in self.results:
            mv_summary = self.results['missing_values']['summary']
            print(f"1. Missing Values:")
            print(f"   ├─ Strategy: {self.results['missing_values']['strategy']}")
            print(f"   ├─ Rows removed: {mv_summary['rows_removed']}")
            print(f"   ├─ Columns removed: {mv_summary['columns_removed']}")
            print(f"   └─ Remaining missing values: {mv_summary['remaining_missing_values']}")
        
        # Step 2: Transformation
        if 'transformation' in self.results:
            tr_summary = self.results['transformation']['summary']
            print(f"2. Data Transformation:")
            print(f"   ├─ Scaler: {tr_summary['scaler_type']}")
            print(f"   ├─ Features transformed: {tr_summary['features_transformed']}")
            print(f"   └─ Feature range: {tr_summary['feature_range_after']['min']:.4f} to {tr_summary['feature_range_after']['max']:.4f}")
        
        # Step 3: Feature Extraction
        if 'feature_extraction' in self.results:
            fe_summary = self.results['feature_extraction']['summary']
            print(f"3. Feature Extraction (PCA):")
            print(f"   ├─ Original features: {fe_summary['original_features']}")
            print(f"   ├─ PCA components: {fe_summary['pca_components']}")
            print(f"   └─ Explained variance: {fe_summary['total_explained_variance']:.4f}")
        
        # Step 4: Imbalanced Data
        if 'imbalanced_data' in self.results and self.results['imbalanced_data']['summary']:
            id_summary = self.results['imbalanced_data']['summary']
            print(f"4. Imbalanced Data Handling:")
            print(f"   ├─ Method: {id_summary['resampling_method']}")
            print(f"   ├─ Original imbalance ratio: {id_summary['original_imbalance_ratio']:.2f}:1")
            print(f"   └─ Final imbalance ratio: {id_summary['resampled_imbalance_ratio']:.2f}:1")
        
        print("\n" + "="*60)
        print("🎉 All preprocessing steps completed successfully!")
        
        return self.processed_data
    
    def save_results(self, output_dir='output'):
        """Save all results to files"""
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        print(f"\n💾 Saving results to '{output_dir}' directory...")
        
        # Save final processed data
        if self.processed_data is not None:
            final_path = os.path.join(output_dir, 'final_processed_data.csv')
            self.processed_data.to_csv(final_path, index=False)
            print(f"   ✅ Final processed data saved: {final_path}")
        
        # Save intermediate results
        if 'missing_values' in self.results:
            mv_path = os.path.join(output_dir, 'data_after_missing_value_handling.csv')
            self.results['missing_values']['cleaned_data'].to_csv(mv_path, index=False)
            print(f"   ✅ Data after missing value handling: {mv_path}")
        
        if 'transformation' in self.results:
            tr_path = os.path.join(output_dir, 'data_after_transformation.csv')
            self.results['transformation']['transformed_data'].to_csv(tr_path, index=False)
            print(f"   ✅ Data after transformation: {tr_path}")
        
        if 'feature_extraction' in self.results:
            fe_path = os.path.join(output_dir, 'data_after_feature_extraction.csv')
            self.results['feature_extraction']['pca_data'].to_csv(fe_path, index=False)
            print(f"   ✅ Data after feature extraction: {fe_path}")
        
        print("   💾 All results saved successfully!")
    
    def run_complete_pipeline(self, output_type='terminal', 
                            mv_strategy='median', 
                            scaler_type='minmax',
                            variance_threshold=0.95,
                            resampling_method='ros'):
        """
        Run the complete preprocessing pipeline
        
        Args:
            output_type: 'terminal' or 'web'
            mv_strategy: missing value strategy
            scaler_type: transformation scaler type
            variance_threshold: PCA variance threshold
            resampling_method: resampling method for imbalanced data
        """
        print("🚀 STARTING COMPLETE CERVICAL CANCER DATA PREPROCESSING PIPELINE")
        print("="*70)
        
        # Step 1: Missing Values
        if not self.run_missing_value_analysis(output_type, mv_strategy):
            return False
        
        # Step 2: Data Transformation
        if not self.run_data_transformation(output_type, scaler_type):
            return False
        
        # Step 3: Feature Extraction
        if not self.run_feature_extraction(output_type, variance_threshold):
            return False
        
        # Step 4: Imbalanced Data Handling
        if not self.run_imbalanced_data_handling(output_type, resampling_method):
            return False
        
        # Generate final summary
        final_data = self.generate_final_summary()
        
        # Save results
        self.save_results()
        
        return final_data

def display_menu():
    """Display interactive menu"""
    print("\n" + "="*60)
    print("🏥 CERVICAL CANCER DATA PREPROCESSING TOOLKIT")
    print("="*60)
    print("Choose an option:")
    print("1. Run Complete Pipeline (Terminal Output)")
    print("2. Run Complete Pipeline (Web/Visual Output)")
    print("3. Run Individual Steps")
    print("4. Custom Configuration")
    print("5. Exit")
    print("-"*60)

def display_individual_menu():
    """Display individual steps menu"""
    print("\n" + "Individual Steps:")
    print("1. Missing Value Analysis")
    print("2. Data Transformation (MinMaxScaler)")
    print("3. Feature Extraction (PCA)")
    print("4. Imbalanced Data Handling (ROS)")
    print("5. Back to Main Menu")

def get_user_choice(prompt, valid_choices):
    """Get valid user choice"""
    while True:
        try:
            choice = input(prompt).strip()
            if choice in valid_choices:
                return choice
            else:
                print(f"Invalid choice. Please choose from: {', '.join(valid_choices)}")
        except KeyboardInterrupt:
            print("\n\nProgram interrupted by user.")
            sys.exit(0)

def main():
    """Main function with interactive menu"""
    # Check if data file exists
    data_path = "risk_factors_cervical_cancer.csv"
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        print("Please ensure the CSV file is in the current directory.")
        return
    
    # Initialize preprocessor
    preprocessor = CervicalCancerPreprocessor(data_path)
    
    if preprocessor.original_data is None:
        print("❌ Failed to load data. Exiting.")
        return
    
    while True:
        display_menu()
        choice = get_user_choice("Enter your choice (1-5): ", ['1', '2', '3', '4', '5'])
        
        if choice == '1':
            # Complete pipeline - terminal output
            print("\n🖥️  Running complete pipeline with terminal output...")
            final_data = preprocessor.run_complete_pipeline(output_type='terminal')
            if final_data is not None:
                print("\n✅ Pipeline completed successfully!")
            else:
                print("\n❌ Pipeline failed!")
        
        elif choice == '2':
            # Complete pipeline - web output
            print("\n🌐 Running complete pipeline with web/visual output...")
            final_data = preprocessor.run_complete_pipeline(output_type='web')
            if final_data is not None:
                print("\n✅ Pipeline completed successfully!")
                print("📊 Check the generated plots and visualizations!")
            else:
                print("\n❌ Pipeline failed!")
        
        elif choice == '3':
            # Individual steps
            while True:
                display_individual_menu()
                step_choice = get_user_choice("Enter your choice (1-5): ", ['1', '2', '3', '4', '5'])
                
                output_choice = get_user_choice("Output type - (t)erminal or (w)eb: ", ['t', 'w'])
                output_type = 'terminal' if output_choice == 't' else 'web'
                
                if step_choice == '1':
                    preprocessor.run_missing_value_analysis(output_type=output_type)
                elif step_choice == '2':
                    preprocessor.run_data_transformation(output_type=output_type)
                elif step_choice == '3':
                    preprocessor.run_feature_extraction(output_type=output_type)
                elif step_choice == '4':
                    preprocessor.run_imbalanced_data_handling(output_type=output_type)
                elif step_choice == '5':
                    break
        
        elif choice == '4':
            # Custom configuration
            print("\n⚙️  Custom Configuration:")
            output_choice = get_user_choice("Output type - (t)erminal or (w)eb: ", ['t', 'w'])
            output_type = 'terminal' if output_choice == 't' else 'web'
            
            mv_strategy = get_user_choice("Missing value strategy - (m)edian, (e)an, mo(d)e, (r)op: ", ['m', 'e', 'd', 'r'])
            mv_strategy_map = {'m': 'median', 'e': 'mean', 'd': 'mode', 'r': 'drop'}
            mv_strategy = mv_strategy_map[mv_strategy]
            
            scaler_choice = get_user_choice("Scaler type - (m)inmax, (s)tandard, (r)obust: ", ['m', 's', 'r'])
            scaler_map = {'m': 'minmax', 's': 'standard', 'r': 'robust'}
            scaler_type = scaler_map[scaler_choice]
            
            resampling_choice = get_user_choice("Resampling method - (r)os, (s)mote, (a)dasyn: ", ['r', 's', 'a'])
            resampling_map = {'r': 'ros', 's': 'smote', 'a': 'adasyn'}
            resampling_method = resampling_map[resampling_choice]
            
            final_data = preprocessor.run_complete_pipeline(
                output_type=output_type,
                mv_strategy=mv_strategy,
                scaler_type=scaler_type,
                resampling_method=resampling_method
            )
            
            if final_data is not None:
                print("\n✅ Custom pipeline completed successfully!")
            else:
                print("\n❌ Custom pipeline failed!")
        
        elif choice == '5':
            print("\n👋 Thank you for using the Cervical Cancer Data Preprocessing Toolkit!")
            break
        
        # Ask if user wants to continue
        continue_choice = get_user_choice("\nDo you want to perform another operation? (y/n): ", ['y', 'n'])
        if continue_choice == 'n':
            print("\n👋 Thank you for using the Cervical Cancer Data Preprocessing Toolkit!")
            break

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {str(e)}")
        print("Please check your data file and try again.")