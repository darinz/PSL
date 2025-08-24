"""
Loan Example - Decision Tree Classification

This module demonstrates decision tree classification for loan applications,
including feature engineering, tree construction, evaluation, and visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class LoanDecisionTree:
    """Decision tree classifier for loan applications"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.tree = None
        self.label_encoders = {}
        self.feature_names = None
        
    def create_loan_dataset(self, n_samples=1000):
        """Create a synthetic loan dataset"""
        np.random.seed(self.random_state)
        
        # Generate features
        data = {
            'credit_history': np.random.choice(['excellent', 'good', 'fair', 'poor'], n_samples, p=[0.3, 0.4, 0.2, 0.1]),
            'income': np.random.choice(['high', 'medium', 'low'], n_samples, p=[0.4, 0.4, 0.2]),
            'loan_term': np.random.choice(['3_years', '5_years', '10_years'], n_samples, p=[0.3, 0.4, 0.3]),
            'age': np.random.normal(35, 10, n_samples).astype(int),
            'marital_status': np.random.choice(['single', 'married', 'divorced'], n_samples, p=[0.4, 0.5, 0.1]),
            'employment_years': np.random.exponential(3, n_samples).astype(int),
            'loan_amount': np.random.lognormal(10, 0.5, n_samples).astype(int)
        }
        
        df = pd.DataFrame(data)
        
        # Create target variable based on business rules
        def determine_loan_status(row):
            # Excellent credit - always safe
            if row['credit_history'] == 'excellent':
                return 'safe'
            
            # Poor credit with low income - risky
            if row['credit_history'] == 'poor' and row['income'] == 'low':
                return 'risky'
            
            # Fair credit with short term - risky
            if row['credit_history'] == 'fair' and row['loan_term'] == '3_years':
                return 'risky'
            
            # Poor credit with high income and long term - safe
            if row['credit_history'] == 'poor' and row['income'] == 'high' and row['loan_term'] == '10_years':
                return 'safe'
            
            # Good credit - generally safe
            if row['credit_history'] == 'good':
                return 'safe'
            
            # Default case
            return 'risky'
        
        df['loan_status'] = df.apply(determine_loan_status, axis=1)
        
        # Add some noise to make it more realistic
        noise_mask = np.random.random(n_samples) < 0.1
        df.loc[noise_mask, 'loan_status'] = np.random.choice(['safe', 'risky'], sum(noise_mask))
        
        return df
    
    def preprocess_data(self, df):
        """Preprocess the loan dataset"""
        # Create a copy to avoid modifying original data
        df_processed = df.copy()
        
        # Encode categorical variables
        categorical_features = ['credit_history', 'income', 'loan_term', 'marital_status']
        
        for feature in categorical_features:
            le = LabelEncoder()
            df_processed[feature] = le.fit_transform(df_processed[feature])
            self.label_encoders[feature] = le
        
        # Encode target variable
        le_target = LabelEncoder()
        df_processed['loan_status'] = le_target.fit_transform(df_processed['loan_status'])
        self.label_encoders['loan_status'] = le_target
        
        return df_processed
    
    def train_tree(self, X, y, max_depth=5, min_samples_split=10):
        """Train the decision tree"""
        self.tree = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=self.random_state
        )
        self.tree.fit(X, y)
        
    def evaluate_tree(self, X_test, y_test):
        """Evaluate the decision tree performance"""
        if self.tree is None:
            raise ValueError("Tree not trained yet. Call train_tree() first.")
        
        # Make predictions
        y_pred = self.tree.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(self.tree, X_test, y_test, cv=5)
        
        return {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': y_pred
        }
    
    def analyze_feature_importance(self, feature_names):
        """Analyze feature importance"""
        if self.tree is None:
            raise ValueError("Tree not trained yet. Call train_tree() first.")
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.tree.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def visualize_tree(self, feature_names, class_names):
        """Visualize the decision tree"""
        if self.tree is None:
            raise ValueError("Tree not trained yet. Call train_tree() first.")
        
        plt.figure(figsize=(20, 12))
        plot_tree(self.tree, 
                 feature_names=feature_names,
                 class_names=class_names,
                 filled=True,
                 rounded=True,
                 fontsize=10)
        plt.savefig('loan_decision_tree.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Decision tree visualization saved as 'loan_decision_tree.png'")
    
    def print_tree_text(self, feature_names, class_names):
        """Print the decision tree in text format"""
        if self.tree is None:
            raise ValueError("Tree not trained yet. Call train_tree() first.")
        
        tree_text = export_text(self.tree, 
                               feature_names=feature_names,
                               class_names=class_names)
        print("Decision Tree Structure:")
        print(tree_text)
    
    def demonstrate_loan_scoring(self, df, sample_applications):
        """Demonstrate loan scoring for specific applications"""
        print("\n=== Loan Application Scoring Examples ===")
        
        for i, application in enumerate(sample_applications, 1):
            print(f"\nApplication {i}:")
            for feature, value in application.items():
                print(f"  {feature}: {value}")
            
            # Preprocess the application
            app_df = pd.DataFrame([application])
            app_processed = self.preprocess_data(app_df)
            
            # Make prediction
            features = app_processed.drop('loan_status', axis=1, errors='ignore')
            prediction = self.tree.predict(features)[0]
            probability = self.tree.predict_proba(features)[0]
            
            # Convert back to original labels
            original_prediction = self.label_encoders['loan_status'].inverse_transform([prediction])[0]
            
            print(f"  Prediction: {original_prediction}")
            print(f"  Confidence: {max(probability):.2f}")
    
    def analyze_decision_paths(self, df, feature_names):
        """Analyze decision paths for different credit histories"""
        print("\n=== Decision Path Analysis ===")
        
        # Group by credit history
        credit_groups = df.groupby('credit_history')
        
        for credit_type, group in credit_groups:
            print(f"\nCredit History: {credit_type}")
            print(f"Number of applications: {len(group)}")
            
            # Calculate approval rate
            approval_rate = (group['loan_status'] == 0).mean()  # Assuming 0 = safe
            print(f"Approval rate: {approval_rate:.2%}")
            
            # Show distribution of other features
            print("Income distribution:")
            income_dist = group['income'].value_counts()
            for income, count in income_dist.items():
                income_label = self.label_encoders['income'].inverse_transform([income])[0]
                print(f"  {income_label}: {count}")
    
    def create_loan_visualizations(self, df):
        """Create visualizations for loan data analysis"""
        print("\n=== Creating Loan Data Visualizations ===")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Credit History vs Loan Status
        credit_status = pd.crosstab(df['credit_history'], df['loan_status'])
        credit_status.plot(kind='bar', ax=axes[0, 0], title='Credit History vs Loan Status')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Income vs Loan Status
        income_status = pd.crosstab(df['income'], df['loan_status'])
        income_status.plot(kind='bar', ax=axes[0, 1], title='Income vs Loan Status')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Loan Term vs Loan Status
        term_status = pd.crosstab(df['loan_term'], df['loan_status'])
        term_status.plot(kind='bar', ax=axes[0, 2], title='Loan Term vs Loan Status')
        axes[0, 2].set_ylabel('Count')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. Age distribution by loan status
        df.boxplot(column='age', by='loan_status', ax=axes[1, 0])
        axes[1, 0].set_title('Age Distribution by Loan Status')
        axes[1, 0].set_xlabel('Loan Status')
        
        # 5. Employment years by loan status
        df.boxplot(column='employment_years', by='loan_status', ax=axes[1, 1])
        axes[1, 1].set_title('Employment Years by Loan Status')
        axes[1, 1].set_xlabel('Loan Status')
        
        # 6. Loan amount by loan status
        df.boxplot(column='loan_amount', by='loan_status', ax=axes[1, 2])
        axes[1, 2].set_title('Loan Amount by Loan Status')
        axes[1, 2].set_xlabel('Loan Status')
        
        plt.tight_layout()
        plt.savefig('loan_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Loan analysis visualizations saved as 'loan_analysis.png'")
    
    def run_complete_analysis(self):
        """Run complete loan analysis"""
        print("=== Loan Decision Tree Analysis ===")
        
        # 1. Create dataset
        print("\n1. Creating loan dataset...")
        df = self.create_loan_dataset(n_samples=1000)
        print(f"Dataset created with {len(df)} samples")
        print(f"Loan status distribution:\n{df['loan_status'].value_counts()}")
        
        # 2. Preprocess data
        print("\n2. Preprocessing data...")
        df_processed = self.preprocess_data(df)
        
        # 3. Split data
        X = df_processed.drop('loan_status', axis=1)
        y = df_processed['loan_status']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state
        )
        
        # 4. Train tree
        print("\n3. Training decision tree...")
        self.train_tree(X_train, y_train, max_depth=5)
        
        # 5. Evaluate performance
        print("\n4. Evaluating performance...")
        results = self.evaluate_tree(X_test, y_test)
        print(f"Accuracy: {results['accuracy']:.3f}")
        print(f"Cross-validation score: {results['cv_mean']:.3f} (+/- {results['cv_std']*2:.3f})")
        
        # 6. Feature importance
        print("\n5. Analyzing feature importance...")
        feature_names = X.columns.tolist()
        importance_df = self.analyze_feature_importance(feature_names)
        print("Feature Importance:")
        print(importance_df)
        
        # 7. Visualize tree
        print("\n6. Creating tree visualization...")
        class_names = ['safe', 'risky']
        self.visualize_tree(feature_names, class_names)
        
        # 8. Print tree structure
        print("\n7. Tree structure in text format:")
        self.print_tree_text(feature_names, class_names)
        
        # 9. Demonstrate loan scoring
        print("\n8. Loan scoring examples...")
        sample_applications = [
            {
                'credit_history': 'excellent',
                'income': 'high',
                'loan_term': '5_years',
                'age': 35,
                'marital_status': 'married',
                'employment_years': 5,
                'loan_amount': 50000
            },
            {
                'credit_history': 'poor',
                'income': 'low',
                'loan_term': '3_years',
                'age': 25,
                'marital_status': 'single',
                'employment_years': 1,
                'loan_amount': 10000
            },
            {
                'credit_history': 'fair',
                'income': 'medium',
                'loan_term': '10_years',
                'age': 45,
                'marital_status': 'married',
                'employment_years': 10,
                'loan_amount': 75000
            }
        ]
        self.demonstrate_loan_scoring(df, sample_applications)
        
        # 10. Analyze decision paths
        print("\n9. Decision path analysis...")
        self.analyze_decision_paths(df_processed, feature_names)
        
        # 11. Create visualizations
        print("\n10. Creating visualizations...")
        self.create_loan_visualizations(df)
        
        return {
            'dataset': df,
            'processed_data': df_processed,
            'results': results,
            'importance': importance_df
        }

def main():
    """Main function to run the loan analysis"""
    # Create loan decision tree instance
    loan_tree = LoanDecisionTree(random_state=42)
    
    # Run complete analysis
    results = loan_tree.run_complete_analysis()
    
    print("\n=== Analysis Complete ===")
    print("Check the generated files:")
    print("- loan_decision_tree.png: Decision tree visualization")
    print("- loan_analysis.png: Data analysis plots")

if __name__ == "__main__":
    main()
