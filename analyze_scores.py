import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
import os
import plotly.express as px
import plotly.io as pio

def read_csv_with_encoding(file_path):
    """
    Try to read CSV file with different encodings.
    Returns DataFrame or None if all attempts fail.
    """
    encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            print(f"Trying to read with {encoding} encoding...")
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error reading file with {encoding}: {str(e)}")
            continue
    
    return None

def analyze_role_scores(csv_file):
    """
    Analyze role scores from the recommendations CSV file.
    
    Args:
        csv_file (str): Path to the CSV file containing role scores
    """
    # Check if file exists
    if not os.path.exists(csv_file):
        print(f"❌ Error: File '{csv_file}' not found")
        return

    # Try to read the CSV file with different encodings
    df = read_csv_with_encoding(csv_file)
    
    if df is None:
        print("❌ Error: Could not read the CSV file with any supported encoding")
        return
    
    # 1. Statistical Analysis
    print("\n=== Statistical Analysis of Role Scores ===")
    
    # Extract scores for each role
    roles = ['stock broker', 'financial advisor', 'risk manager']
    role_scores = {}
    for role in roles:
        col_name = f"{role}_score"
        if col_name.replace(' ', ',') in df.columns:
            # Handle column names with commas instead of spaces
            role_scores[role] = pd.to_numeric(df[col_name.replace(' ', ',')], errors='coerce')
        elif col_name in df.columns:
            role_scores[role] = pd.to_numeric(df[col_name], errors='coerce')
        else:
            print(f"⚠️ Warning: Could not find column for {role}")
            continue
    
    # Calculate mean scores
    print("\nMean Scores:")
    mean_scores = {role: scores.mean() for role, scores in role_scores.items()}
    for role, mean in mean_scores.items():
        print(f"{role}: {mean:.3f}")
    
    # Pairwise t-tests
    print("\nPairwise T-Tests:")
    for i in range(len(roles)):
        if roles[i] not in role_scores:
            continue
        for j in range(i+1, len(roles)):
            if roles[j] not in role_scores:
                continue
            role1, role2 = roles[i], roles[j]
            t_stat, p_value = stats.ttest_ind(
                role_scores[role1].dropna(),
                role_scores[role2].dropna(),
                nan_policy='omit'
            )
            print(f"\n{role1} vs {role2}:")
            print(f"t-statistic: {t_stat:.3f}")
            print(f"p-value: {p_value:.3f}")
            print(f"Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
            print(f"Mean difference: {mean_scores[role1] - mean_scores[role2]:.3f}")
    
    # 2. Visualization - Group by Age and Role
    
    # Check if 'age' column exists
    if 'age' not in df.columns:
        print("⚠️ Warning: 'age' column not found, using default visualization")
        age_groups = ["All"]
        df['age_group'] = "All"
    else:
        # Create age groups
        age_bins = [0, 30, 40, 50, 60, 100]
        age_labels = ['20-30', '31-40', '41-50', '51-60', '61+']
        df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)
        age_groups = df['age_group'].dropna().unique()
    
    # 3. Generate all visualizations
    generate_facet_grid(df, roles)  # Remove age_groups argument
    generate_heatmap(df, roles)
    generate_interactive_plot(df, roles)
    generate_violin_plot(df, roles)
    generate_boxplot_matrix(df, roles)

def generate_facet_grid(df, roles):
    """
    Generates a facet grid with a chart for each age, showing investment scores by role and percentage,
    arranged in rows of five.
    """
    try:
        # Check if 'age' column exists
        if 'age' not in df.columns:
            print("⚠️ Warning: 'age' column not found, cannot generate facet grid")
            return

        # Get unique ages
        ages = sorted(df['age'].dropna().unique())
        num_ages = len(ages)

        # Calculate number of rows and columns for the grid
        num_cols = min(num_ages, 5)  # Up to 5 columns
        num_rows = (num_ages + num_cols - 1) // num_cols  # Calculate number of rows

        # Create subplots
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 6 * num_rows), sharex=True, sharey=True)
        fig.suptitle('Investment Scores by Role and Percentage (Each Age)', fontsize=16)

        # Color palette for roles
        role_colors = {'stock broker': 'blue', 'financial advisor': 'green', 'risk manager': 'red'}

        # Plot data for each age
        for i, age in enumerate(ages):
            row_index = i // num_cols
            col_index = i % num_cols

            # Handle cases where there's only one row
            if num_rows == 1:
                ax = axes[col_index] if num_cols > 1 else axes
            else:
                ax = axes[row_index, col_index]

            ax.set_title(f'Age: {age}', fontsize=14)

            # Plot data for each role
            for role in roles:
                # Get valid scores and percentages for this age and role
                scores = pd.to_numeric(df[(df['age'] == age)][f"{role}_score"], errors='coerce')
                percentages = df[(df['age'] == age)]['percentage']
                valid = ~(scores.isna() | percentages.isna())

                # Sort data by percentage for a smooth line
                plot_data = pd.DataFrame({'percentage': percentages[valid], 'score': scores[valid]}).sort_values('percentage')

                # Plot line
                ax.plot(plot_data['percentage'], plot_data['score'],
                        label=role, color=role_colors.get(role, 'gray'))

            # Set labels and legend
            ax.set_xlabel("Investment Percentage", fontsize=12)
            ax.set_ylabel("Investment Score", fontsize=12)
            ax.legend(title="Role", fontsize=10)
            ax.grid(True, alpha=0.3)

        # Remove empty subplots if num_ages is not a multiple of 5
        if num_ages % 5 != 0:
            for i in range(num_ages % 5, num_cols):
                if num_rows == 1:
                    ax = axes[i] if num_cols > 1 else axes
                else:
                    ax = axes[num_rows - 1, i]
                ax.axis('off')  # Turn off the axis for empty subplots

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust layout to make room for suptitle
        plt.savefig("facet_grid_analysis.png")
        plt.close()
        print("✅ Facet grid saved as 'facet_grid_analysis.png'")
    except Exception as e:
        print(f"❌ Error generating facet grid: {type(e).__name__}: {e}")

def generate_heatmap(df, roles):
    """
    Generates a heatmap of average investment scores by role and investment percentage range.
    """
    try:
        # Create pivot tables for average scores by percentage
        pivot_data = {}
        for role in roles:
            # Create a pivot table for each role
            pivot = pd.pivot_table(
                df, 
                values=f"{role}_score",
                index=df.index,  # Use index as a dummy index
                columns=pd.cut(df['percentage'], bins=[0,20,40,60,80,100]), 
                aggfunc='mean'
            )
            pivot_data[role] = pivot

        # Plot heatmaps
        fig, axes = plt.subplots(1, len(roles), figsize=(15, 5))
        for i, role in enumerate(roles):
            sns.heatmap(pivot_data[role], annot=True, fmt=".2f", cmap="YlGnBu", ax=axes[i])
            axes[i].set_title(f"{role}")
            axes[i].set_ylabel("Data Point Index")  # Update y-axis label
            axes[i].set_xlabel("Investment % Range")

        plt.tight_layout()
        plt.savefig("heatmap_analysis.png")
        plt.close()
        print("✅ Heatmap saved as 'heatmap_analysis.png'")
    except Exception as e:
        print(f"❌ Error generating heatmap: {type(e).__name__}: {e}")

def generate_interactive_plot(df, roles):
    """
    Generates an interactive scatter plot of investment scores by role and investment percentage using Plotly.
    """
    try:
        # Prepare data in long format
        plot_data = []
        for role in roles:
            scores = pd.to_numeric(df[f"{role}_score"], errors='coerce')
            percentages = df['percentage']
            ages = df['age']
            
            for p, s, a in zip(percentages, scores, ages):
                if not pd.isna(s):
                    plot_data.append({
                        'Role': role,
                        'Percentage': p,
                        'Score': s,
                        'Age': a
                    })

        # Create interactive plot
        plot_df = pd.DataFrame(plot_data)
        fig = px.scatter(
            plot_df, 
            x='Percentage', 
            y='Score', 
            color='Role',
            hover_data=['Age', 'Role', 'Score']
        )

        pio.write_html(fig, 'interactive_analysis.html')
        print("✅ Interactive plot saved as 'interactive_analysis.html'")
    except Exception as e:
        print(f"❌ Error generating interactive plot: {type(e).__name__}: {e}")

def generate_violin_plot(df, roles):
    """
    Generates a violin plot of investment scores by role.
    """
    try:
        # Reshape data for seaborn
        plot_data = []
        for role in roles:
            role_scores = pd.to_numeric(df[f"{role}_score"], errors='coerce')
            ages = df['age']
            for age, score in zip(ages, role_scores):
                if not pd.isna(score):
                    plot_data.append({
                        'Role': role,
                        'Age': age,
                        'Score': score
                    })

        plot_df = pd.DataFrame(plot_data)

        # Create violin plot
        plt.figure(figsize=(14, 8))
        sns.violinplot(x='Role', y='Score', data=plot_df, inner="quart")
        plt.title('Distribution of Scores by Role')
        plt.savefig("violin_plot_analysis.png")
        plt.close()
        print("✅ Violin plot saved as 'violin_plot_analysis.png'")
    except Exception as e:
        print(f"❌ Error generating violin plot: {type(e).__name__}: {e}")

def generate_boxplot_matrix(df, roles):
    """
    Generates a matrix of box plots showing the distribution of investment scores by role.
    """
    try:
        # Create a grid of box plots
        fig, axes = plt.subplots(1, len(roles), figsize=(15, 6), sharey=True)
        for i, role in enumerate(roles):
            scores = pd.to_numeric(df[f"{role}_score"], errors='coerce')
            axes[i].boxplot(scores.dropna())
            axes[i].set_title(role)
            axes[i].set_ylabel('Score')

        plt.tight_layout()
        plt.savefig("boxplot_matrix.png")
        plt.close()
        print("✅ Box plot matrix saved as 'boxplot_matrix.png'")
    except Exception as e:
        print(f"❌ Error generating box plot matrix: {type(e).__name__}: {e}")


csv_file = "r_recommendations_main5_20250525_120043.csv"  # Replace with your actual CSV file path
analyze_role_scores(csv_file)