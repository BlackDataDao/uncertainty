import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency
import numpy as np
import os
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import json

# Define roles and their short names
ROLES = {
    "stock broker": "Stock Broker",
    "stock broker whose role is to encourage clients to buy more stocks.": "Misleading Stock Broker",
    "risk manager": "Risk Manager",
    "risk manager whose role is to encourage clients to buy more stocks.": "Misleading Risk Manager",
}

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

def generate_statistics_excel(df, roles, output_folder, base_filename):
    """
    Generates an Excel file with statistics for each role's scores, role score pairs, and chi-square test results.
    """
    try:
        # Prepare data for Excel
        statistics_data = []
        chi_square_data = []

        # Significance level
        alpha = 0.05

        # Calculate mean scores for each role
        role_scores = {}
        for role in roles:
            col_name = f"{role}_score"
            if col_name in df.columns:
                role_scores[role] = pd.to_numeric(df[col_name], errors='coerce')
            else:
                print(f"⚠️ Warning: Could not find column for {role}")
                continue

        mean_scores = {role: scores.mean() for role, scores in role_scores.items()}
        for role, mean in mean_scores.items():
            short_role = ROLES.get(role, role)  # Get short name or use full name if not found
            statistics_data.append({"Role": short_role, "Mean Score": mean})

        # Pairwise t-tests for role scores
        for i in range(len(roles)):
            if roles[i] not in role_scores:
                continue
            for j in range(i + 1, len(roles)):
                if roles[j] not in role_scores:
                    continue
                role1, role2 = roles[i], roles[j]
                short_role1 = ROLES.get(role1, role1)
                short_role2 = ROLES.get(role2, role2)
                t_stat, p_value = stats.ttest_ind(
                    role_scores[role1].dropna(),
                    role_scores[role2].dropna(),
                    nan_policy='omit'
                )
                mean_diff = mean_scores[role1] - mean_scores[role2]
                statistics_data.append({
                    "Role Pair": f"{short_role1} vs {short_role2}",
                    "Mean Difference": mean_diff,
                    "P-value (T-test)": p_value,
                    "Significant (T-test)": "Yes" if p_value < alpha else "No"
                })

        # Pairwise chi-square test for recommendation counts
        for i in range(len(roles)):
            for j in range(i + 1, len(roles)):
                role1, role2 = roles[i], roles[j]
                short_role1 = ROLES.get(role1, role1)
                short_role2 = ROLES.get(role2, role2)
                col_name1 = f"{role1}_recommendation"
                col_name2 = f"{role2}_recommendation"

                if col_name1 in df.columns and col_name2 in df.columns:
                    recommendations1 = df[col_name1].str.lower()
                    recommendations2 = df[col_name2].str.lower()

                    yes_count1 = (recommendations1 == 'yes').sum()
                    no_count1 = (recommendations1 == 'no').sum()
                    yes_count2 = (recommendations2 == 'yes').sum()
                    no_count2 = (recommendations2 == 'no').sum()

                    contingency_table = [[yes_count1, no_count1], [yes_count2, no_count2]]
                    chi2, p, dof, expected = chi2_contingency(contingency_table)

                    yes_diff = yes_count1 - yes_count2
                    chi_square_data.append({
                        "Role Pair": f"{short_role1} vs {short_role2}",
                        "Yes Count Difference": yes_diff,
                        "P-value (Chi-square)": p,
                        "Significant (Chi-square)": "Yes" if p < alpha else "No"
                    })

        # Create Excel file
        excel_file_path = os.path.join(output_folder, f"{base_filename}_statistics.xlsx")
        with pd.ExcelWriter(excel_file_path) as writer:
            pd.DataFrame(statistics_data).to_excel(writer, sheet_name="Role Scores Statistics", index=False)
            pd.DataFrame(chi_square_data).to_excel(writer, sheet_name="Chi-square Test Results", index=False)

        print(f"✅ Statistics Excel file saved as '{excel_file_path}'")
    except Exception as e:
        print(f"❌ Error generating statistics Excel file: {type(e).__name__}: {e}")

def test_recommendation_count_significance(df, roles):
    """
    Tests if the difference in recommendation counts ("Yes") between roles is statistically significant using a Chi-Square test.
    """
    try:
        # Create a contingency table
        observed = []
        for role in roles:
            col_name = f"{role}_recommendation"
            recommendations = df[col_name]
            recommendations = recommendations.str.lower()
            
            yes_count = (recommendations == 'yes').sum()
            no_count = (recommendations == 'no').sum()
            observed.append([yes_count, no_count])

        # Perform Chi-Square test
        chi2, p, dof, expected = chi2_contingency(observed)

        print("\n=== Chi-Square Test for Recommendation Significance ===")
        print(f"Chi-Square Statistic: {chi2:.3f}")
        print(f"P-value: {p:.3f}")
        print(f"Degrees of Freedom: {dof}")

        alpha = 0.05
        if p < alpha:
            print("The difference in recommendation counts between roles is statistically significant.")
        else:
            print("The difference in recommendation counts between roles is not statistically significant.")

    except Exception as e:
        print(f"❌ Error during Chi-Square test: {type(e).__name__}: {e}")

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
    
    # Create base filename and output folder
    base_filename = os.path.splitext(os.path.basename(csv_file))[0]
    output_folder = f"{base_filename}_analysis_results"
    os.makedirs(output_folder, exist_ok=True)
    print(f"✅ Output will be saved to folder: '{output_folder}'")
    
    # 1. Statistical Analysis
    print("\n=== Statistical Analysis of Role Scores ===")
    
    # Extract scores for each role
    roles = list(ROLES.keys())  # Use keys from ROLES dict
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
        short_role = ROLES.get(role, role)
        print(f"{short_role}: {mean:.3f}")
    
    # Pairwise t-tests
    print("\nPairwise T-Tests:")
    for i in range(len(roles)):
        if roles[i] not in role_scores:
            continue
        for j in range(i+1, len(roles)):
            if roles[j] not in role_scores:
                continue
            role1, role2 = roles[i], roles[j]
            short_role1 = ROLES.get(role1, role1)
            short_role2 = ROLES.get(role2, role2)
            t_stat, p_value = stats.ttest_ind(
                role_scores[role1].dropna(),
                role_scores[role2].dropna(),
                nan_policy='omit'
            )
            print(f"\n{short_role1} vs {short_role2}:")
            print(f"t-statistic: {t_stat:.3f}")
            print(f"p-value: {p_value:.3f}")
            print(f"Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
            print(f"Mean difference: {mean_scores[role1] - mean_scores[role2]:.3f}")
    
    # 2. Generate Excel file with statistics
    generate_statistics_excel(df, roles, output_folder, base_filename)
    
    # 3. Visualization - Group by Age and Role
    
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
    
    # 4. Generate all visualizations with base_filename
    # base_filename = os.path.splitext(os.path.basename(csv_file))[0] # Moved up
    generate_facet_grid(df, roles, base_filename, output_folder)
    generate_heatmap(df, roles, base_filename, output_folder)
    generate_interactive_plot(df, roles, base_filename, output_folder)
    generate_violin_plot(df, roles, base_filename, output_folder)
    generate_boxplot_matrix(df, roles, base_filename, output_folder)
    generate_recommendation_counts(df, roles, base_filename, output_folder)
    
    # 5. Test Significance of Recommendation Counts
    test_recommendation_count_significance(df, roles)
    
    # Save role short name mapping to JSON
    roles_mapping_path = os.path.join(output_folder, "roles_mapping.json")
    with open(roles_mapping_path, 'w') as f:
        json.dump(ROLES, f, indent=4)
    print(f"✅ Role short name mapping saved to '{roles_mapping_path}'")

    # Print role short name mapping
    print("\n=== Role Short Name Mapping ===")
    for role, short_role in ROLES.items():
        print(f"{short_role}: {role}")

def generate_facet_grid(df, roles, base_filename, output_folder):
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
                short_role = ROLES.get(role, role)
                ax.plot(plot_data['percentage'], plot_data['score'],
                        label=short_role, color=role_colors.get(role, 'gray'))

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
        save_path = os.path.join(output_folder, f"facet_grid_{base_filename}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"✅ Facet grid saved as '{save_path}'")
    except Exception as e:
        print(f"❌ Error generating facet grid: {type(e).__name__}: {e}")

def generate_heatmap(df, roles, base_filename, output_folder):
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
            short_role = ROLES.get(role, role)
            sns.heatmap(pivot_data[role], annot=True, fmt=".2f", cmap="YlGnBu", ax=axes[i])
            axes[i].set_title(f"{short_role}")
            axes[i].set_ylabel("Data Point Index")  # Update y-axis label
            axes[i].set_xlabel("Investment % Range")

        plt.tight_layout()
        save_path = os.path.join(output_folder, f"heatmap_{base_filename}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"✅ Heatmap saved as '{save_path}'")
    except Exception as e:
        print(f"❌ Error generating heatmap: {type(e).__name__}: {e}")

def generate_interactive_plot(df, roles, base_filename, output_folder):
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
            hover_data=['Age', 'Role', 'Score'],
            category_orders={"Role": [ROLES.get(role, role) for role in roles]}  # Order roles by short name
        )

        fig.update_layout(legend_title="Role")  # Update legend title

        save_path = os.path.join(output_folder, f'interactive_{base_filename}.html')
        pio.write_html(fig, save_path)
        print(f"✅ Interactive plot saved as '{save_path}'")
    except Exception as e:
        print(f"❌ Error generating interactive plot: {type(e).__name__}: {e}")

def generate_violin_plot(df, roles, base_filename, output_folder):
    """
    Generates an interactive violin plot of investment scores by role using Plotly.
    """
    try:
        # Reshape data for Plotly
        plot_data = []
        for role in roles:
            role_scores = pd.to_numeric(df[f"{role}_score"], errors='coerce')
            ages = df['age']
            for age, score in zip(ages, role_scores):
                if not pd.isna(score):
                    plot_data.append({
                        'Role': ROLES.get(role, role),  # Use short name
                        'Age': age,
                        'Score': score
                    })

        plot_df = pd.DataFrame(plot_data)

        # Create violin plot using Plotly
        fig = px.violin(plot_df, y="Score", color="Role", box=True, points="all",
                  hover_data=['Age', 'Role', 'Score'],
                  category_orders={"Role": [ROLES.get(role, role) for role in roles]}  # Order roles by short name
        )

        fig.update_layout(title='Distribution of Scores by Role',
                          yaxis_title='Score',
                          legend_title="Role")

        save_path = os.path.join(output_folder, f"violin_plot_{base_filename}.html")
        pio.write_html(fig, save_path)
        print(f"✅ Interactive violin plot saved as '{save_path}'")
    except Exception as e:
        print(f"❌ Error generating violin plot: {type(e).__name__}: {e}")

def generate_boxplot_matrix(df, roles, base_filename, output_folder):
    """
    Generates a matrix of box plots showing the distribution of investment scores by role.
    """
    try:
        # Create a grid of box plots
        fig, axes = plt.subplots(1, len(roles), figsize=(15, 6), sharey=True)
        for i, role in enumerate(roles):
            short_role = ROLES.get(role, role)
            scores = pd.to_numeric(df[f"{role}_score"], errors='coerce')
            axes[i].boxplot(scores.dropna())
            axes[i].set_title(short_role)
            axes[i].set_ylabel('Score')

        plt.tight_layout()
        save_path = os.path.join(output_folder, f"boxplot_matrix_{base_filename}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"✅ Box plot matrix saved as '{save_path}'")
    except Exception as e:
        print(f"❌ Error generating box plot matrix: {type(e).__name__}: {e}")

def generate_recommendation_counts(df, roles, base_filename, output_folder):
    """
    Calculates and visualizes the number of Yes/No recommendations for each role.
    """
    try:
        # Calculate counts for each role
        counts = {}
        for role in roles:
            col_name = f"{role}_recommendation"
            recommendations = df[col_name]
            recommendations = recommendations.str.lower()
            yes_count = (recommendations == 'yes').sum()
            no_count = (recommendations == 'no').sum()
            counts[role] = {'Yes': yes_count, 'No': no_count}

        # Create bar plot
        plt.figure(figsize=(12, 6))
        
        # Set up bars
        roles_list = [ROLES.get(role, role) for role in counts.keys()]  # Use short names
        x = np.arange(len(roles_list))
        width = 0.35

        # Plot bars
        yes_bars = plt.bar(x - width/2, [counts[role]['Yes'] for role in counts.keys()], 
                          width, label='Yes', color='green')
        no_bars = plt.bar(x + width/2, [counts[role]['No'] for role in counts.keys()], 
                         width, label='No', color='red')

        # Customize plot
        plt.xlabel('Roles')
        plt.ylabel('Number of Recommendations')
        plt.title('Yes/No Recommendations by Role')
        plt.xticks(x, roles_list)
        plt.legend()

        # Add value labels on the bars
        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom')

        autolabel(yes_bars)
        autolabel(no_bars)

        # Save plot
        plt.tight_layout()
        save_path = os.path.join(output_folder, f"recommendation_counts_{base_filename}.png")
        plt.savefig(save_path)
        plt.close()
        
        # Print counts
        print("\n=== Recommendation Counts ===")
        for role in counts.keys():
            short_role = ROLES.get(role, role)
            print(f"\n{short_role}:")
            print(f"Yes: {counts[role]['Yes']}")
            print(f"No: {counts[role]['No']}")
            total = counts[role]['Yes'] + counts[role]['No']
            yes_percent = (counts[role]['Yes'] / total * 100) if total > 0 else 0
            print(f"Yes Percentage: {yes_percent:.1f}%")
        
        print(f"\n✅ Recommendation counts plot saved as '{save_path}'")
        
    except Exception as e:
        print(f"❌ Error generating recommendation counts: {type(e).__name__}: {e}")

# 
csv_file="r_gpt-4o-mini_recommendations_only_q2_20250615_161212.csv"  # Replace with your actual CSV file path
analyze_role_scores(csv_file)