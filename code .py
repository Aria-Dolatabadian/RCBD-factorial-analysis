import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from itertools import combinations
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# PART 1: READ DATA
# ============================================================================

print("=" * 80)
print("CANOLA EXPERIMENT - RCBD FACTORIAL ANALYSIS")
print("=" * 80)

# Read data from CSV file
data = pd.read_csv('canola_experiment_data.csv')

print("\n✓ Data successfully loaded from 'canola_experiment_data.csv'")
print(f"Total observations: {len(data)}")
print("\nFirst few rows:")
print(data.head(10))
print("\nData structure:")
print(data.info())
print("\nDescriptive statistics:")
print(data.describe())

# ============================================================================
# PART 2: STATISTICAL ANALYSIS
# ============================================================================

# Define response variables (traits)
traits = ['Seed_Num_Per_Pod', 'Thousand_Seed_Weight', 'Pod_Num_Per_Plant',
          'Seed_Yield', 'Oil_Percentage', 'Oil_Yield', 'Plant_Biomass', 'Harvest_Index']

# Extract experimental factors from data
nitrogen_levels = sorted(data['Nitrogen'].unique())
irrigation_levels = sorted(data['Irrigation'].unique())
cultivars = sorted(data['Cultivar'].unique())

# Convert factors to categorical
data['Block'] = data['Block'].astype('category')
data['Nitrogen'] = data['Nitrogen'].astype('category')
data['Irrigation'] = data['Irrigation'].astype('category')
data['Cultivar'] = data['Cultivar'].astype('category')

print("\n" + "=" * 80)
print("STATISTICAL ANALYSIS - RCBD FACTORIAL DESIGN")
print("=" * 80)
print(f"\nExperimental Design:")
print(f"  Nitrogen levels: {nitrogen_levels}")
print(f"  Irrigation levels: {irrigation_levels}")
print(f"  Cultivars: {cultivars}")
print(f"  Blocks/Replicates: {sorted(data['Block'].unique())}")
print(f"  Total treatment combinations: {len(nitrogen_levels) * len(irrigation_levels) * len(cultivars)}")
print(f"  Total observations: {len(data)}")


# ============================================================================
# ANOVA Function
# ============================================================================

def rcbd_factorial_anova(data, trait):
    """Perform ANOVA for RCBD factorial design"""
    from scipy.stats import f

    # Get data
    y = data[trait].values
    n = len(y)
    grand_mean = np.mean(y)

    # Total SS
    ss_total = np.sum((y - grand_mean) ** 2)
    df_total = n - 1

    # Block SS
    block_means = data.groupby('Block')[trait].mean()
    n_per_block = data.groupby('Block').size()
    ss_block = np.sum(n_per_block * (block_means - grand_mean) ** 2)
    df_block = len(block_means) - 1

    # Treatment effects
    n_nitrogen = data.groupby('Nitrogen')[trait].mean()
    n_irrigation = data.groupby('Irrigation')[trait].mean()
    n_cultivar = data.groupby('Cultivar')[trait].mean()

    # Main effect SS
    ss_nitrogen = len(data) / len(n_nitrogen) / len(irrigation_levels) / len(cultivars) * \
                  np.sum((n_nitrogen - grand_mean) ** 2)
    df_nitrogen = len(n_nitrogen) - 1

    ss_irrigation = len(data) / len(nitrogen_levels) / len(n_irrigation) / len(cultivars) * \
                    np.sum((n_irrigation - grand_mean) ** 2)
    df_irrigation = len(n_irrigation) - 1

    ss_cultivar = len(data) / len(nitrogen_levels) / len(irrigation_levels) / len(n_cultivar) * \
                  np.sum((n_cultivar - grand_mean) ** 2)
    df_cultivar = len(n_cultivar) - 1

    # Interaction effects
    # N x I
    ni_means = data.groupby(['Nitrogen', 'Irrigation'])[trait].mean()
    ni_effect = ni_means - grand_mean
    ss_ni = len(data) / len(ni_means) / len(cultivars) * np.sum(ni_effect ** 2) - ss_nitrogen - ss_irrigation
    df_ni = df_nitrogen * df_irrigation

    # N x C
    nc_means = data.groupby(['Nitrogen', 'Cultivar'])[trait].mean()
    nc_effect = nc_means - grand_mean
    ss_nc = len(data) / len(nc_means) / len(irrigation_levels) * np.sum(nc_effect ** 2) - ss_nitrogen - ss_cultivar
    df_nc = df_nitrogen * df_cultivar

    # I x C
    ic_means = data.groupby(['Irrigation', 'Cultivar'])[trait].mean()
    ic_effect = ic_means - grand_mean
    ss_ic = len(data) / len(ic_means) / len(nitrogen_levels) * np.sum(ic_effect ** 2) - ss_irrigation - ss_cultivar
    df_ic = df_irrigation * df_cultivar

    # N x I x C
    nic_means = data.groupby(['Nitrogen', 'Irrigation', 'Cultivar'])[trait].mean()
    nic_effect = nic_means - grand_mean
    ss_nic = len(data) / len(nic_means) * np.sum(
        nic_effect ** 2) - ss_nitrogen - ss_irrigation - ss_cultivar - ss_ni - ss_nc - ss_ic
    df_nic = df_nitrogen * df_irrigation * df_cultivar

    # Error SS
    ss_error = ss_total - ss_block - ss_nitrogen - ss_irrigation - ss_cultivar - ss_ni - ss_nc - ss_ic - ss_nic
    df_error = df_total - df_block - df_nitrogen - df_irrigation - df_cultivar - df_ni - df_nc - df_ic - df_nic

    # Mean squares
    ms_block = ss_block / df_block if df_block > 0 else 0
    ms_nitrogen = ss_nitrogen / df_nitrogen
    ms_irrigation = ss_irrigation / df_irrigation
    ms_cultivar = ss_cultivar / df_cultivar
    ms_ni = ss_ni / df_ni if df_ni > 0 else 0
    ms_nc = ss_nc / df_nc if df_nc > 0 else 0
    ms_ic = ss_ic / df_ic if df_ic > 0 else 0
    ms_nic = ss_nic / df_nic if df_nic > 0 else 0
    ms_error = ss_error / df_error

    # F-statistics
    f_nitrogen = ms_nitrogen / ms_error
    f_irrigation = ms_irrigation / ms_error
    f_cultivar = ms_cultivar / ms_error
    f_ni = ms_ni / ms_error if ms_ni > 0 else 0
    f_nc = ms_nc / ms_error if ms_nc > 0 else 0
    f_ic = ms_ic / ms_error if ms_ic > 0 else 0
    f_nic = ms_nic / ms_error if ms_nic > 0 else 0

    # P-values
    p_nitrogen = 1 - f.cdf(f_nitrogen, df_nitrogen, df_error)
    p_irrigation = 1 - f.cdf(f_irrigation, df_irrigation, df_error)
    p_cultivar = 1 - f.cdf(f_cultivar, df_cultivar, df_error)
    p_ni = 1 - f.cdf(f_ni, df_ni, df_error) if f_ni > 0 else 1
    p_nc = 1 - f.cdf(f_nc, df_nc, df_error) if f_nc > 0 else 1
    p_ic = 1 - f.cdf(f_ic, df_ic, df_error) if f_ic > 0 else 1
    p_nic = 1 - f.cdf(f_nic, df_nic, df_error) if f_nic > 0 else 1

    # Create ANOVA table
    anova_table = pd.DataFrame({
        'Source': ['Block', 'Nitrogen', 'Irrigation', 'Cultivar', 'N×I', 'N×C', 'I×C', 'N×I×C', 'Error', 'Total'],
        'DF': [df_block, df_nitrogen, df_irrigation, df_cultivar, df_ni, df_nc, df_ic, df_nic, df_error, df_total],
        'SS': [ss_block, ss_nitrogen, ss_irrigation, ss_cultivar, ss_ni, ss_nc, ss_ic, ss_nic, ss_error, ss_total],
        'MS': [ms_block, ms_nitrogen, ms_irrigation, ms_cultivar, ms_ni, ms_nc, ms_ic, ms_nic, ms_error, np.nan],
        'F': [np.nan, f_nitrogen, f_irrigation, f_cultivar, f_ni, f_nc, f_ic, f_nic, np.nan, np.nan],
        'P-value': [np.nan, p_nitrogen, p_irrigation, p_cultivar, p_ni, p_nc, p_ic, p_nic, np.nan, np.nan]
    })

    return anova_table


# ============================================================================
# Mean Comparison Function (LSD)
# ============================================================================

def lsd_test(data, trait, factor):
    """Perform LSD test for mean comparison"""
    from scipy.stats import t

    # Get means
    means = data.groupby(factor)[trait].agg(['mean', 'std', 'count'])
    means = means.sort_values('mean', ascending=False)

    # Calculate MSE from ANOVA
    y = data[trait].values
    grand_mean = np.mean(y)
    residuals = []

    for idx, row in data.iterrows():
        pred = data[data[factor] == row[factor]][trait].mean()
        residuals.append(row[trait] - pred)

    mse = np.sum(np.array(residuals) ** 2) / (len(data) - len(means))

    # LSD calculation
    n_per_group = data.groupby(factor).size().values[0]
    df_error = len(data) - len(means)
    t_crit = t.ppf(0.975, df_error)  # 95% confidence
    lsd = t_crit * np.sqrt(2 * mse / n_per_group)

    # Assign letters
    letters = []
    current_letter = 'a'
    for i, (idx, row) in enumerate(means.iterrows()):
        if i == 0:
            letters.append('a')
        else:
            if abs(row['mean'] - means.iloc[0]['mean']) <= lsd:
                letters.append('a')
            else:
                # Check difference from previous
                diff = abs(row['mean'] - means.iloc[i - 1]['mean'])
                if diff <= lsd:
                    letters.append(letters[i - 1])
                else:
                    current_letter = chr(ord(letters[i - 1]) + 1)
                    letters.append(current_letter)

    means['Letters'] = letters
    means = means.reset_index()

    return means, lsd


# ============================================================================
# Perform ANOVA for all traits
# ============================================================================

all_anova_results = {}
for trait in traits:
    print(f"\n{'=' * 80}")
    print(f"ANOVA for {trait}")
    print('=' * 80)
    anova_table = rcbd_factorial_anova(data, trait)
    print(anova_table.to_string(index=False))
    all_anova_results[trait] = anova_table

    # Export ANOVA table
    anova_table.to_csv(f'anova_{trait}.csv', index=False)

# ============================================================================
# Mean Comparisons
# ============================================================================

comparison_results = {}

# Main effects
for factor in ['Nitrogen', 'Irrigation', 'Cultivar']:
    comparison_results[factor] = {}
    for trait in traits:
        means_table, lsd_value = lsd_test(data, trait, factor)
        comparison_results[factor][trait] = (means_table, lsd_value)

        # Export
        means_table.to_csv(f'means_{factor}_{trait}.csv', index=False)

# Interactions
interactions = [
    ['Nitrogen', 'Irrigation'],
    ['Nitrogen', 'Cultivar'],
    ['Irrigation', 'Cultivar'],
    ['Nitrogen', 'Irrigation', 'Cultivar']
]

for interaction in interactions:
    int_name = '_x_'.join(interaction)
    comparison_results[int_name] = {}

    for trait in traits:
        means = data.groupby(interaction)[trait].agg(['mean', 'std', 'count']).reset_index()
        means = means.sort_values('mean', ascending=False)
        comparison_results[int_name][trait] = means

        # Export
        means.to_csv(f'means_{int_name}_{trait}.csv', index=False)

print("\n✓ All ANOVA tables and mean comparison tables exported")

# ============================================================================
# PART 3: VISUALIZATIONS
# ============================================================================

print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# 1. Main Effects - Bar plots with error bars
for factor in ['Nitrogen', 'Irrigation', 'Cultivar']:
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'Main Effect of {factor} on All Traits', fontsize=16, fontweight='bold')

    for idx, trait in enumerate(traits):
        ax = axes[idx // 4, idx % 4]

        means_table = comparison_results[factor][trait][0]

        ax.bar(range(len(means_table)), means_table['mean'],
               yerr=means_table['std'], capsize=5, alpha=0.7, color='steelblue')
        ax.set_xticks(range(len(means_table)))
        ax.set_xticklabels(means_table[factor], rotation=45 if factor == 'Nitrogen' else 0)
        ax.set_ylabel(trait.replace('_', ' '))
        ax.set_title(trait.replace('_', ' '))

        # Add letters
        for i, (idx_val, row) in enumerate(means_table.iterrows()):
            ax.text(i, row['mean'] + row['std'], row['Letters'],
                    ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'main_effect_{factor}.png', dpi=300, bbox_inches='tight')
    plt.close()

# 2. Box plots for main effects
for factor in ['Nitrogen', 'Irrigation', 'Cultivar']:
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'Distribution of Traits by {factor}', fontsize=16, fontweight='bold')

    for idx, trait in enumerate(traits):
        ax = axes[idx // 4, idx % 4]

        data.boxplot(column=trait, by=factor, ax=ax, patch_artist=True)
        ax.set_title(trait.replace('_', ' '))
        ax.set_xlabel(factor)
        ax.set_ylabel(trait.replace('_', ' '))
        plt.sca(ax)
        plt.xticks(rotation=45 if factor == 'Nitrogen' else 0)

    plt.tight_layout()
    plt.savefig(f'boxplot_{factor}.png', dpi=300, bbox_inches='tight')
    plt.close()

# 3. Bar charts for two-way interactions
for interaction in [['Nitrogen', 'Irrigation'], ['Nitrogen', 'Cultivar'],
                    ['Irrigation', 'Cultivar']]:
    int_name = '_x_'.join(interaction)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'Interaction: {" × ".join(interaction)} - Bar Charts',
                 fontsize=16, fontweight='bold')

    for idx, trait in enumerate(traits):
        ax = axes[idx // 4, idx % 4]

        # Prepare data for grouped bar chart
        means_int = data.groupby(interaction)[trait].mean().reset_index()
        std_int = data.groupby(interaction)[trait].std().reset_index()

        # Get unique levels
        level1_vals = means_int[interaction[0]].unique()
        level2_vals = means_int[interaction[1]].unique()

        # Bar positions
        x = np.arange(len(level1_vals))
        width = 0.8 / len(level2_vals)

        # Plot bars
        for i, level2 in enumerate(level2_vals):
            mask = means_int[interaction[1]] == level2
            heights = means_int[mask][trait].values
            errors = std_int[mask][trait].values

            positions = x + (i - len(level2_vals) / 2 + 0.5) * width
            ax.bar(positions, heights, width, label=str(level2),
                   yerr=errors, capsize=3, alpha=0.8)

        ax.set_xlabel(interaction[0])
        ax.set_ylabel(trait.replace('_', ' '))
        ax.set_title(trait.replace('_', ' '))
        ax.set_xticks(x)
        ax.set_xticklabels(level1_vals)
        ax.legend(title=interaction[1], loc='best')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'bar_interaction_{int_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

# 4. Interaction line plots (traditional interaction plots)
for interaction in [['Nitrogen', 'Irrigation'], ['Nitrogen', 'Cultivar'],
                    ['Irrigation', 'Cultivar']]:
    int_name = '_x_'.join(interaction)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'Interaction: {" × ".join(interaction)} - Line Plots',
                 fontsize=16, fontweight='bold')

    for idx, trait in enumerate(traits):
        ax = axes[idx // 4, idx % 4]

        pivot_data = data.pivot_table(values=trait, index=interaction[0],
                                      columns=interaction[1], aggfunc='mean')

        for col in pivot_data.columns:
            ax.plot(pivot_data.index, pivot_data[col], marker='o',
                    label=str(col), linewidth=2, markersize=8)

        ax.set_xlabel(interaction[0])
        ax.set_ylabel(trait.replace('_', ' '))
        ax.set_title(trait.replace('_', ' '))
        ax.legend(title=interaction[1], loc='best')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'line_interaction_{int_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

# 5. Three-way interaction - Bar charts (one per trait)
for trait in traits:
    fig, ax = plt.subplots(figsize=(16, 6))
    fig.suptitle(f'{trait.replace("_", " ")} - Three-way Interaction: Nitrogen × Irrigation × Cultivar',
                 fontsize=14, fontweight='bold')

    # Prepare data
    means_3way = data.groupby(['Nitrogen', 'Irrigation', 'Cultivar'])[trait].mean().reset_index()
    std_3way = data.groupby(['Nitrogen', 'Irrigation', 'Cultivar'])[trait].std().reset_index()

    # Define colors and patterns
    color_map = {'Cultivar_A': '#2E86AB', 'Cultivar_B': '#A23B72'}
    hatch_map = {'50% FC': '', '75% FC': '///'}

    x_pos = 0
    x_ticks = []
    x_tick_labels = []
    bar_positions = []

    for n_idx, n in enumerate(nitrogen_levels):
        if n_idx > 0:
            x_pos += 1.0  # Add space between nitrogen levels

        group_start = x_pos

        for irr_idx, irr in enumerate(irrigation_levels):
            for cult_idx, cult in enumerate(cultivars):
                mask = (means_3way['Nitrogen'] == n) & \
                       (means_3way['Irrigation'] == irr) & \
                       (means_3way['Cultivar'] == cult)

                if mask.any():
                    height = means_3way[mask][trait].values[0]
                    error = std_3way[mask][trait].values[0]

                    # Plot bar with color (cultivar) and hatch (irrigation)
                    ax.bar(x_pos, height, yerr=error, capsize=3,
                           color=color_map[cult], edgecolor='black', linewidth=1.5,
                           hatch=hatch_map[irr], alpha=0.8, width=0.7)

                    bar_positions.append(x_pos)
                    x_pos += 0.8

        # Mark center of each nitrogen group
        group_center = (group_start + x_pos - 0.8) / 2
        x_ticks.append(group_center)
        x_tick_labels.append(f'{n} kg/ha')

    ax.set_ylabel(trait.replace('_', ' '), fontsize=12, fontweight='bold')
    ax.set_xlabel('Nitrogen Level', fontsize=12, fontweight='bold')
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels, fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # Create custom legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor='#2E86AB', label='Cultivar A', edgecolor='black'),
        Patch(facecolor='#A23B72', label='Cultivar B', edgecolor='black'),
        Patch(facecolor='white', label='50% FC', edgecolor='black', hatch=''),
        Patch(facecolor='white', label='75% FC', edgecolor='black', hatch='///')
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=10,
              title='Cultivar | Irrigation', title_fontsize=10)

    plt.tight_layout()
    plt.savefig(f'bar_3way_{trait}.png', dpi=300, bbox_inches='tight')
    plt.close()

# 6. Three-way interaction - Line plots (faceted by Nitrogen)
for trait in traits:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'{trait.replace("_", " ")} - Three-way Interaction (Line Plots by Nitrogen)',
                 fontsize=14, fontweight='bold')

    for idx, n_level in enumerate(nitrogen_levels):
        ax = axes[idx]

        subset = data[data['Nitrogen'] == n_level]

        for cult in cultivars:
            cult_data = subset[subset['Cultivar'] == cult]
            means = cult_data.groupby('Irrigation')[trait].mean()

            ax.plot(means.index, means.values, marker='o',
                    label=cult, linewidth=2, markersize=10)

        ax.set_xlabel('Irrigation Level')
        ax.set_ylabel(trait.replace('_', ' '))
        ax.set_title(f'Nitrogen: {n_level} kg/ha')
        ax.legend(title='Cultivar', loc='best')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'line_3way_by_nitrogen_{trait}.png', dpi=300, bbox_inches='tight')
    plt.close()

# 7. Three-way interaction - Line plots (faceted by Irrigation)
for trait in traits:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{trait.replace("_", " ")} - Three-way Interaction (Line Plots by Irrigation)',
                 fontsize=14, fontweight='bold')

    for idx, irr_level in enumerate(irrigation_levels):
        ax = axes[idx]

        subset = data[data['Irrigation'] == irr_level]

        for cult in cultivars:
            cult_data = subset[subset['Cultivar'] == cult]
            means = cult_data.groupby('Nitrogen')[trait].mean()

            ax.plot(means.index, means.values, marker='o',
                    label=cult, linewidth=2, markersize=10)

        ax.set_xlabel('Nitrogen Level (kg/ha)')
        ax.set_ylabel(trait.replace('_', ' '))
        ax.set_title(f'Irrigation: {irr_level}')
        ax.legend(title='Cultivar', loc='best')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'line_3way_by_irrigation_{trait}.png', dpi=300, bbox_inches='tight')
    plt.close()

# 8. Three-way interaction - Line plots (faceted by Cultivar)
for trait in traits:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{trait.replace("_", " ")} - Three-way Interaction (Line Plots by Cultivar)',
                 fontsize=14, fontweight='bold')

    for idx, cult in enumerate(cultivars):
        ax = axes[idx]

        subset = data[data['Cultivar'] == cult]

        for irr_level in irrigation_levels:
            irr_data = subset[subset['Irrigation'] == irr_level]
            means = irr_data.groupby('Nitrogen')[trait].mean()

            ax.plot(means.index, means.values, marker='o',
                    label=irr_level, linewidth=2, markersize=10)

        ax.set_xlabel('Nitrogen Level (kg/ha)')
        ax.set_ylabel(trait.replace('_', ' '))
        ax.set_title(f'{cult}')
        ax.legend(title='Irrigation', loc='best')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'line_3way_by_cultivar_{trait}.png', dpi=300, bbox_inches='tight')
    plt.close()

# 9. Heatmap for three-way interaction - REMOVED (too many small heatmaps)

print("✓ All visualization plots saved")

# ============================================================================
# PART 4: CORRELATION ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("CORRELATION ANALYSIS")
print("=" * 80)

# Calculate Pearson correlation
trait_data = data[traits]
correlation_matrix = trait_data.corr(method='pearson')

print("\nCorrelation Matrix:")
print(correlation_matrix)

# Export correlation matrix
correlation_matrix.to_csv('correlation_matrix.csv')


# Calculate p-values for correlations
def calculate_pvalues(df):
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            pvalues[r][c] = stats.pearsonr(df[r], df[c])[1]
    return pvalues


pvalues = calculate_pvalues(trait_data)
pvalues.to_csv('correlation_pvalues.csv')

# Visualization - ONE comprehensive correlation heatmap with significance
fig, ax = plt.subplots(figsize=(14, 12))

# Create annotation with significance stars
annot = np.empty_like(correlation_matrix, dtype=object)
for i in range(correlation_matrix.shape[0]):
    for j in range(correlation_matrix.shape[1]):
        corr_val = correlation_matrix.iloc[i, j]
        p_val = pvalues.iloc[i, j]

        if i == j:
            annot[i, j] = '1.00'
        else:
            if p_val < 0.001:
                sig = '***'
            elif p_val < 0.01:
                sig = '**'
            elif p_val < 0.05:
                sig = '*'
            else:
                sig = ''

            annot[i, j] = f'{corr_val:.2f}{sig}'

sns.heatmap(correlation_matrix, annot=annot, fmt='',
            cmap='coolwarm', center=0, square=True, linewidths=1.5,
            cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient"},
            ax=ax, vmin=-1, vmax=1)

ax.set_title('Correlation Matrix of Canola Traits with Significance Levels\n(*** p<0.001, ** p<0.01, * p<0.05)',
             fontsize=14, fontweight='bold', pad=20)

# Rotate labels for better readability
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Correlation analysis completed and exported")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE - SUMMARY OF OUTPUTS")
print("=" * 80)
print("\nGenerated Files:")
print("\nData Files:")
print("  • canola_experiment_data.csv - Raw experimental data")
print("\nANOVA Tables:")
for trait in traits:
    print(f"  • anova_{trait}.csv")
print("\nMean Comparison Tables:")
print("  - Main effects: means_[Factor]_[Trait].csv")
print("  - Interactions: means_[Factor1]_x_[Factor2]_[Trait].csv")
print("\nVisualization Plots:")
print("  • main_effect_[Factor].png - Bar plots with letters for main effects")
print("  • boxplot_[Factor].png - Box plots for main effects")
print("  • bar_interaction_[Factor1]_x_[Factor2].png - Bar charts for 2-way interactions")
print("  • line_interaction_[Factor1]_x_[Factor2].png - Line plots for 2-way interactions")
print("  • bar_3way_[Trait].png - Bar chart for 3-way interaction (one per trait)")
print("  • line_3way_by_nitrogen_[Trait].png - 3-way interaction lines faceted by Nitrogen")
print("  • line_3way_by_irrigation_[Trait].png - 3-way interaction lines faceted by Irrigation")
print("  • line_3way_by_cultivar_[Trait].png - 3-way interaction lines faceted by Cultivar")
print("\nCorrelation Files:")
print("  • correlation_matrix.csv - Pearson correlation coefficients")
print("  • correlation_pvalues.csv - P-values for correlations")
print("  • correlation_heatmap.png - Main correlation visualization with significance levels")
print("\n" + "=" * 80)
print("All analyses completed successfully!")
print("=" * 80)
