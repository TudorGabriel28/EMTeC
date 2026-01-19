import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

# ==========================================
# 1. LOAD DATA & PREPROCESSING
# ==========================================
df = pd.read_csv('reading_measures_corrected.csv', sep='\t')

print(f"Data Loaded. Rows: {len(df)}")

controls = ['word_length_without_punct', 'zipf_freq']

# ==========================================
# 2. COMPARE LLM PREDICTIVE POWER
# ==========================================
llm_columns = [
    'surprisal_gpt2',
    'surprisal_mistral-instruct',
    'surprisal_p_llama2-7b',
    'surprisal_phi2',
    'surprisal_pythia-6.9b'
]

print("\n--- RESULTS 1: WHICH LLM BEST PREDICTS READING TIME? ---")

results = []

for llm in llm_columns:
    if llm in df.columns:
        data_subset = df[controls + [llm] + ['FPRT']].dropna()

        X = data_subset[controls + [llm]]
        y = data_subset['FPRT']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)

        surprisal_idx = 2
        importance_mean = perm_importance.importances_mean[surprisal_idx]

        results.append({
            'Model_Source': llm,
            'Surprisal_Importance': importance_mean,
            'R_squared': model.score(X_test, y_test),
            'Effect_of_Surprisal': model.coef_[-1],
        })

imp_df = pd.DataFrame(results).sort_values(by='Surprisal_Importance', ascending=False)
print(imp_df)

plt.figure(figsize=(10, 6))
sns.barplot(data=imp_df, x='Model_Source', y='R_squared', palette='magma')
plt.title('Predictive Power of Different LLMs on Human Reading Time')
plt.xticks(rotation=45)
plt.ylabel('R-Squared')
plt.tight_layout()
plt.savefig("r2_llms.png")
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(data=imp_df, x='Model_Source', y='Surprisal_Importance', palette='magma')
plt.title('Feature Importance of Surprisal across LLMs', fontsize=14)
plt.ylabel('Importance', fontsize=12)
plt.xlabel('LLM Source', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("surprisal_importance_llms.png")
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(
    data=imp_df,
    x='Model_Source',
    y='Effect_of_Surprisal',
    palette='viridis'
)
plt.title('Effect of Surprisal on Reading Speed by LLM Model', fontsize=14)
plt.ylabel('Coefficient', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("surprisal_effect_model_type.png")
plt.show()


# ==========================================
# 3. ANALYZE BY TEXT TYPE
# ==========================================
print(f"\n--- RESULTS 2: SURPRISAL EFFECT BY TEXT TYPE ---")

def standardize_genre(row):
    val = str(row['task']).lower().strip()

    if 'non-fiction' in val: return 'Non-fiction'
    if 'fiction' in val and 'non' not in val: return 'Fiction'
    if 'poetry' in val or 'poem' in val: return 'Poetry'
    if 'summarization' in val: return 'Summarization'
    if 'synopsis' in val: return 'Synopsis'
    if 'words_given' in val: return 'Key words'
    return val.capitalize()

if 'task' in df.columns:
    df['plot_genre'] = df.apply(standardize_genre, axis=1)
else:
    print("Error: 'task' column not found for genre analysis.")

target_order = ['Non-fiction', 'Fiction', 'Poetry', 'Summarization', 'Synopsis', 'Key words']
target_surprisal = 'surprisal_gpt2' # Using GPT-2 as the baseline metric
genre_results = []

for genre in target_order:
    subset = df[df['plot_genre'] == genre]

    if len(subset) > 100:
        X_sub = subset[controls + [target_surprisal]]
        X_sub = sm.add_constant(X_sub)
        y_sub = subset['FPRT']

        model_sub = sm.OLS(y_sub, X_sub).fit()

        genre_results.append({
            'Genre': genre,
            'Effect_of_Surprisal': model_sub.params[target_surprisal],
            'R_squared': model_sub.rsquared
        })

genre_df = pd.DataFrame(genre_results)
print(genre_df)

if not genre_df.empty:
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=genre_df,
        x='Genre',
        y='Effect_of_Surprisal',
        order=target_order,
        palette='viridis'
    )
    plt.axvline(x=2.5, color='gray', linestyle='--', linewidth=2, alpha=0.7)
    plt.title('Effect of Surprisal on Reading Speed by Text Type', fontsize=14)
    plt.ylabel('Coefficient', fontsize=12)
    plt.xlabel('')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("surprisal_effect_text_type.png")
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=genre_df,
        x='Genre',
        y='R_squared',
        order=target_order,
        palette='magma'
    )
    plt.axvline(x=2.5, color='gray', linestyle='--', linewidth=2, alpha=0.7)
    plt.title('Predictive Power by Text Type', fontsize=14)
    plt.ylabel('R-Squared', fontsize=12)
    plt.xlabel('')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("r2_text_type.png")
    plt.show()

