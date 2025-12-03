import seaborn as sns
import matplotlib.pyplot as plt

def generate_heatmap(df):
    corr_matrix = df.corr()
    target_corr = corr_matrix['goal'].abs().sort_values(ascending=False)
    top_features = target_corr.head(15).index
    filtered_corr = df[top_features].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(filtered_corr, 
                annot=True, 
                fmt=".2f", 
                cmap='coolwarm', 
                vmin=-1, 
                vmax=1)

    plt.title("Top Correlations with 'Goal'")
    plt.show()
    plt.savefig("heatmap.png", dpi=300, bbox_inches='tight')
    print("Heatmap saved as heatmap.png")