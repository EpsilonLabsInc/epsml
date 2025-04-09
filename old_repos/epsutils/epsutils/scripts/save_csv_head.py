import pandas as pd

input_file = r"C:\Users\Andrej\Desktop\cleaned_CR_labels_for_binary_classification_GRADIENT_CR_ALL_CHEST_BATCHES_cleaned_no_findings_labels.csv"
output_file = r"C:\Users\Andrej\Desktop\no_findings_subset.csv"

df = pd.read_csv(input_file)
df_subset = df.head(1001)
df_subset.to_csv(output_file, index=False)
