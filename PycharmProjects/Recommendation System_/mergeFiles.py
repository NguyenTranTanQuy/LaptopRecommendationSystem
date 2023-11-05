import os
import pandas as pd
import glob
import random

current_file = __file__
f = os.path.dirname(os.path.abspath(current_file))


def mergeFiles():
    excel_files = glob.glob(f + '/datasets/*.xlsx')

    # Create an empty DataFrame to store the merged data
    merged_data = pd.DataFrame()

    # Loop through each Excel file and append its data to the merged data DataFrame
    for file in excel_files:
        data = pd.read_excel(file)
        merged_data = merged_data._append(data)

    # Get the number of rows
    n_rows = len(merged_data)

    # Generate a list of random indices
    indices = list(range(n_rows))
    random.shuffle(indices)

    # Reorder the dataframe by the random indices
    merged_data = merged_data.iloc[indices]

    # Write the merged data to a new Excel file
    merged_data.to_csv(f + '/datasets/Laptops.csv', encoding='utf-8', index=False)
