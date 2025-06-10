import pickle
import sys
import pandas as pd
from collections import defaultdict

def print_cutflow_tabular(filename):
    try:
        with open(filename, 'rb') as f:
            output = pickle.load(f)
        
        # Extract the era (like 'UL18NanoAODv9') and the corresponding data
        for era, data in output.items():
            print(f"Era: {era}")
            
            # Separate the entries into 'gen' and 'reco' based on the keys
            gen_data = {}
            reco_data = {}
            for key, value in data[next(iter(data))].items():
                if 'gen' in key or 'Gen' in key:
                    gen_data[key] = value
                elif 'reco' in key or 'Reco' in key:
                    reco_data[key] = value
                else:
                    # Include keys without 'gen' or 'reco' in both tables for reference
                    gen_data[key] = value
                    reco_data[key] = value
            
            # Convert to DataFrames for easier tabular display
            gen_df = pd.DataFrame(list(gen_data.items()), columns=['Cut', 'Gen Count'])
            reco_df = pd.DataFrame(list(reco_data.items()), columns=['Cut', 'Reco Count'])
            
            # Merge the two dataframes on the 'Cut' column to create a tabular view with Gen and Reco counts
            cutflow_df = pd.merge(gen_df, reco_df, on="Cut", how="outer").fillna("")
            
            # Display the DataFrame
            print(cutflow_df.to_string(index=False))
            print("\n" + "-"*50 + "\n")
            
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <filename>")
    else:
        filename = sys.argv[1]
        print_cutflow_tabular(filename)
