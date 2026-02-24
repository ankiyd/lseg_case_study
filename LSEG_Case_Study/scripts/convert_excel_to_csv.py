import pandas as pd
from pathlib import Path


def convert_excel_to_csv():
    # Define base project path
    base_path = Path(__file__).resolve().parent.parent
    
    input_file = base_path / "data" / "raw" / "Westworld Group Dataset v1.xlsx"
    
    # Output paths
    hris_output = base_path / "data" / "raw" / "hris_extract.csv"
    exit_output = base_path / "data" / "raw" / "exit_survey.csv"
    
    print("Reading Excel file...")
    
    # Read HRIS sheet
    hris_df = pd.read_excel(input_file, sheet_name="HRIS Extract")
    hris_df.to_csv(hris_output, index=False)
    print(f"HRIS extract saved: {hris_output} | Shape: {hris_df.shape}")
    
    # Read Exit Survey sheet
    exit_df = pd.read_excel(input_file, sheet_name="Exit Survey")
    exit_df.to_csv(exit_output, index=False)
    print(f"Exit survey saved: {exit_output} | Shape: {exit_df.shape}")
    
    print("Conversion complete.")


if __name__ == "__main__":
    convert_excel_to_csv()

