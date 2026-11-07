import argparse
import pandas as pd
from chembl_webresource_client.new_client import new_client


def download_chembl_activities(target_chembl_id, output_file):
    target = new_client.target
    target_data = target.get(target_chembl_id)

    if not target_data:
        print(f"No target found with ID: {target_chembl_id}")
        return

    print(f"Target: {target_data['target_chembl_id']} - {target_data['pref_name']}")

    activity = new_client.activity
    activities = activity.filter(
        target_chembl_id=target_chembl_id,
        standard_type__in=["IC50", "Ki", "EC50", "AC50", "Kd"],
    )

    # Convert to DataFrame
    activities_list = list(activities)
    df = pd.DataFrame(activities_list)

    # Print available columns for debugging
    print(f"Available columns: {list(df.columns)}")

    # Map ChEMBL API field names to your required columns
    column_mapping = {
        "molecule_chembl_id": "Molecule ChEMBL ID",
        "canonical_smiles": "Smiles",
        "standard_type": "Standard Type",
        "pchembl_value": "pChEMBL Value",
    }

    # Find compound key - it might be named differently
    compound_key_col = None
    for col in df.columns:
        if "key" in col.lower() or "compound" in col.lower():
            compound_key_col = col
            break

    if compound_key_col:
        column_mapping[compound_key_col] = "Compound Key"

    # Select only the columns we need
    needed_api_cols = list(column_mapping.keys())
    available_cols = [c for c in needed_api_cols if c in df.columns]
    df = df[available_cols]

    # Rename columns
    df = df.rename(columns=column_mapping)

    # Ensure we have all 5 required columns
    required_cols = [
        "Molecule ChEMBL ID",
        "Compound Key",
        "Smiles",
        "Standard Type",
        "pChEMBL Value",
    ]
    for col in required_cols:
        if col not in df.columns:
            print(f"Warning: Missing column '{col}' - adding empty column")
            df[col] = ""

    # Reorder to match original
    df = df[required_cols]

    # Save with semicolon delimiter
    df.to_csv(output_file, index=False, sep=";")
    print(f"Saved {len(df)} activities to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download ChEMBL activity data for a target"
    )
    parser.add_argument("target_id", help="ChEMBL target ID (e.g., CHEMBL215)")
    parser.add_argument("-o", "--output", help="Output CSV filename", required=True)
    args = parser.parse_args()

    download_chembl_activities(args.target_id, args.output)
