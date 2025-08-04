import scanpy as sc
import numpy as np
import cellxgene_census
import anndata
from typing import List
import os
import path

def get_and_save_data(dataset_id: str, cell_types: List[str], output_location: str, cell_type_column: str = "cell_type"):
    """
    Downloads a full dataset from the CELLxGENE Census, filters it for
    specific cell types, and saves the filtered data to a new h5ad file.
    The original downloaded file is deleted after successful processing.

    Args:
        dataset_id: The UUID of the dataset to query.
        cell_types: A list of cell type names to filter for.
        output_location: The full path to save the final filtered h5ad file.
        cell_type_column: The name of the column containing cell type information in adata.obs.
                          Defaults to 'cell_type'.
    """
    # Create the output directory if it doesn't exist
    output_directory = os.path.dirname(output_location)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
    
    # Define a temporary path for the full downloaded file
    temp_input_path = os.path.join(output_directory, "temp_dataset")
    
    # --- Step 1: Download the full h5ad file from the Census ---
    if not os.path.exists(temp_input_path):
        print(f"Downloading full dataset for ID: {dataset_id}...")
        try:
            with cellxgene_census.open_soma() as census:
                cellxgene_census.download_source_h5ad(
                    dataset_id=dataset_id,
                    to_path=temp_input_path
                )
            print("Download complete.")
        except Exception as e:
            print(f"Error during download: {e}")
            return
    else:
        print("Full dataset already exists. Skipping download.")

    # --- Step 2: Load and filter the local file with Scanpy ---
    print("Loading local h5ad file and filtering with Scanpy...")
    try:
        adata = sc.read_h5ad(temp_input_path)
        print(f"Loaded AnnData with {adata.n_obs} cells and {adata.n_vars} genes.")
        
        # Filter the AnnData object to keep only the desired cell types
        # This will use the cell_type_column parameter
        filtered_adata = adata[adata.obs[cell_type_column].isin(cell_types)].copy()
        print(f"Filtered data to {filtered_adata.n_obs} cells.")

        # --- Step 3: Save the filtered data ---
        filtered_adata.write(output_location)
        print(f"Filtered data saved to {output_location}")

        # --- Step 4: Delete the original downloaded file ---
        os.remove(temp_input_path)
        print(f"Successfully deleted original downloaded file: {temp_input_path}")

    except FileNotFoundError:
        print(f"Error: Temporary file {temp_input_path} not found.")
    except KeyError:
        print(f"Error: The column '{cell_type_column}' was not found in the dataset.")
    except Exception as e:
        print(f"An error occurred during filtering and saving: {e}")



def prepare_h5ad_for_geneformer(input_path, output_path):
    adata = sc.read_h5ad(input_path)

    print(f"Loaded AnnData with {adata.n_obs} cells and {adata.n_vars} genes.")

    # 1. Add n_counts to obs
    if 'n_counts' not in adata.obs.columns:
        print("Adding 'n_counts' to adata.obs...")
        if hasattr(adata.X, 'sum'):
            adata.obs['n_counts'] = np.array(adata.X.sum(axis=1)).flatten()
        else:
            adata.obs['n_counts'] = adata.X.sum(axis=1)

    # 2. Add ensembl_id to var
    if 'ensembl_id' not in adata.var.columns:
        print("Adding 'ensembl_id' column by copying var_names...")
        adata.var['ensembl_id'] = adata.var_names

    # 3. Remove genes with missing ensembl_id (Geneformer requirement)
    pre_filter_genes = adata.n_vars
    adata = adata[:, ~adata.var['ensembl_id'].isna()]
    print(f"Filtered out {pre_filter_genes - adata.n_vars} genes with missing Ensembl IDs.")

    # 4. Save cleaned dataset
    adata.write(output_path)
    print(f"Saved cleaned h5ad file to {output_path}")

    # 5. Delete the original input file
    try:
        os.remove(input_path)
        print(f"Successfully deleted original input file: {input_path}")
    except FileNotFoundError:
        print(f"Error: Input file {input_path} not found for deletion.")
    except Exception as e:
        print(f"An error occurred while deleting {input_path}: {e}")


def main():
    
    data_id = "a199ca73-035d-44e2-9893-4c493151db21"
    desired_cell_types = [
        'CD4-positive, CD25-positive, alpha-beta regulatory T cell',
        'effector CD4-positive, alpha-beta T cell',
        'naive thymus-derived CD4-positive, alpha-beta T cell'
    ]
    file_name = "CD4_alltypes.h5ad"
    output_location = f"../dataset/{file_name}"
    
    # You may need to change this if the authored category has a different name
    cell_type_column = "cell_type" 

    get_and_save_data(data_id, desired_cell_types, output_location, cell_type_column)
    
    prepare_h5ad_for_geneformer(
        input_path=f"../dataset/{file_name}",
        output_path="../dataset/CD4_prepped.h5ad",
    )

if __name__ == "__main__":
    main()

