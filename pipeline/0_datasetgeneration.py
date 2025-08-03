import scanpy as sc
import numpy as np
import cellxgene_census
import anndata
from typing import List
import os

def get_and_save_data(dataset_id: str, cell_types: List[str], output_filename: str):
    """
    Queries the CELLxGENE Census API for specific cell types from a given dataset
    and saves the filtered data to a h5ad file.

    Args:
        dataset_id: The UUID of the dataset to query.
        cell_types: A list of cell type names to filter for.
        output_filename: The name of the h5ad file to save the data to.
    """
    # Construct the filter string
    # Assuming the column name for cell types is 'cell_type'
    # Check the obs table of your dataset if you get a SOMAError
    filter_string = f"dataset_id == '{dataset_id}' and cell_type in {cell_types}"
    
    print(f"Querying Census with filter: {filter_string}")

    try:
        with cellxgene_census.open_soma() as census:
            adata_filtered = cellxgene_census.get_anndata(
                census,
                organism="homo_sapiens",
                obs_value_filter=filter_string,
                measurement_name="RNA"
            )

        print(f"Retrieved AnnData object with {adata_filtered.n_obs} cells and {adata_filtered.n_vars} genes.")

        # Save the filtered AnnData object to a h5ad file
        adata_filtered.write(output_filename)
        print(f"Filtered data saved to {output_filename}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please check your dataset_id and cell_type names for accuracy.")



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


def main():
    
    dataset_id = "a199ca73-035d-44e2-9893-4c493151db21"
    desired_cell_types = [
        'CD4-positive, CD25-positive, alpha-beta regulatory T cell',
        'effector CD4-positive, alpha-beta T cell',
        'naive thymus-derived CD4-positive, alpha-beta T cell'
    ]
    file_name = "CD4_alltypes.h5ad"
    output_location = f"../dataset/{file_name}"

    output_directory = os.path.dirname(output_location)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
    
    get_and_save_data(dataset_id, desired_cell_types, output_location)
    
    prepare_h5ad_for_geneformer(
        input_path=f"../dataset/{file_name}",
        output_path="../dataset/CD4_prepped.h5ad",
    )

if __name__ == "__main__":
    main()

