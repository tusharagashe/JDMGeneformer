import scanpy as sc
import numpy as np

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

# Example usage
prepare_h5ad_for_geneformer(
    input_path="dataset/CD4.h5ad",
    output_path="dataset/CD4_prepped.h5ad",
)
