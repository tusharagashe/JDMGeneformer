from geneformer import TranscriptomeTokenizer


tk = TranscriptomeTokenizer(
  custom_attr_name_dict={
    "disease_group": "disease_group",
    "sex": "sex",
    "self_reported_ethnicity": "self_reported_ethnicity",
    "donor_id": "donor_id",
    "observation_joinid": "observation_joinid"
  },
  nproc=16
)


tk.tokenize_data(
    "dataset",
    "tokenized_dataset",
    "tokenized",
    file_format = "h5ad"  
)