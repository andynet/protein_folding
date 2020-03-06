# configfile: "config.yaml"
# 
# include: ""
# 
# wildcard_constraints:
#     bin_size=""
# 
# onerror:
#     print("An error occurred")
#     shell("mail -s 'an error occurred' andrejbalaz001@gmail.com < {log}")

rule predict_alphafold:
    input:
        "{data_dir}/prospr/models/{seed}",
        "{data_dir}/prospr/stats/{seed}",
    output:
        

rule train_alphafold:
    input:
        "{data_dir}/prospr/tensors/",
    output:
        "{data_dir}/prospr/models/{seed}/",
        "{data_dir}/prospr/stats/{seed}/",
    shell:
        """
        python scripts/alphafold_training.py
        """

rule create_tensors:
    input:
        "{data_dir}/prospr/dicts/",
        "{data_dir}/prospr/potts/",
    output:
        "{data_dir}/prospr/tensors/",
    shell:
        """
        python preprocess_data_v2.py
        """

rule download_prospr_data:
    output:
        "{data_dir}/prospr/dicts/",
        "{data_dir}/prospr/potts/",
    shell:
        """
        wget --recursive --no-parent --no-host-directories --cut-dirs=2 https://files.physics.byu.edu/data/prospr/potts/
        wget --recursive --no-parent --no-host-directories --cut-dirs=2 https://files.physics.byu.edu/data/prospr/dicts/
        """

