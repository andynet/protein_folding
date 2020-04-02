# configfile: "./configs/config.yaml"

include: "./scripts/input_functions.py"

# run with:
# snakemake --profile ./configs/slurm/ 
# I recommend running in background using GNU screen application

rule main:
    input:
        get_input,

rule make_X:
    input:
        "{base_dir}/data/our_input/potts/{domain}_FN.pt",
        "{base_dir}/data/our_input/potts/{domain}_h.pt",
        "{base_dir}/data/our_input/potts/{domain}_J.pt",
        "{base_dir}/data/our_input/pssm/{domain}_pssm.pt",
        "{base_dir}/data/our_input/sequences/{domain}.fasta",
    output:
        "{base_dir}/data/our_input/tensors/{domain}_X.pt",
    conda:
        "envs/input_generation.yml",
    shell:
        """
        python {wildcards.base_dir}/scripts/preprocess_our_data.py  \
            --fn_file {input[0]}                                    \
            --h_file {input[1]}                                     \
            --j_file {input[2]}                                     \
            --pssm_file {input[3]}                                  \
            --seq_file {input[4]}                                   \
            --output_file {output[0]}
        """

rule frobenius:
    input:
        "{base_dir}/data/our_input/temp/{domain}.couplings",
    output:
        "{base_dir}/data/our_input/potts/{domain}_FN.pt",
    conda:
        "envs/input_generation.yml",
    shell:
        """
        python {wildcards.base_dir}/scripts/pipeline_scripts/read_couplings.py \
            {input[0]}\
            {output[0]}
        """

rule hJ:
    input:
        "{base_dir}/data/our_input/temp/{domain}.paramfile",
    output:
        "{base_dir}/data/our_input/potts/{domain}_h.pt",
        "{base_dir}/data/our_input/potts/{domain}_J.pt",
    conda:
        "envs/input_generation.yml",
    shell:
        """
        python {wildcards.base_dir}/scripts/pipeline_scripts/read_params.py    \
            {input[0]}                                      \
            {output[0]}                                     \
            {output[1]}
        """

rule plmc:
    input:
        "{base_dir}/data/our_input/msa/{domain}.fasta",
    output:
        temp("{base_dir}/data/our_input/temp/{domain}.couplings"),
        temp("{base_dir}/data/our_input/temp/{domain}.paramfile"),
    threads: 16
    params:
        config["PLMC_MAX_ITER"],
    shell:
        """
        {wildcards.base_dir}/data/pipeline_tools/plmc/bin/plmc     \
             -c {output[0]}                     \
             -o {output[1]}                     \
             -m {params[0]}                     \
             -n {threads} -f {wildcards.domain} \
             {input[0]}
        """

rule reformat:
    input:
        "{base_dir}/data/our_input/temp/{domain}.a3m",
    output:
        "{base_dir}/data/our_input/msa/{domain}.fasta",
    conda:
        "envs/input_generation.yml",
    shell:
        """
        {wildcards.base_dir}/.snakemake/conda/d4f2cb04/scripts/reformat.pl \
            a3m         \
            fas         \
            {input[0]}  \
            {output[0]}
        """

rule hhblits:
    input:
        "{base_dir}/data/our_input/sequences/{domain}.fasta",
    output:
        temp("{base_dir}/data/our_input/temp/{domain}.hh"),
        temp("{base_dir}/data/our_input/temp/{domain}.a3m"),
    params:
        "{base_dir}/data/pipeline_tools/uniclust/UniRef30_2020_01",
        config["HHBLITS_EVALUE"],
        config["HHBLITS_ITER"],
    threads: 16
    conda:
        "envs/input_generation.yml",
    shell:
        """
        hhblits                 \
            -i {input[0]}       \
            -d {params[0]}      \
            -o {output[0]}      \
            -oa3m {output[1]}   \
            -v 2                \
            -cpu {threads}      \
            -e {params[1]}      \
            -n {params[2]}
        """

rule pssm:
    input:       
        "{base_dir}/data/our_input/temp/{domain}.pssm",
    output:
        "{base_dir}/data/our_input/pssm/{domain}_pssm.pt",
    conda:
        "envs/input_generation.yml",
    shell:
        """
        python {wildcards.base_dir}/scripts/pipeline_scripts/pssm.py \
            {input[0]} \
            {output[0]}
        """

rule psiblast:
    input:
        "{base_dir}/data/our_input/sequences/{domain}.fasta",
    output:
        temp("{base_dir}/data/our_input/temp/{domain}.out"),
        temp("{base_dir}/data/our_input/temp/{domain}.pssm"),
    params:
        "{base_dir}/data/pipeline_tools/nrdb/nr",
        config["PSIBLAST_ITER"],
        config["PSIBLAST_EVALUE"],
        config["PSIBLAST_ETHRESH"],
    threads: 16
    conda:
        "envs/input_generation.yml",        
    shell:
        """
        psiblast                                \
            -query {input[0]}                   \
            -db {params[0]}                     \
            -out {output[0]}                    \
            -num_iterations {params[1]}         \
            -num_threads {threads}              \
            -evalue {params[2]}                 \
            -inclusion_ethresh {params[3]}      \
            -save_pssm_after_last_round         \
            -out_ascii_pssm {output[1]} 
        """

