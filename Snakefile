configfile: "./configs/config.yaml"

include: "./scripts/input_functions.py"

# run with:
# snakemake -s tmp.smk --use-conda --cluster 'sbatch --mem=64g -c 16 -t 720' --jobs 10000 --batch main=1/1000 
# I recommend running in background using GNU screen application

rule main:
    input:
        get_input,
        # generates all files with the path:
         #"{data_dir}/our_input/potts/{domain}_FN.pt",
         #"{data_dir}/our_input/potts/{domain}_h.pt",
         #"{data_dir}/our_input/potts/{domain}_J.pt",
         #"{data_dir}/our_input/pssm/{domain}_pssm.pt",
        # where data_dir is from config and domain is from {data_dir}/sequences/*.fasta

rule frobenius:
    input:
        "{data_dir}/our_input/temp/{domain}.couplings",
    output:
        "{data_dir}/our_input/potts/{domain}_FN.pt",
    conda:
        "envs/input_generation.yml",
    shell:
        """
        python ./scripts/pipeline_scripts/read_couplings.py \
            {input[0]}\
            {output[0]}
        """

rule hJ:
    input:
        "{data_dir}/our_input/temp/{domain}.paramfile",
    output:
        "{data_dir}/our_input/potts/{domain}_h.pt",
        "{data_dir}/our_input/potts/{domain}_J.pt",
    conda:
        "envs/input_generation.yml",
    shell:
        """
        python ./scripts/pipeline_scripts/read_params.py    \
            {input[0]}                                      \
            {output[0]}                                     \
            {output[1]}
        """

rule plmc:
    input:
        "{data_dir}/our_input/msa/{domain}.fasta",
    output:
        temp("{data_dir}/our_input/temp/{domain}.couplings"),
        temp("{data_dir}/our_input/temp/{domain}.paramfile"),
    threads: 16
    params:
        config["PLMC_MAX_ITER"],
    shell:
        """
        ./data/pipeline_tools/plmc/bin/plmc     \
             -c {output[0]}                     \
             -o {output[1]}                     \
             -m {params[0]}                     \
             -n {threads} -f {wildcards.domain} \
             {input[0]}
        """

rule reformat:
    input:
        "{data_dir}/our_input/temp/{domain}.a3m",
    output:
        "{data_dir}/our_input/msa/{domain}.fasta",
    conda:
        "envs/input_generation.yml",
    shell:
        """
        /faststorage/project/deeply_thinking_potato/.snakemake/conda/d4f2cb04/scripts/reformat.pl \
            a3m         \
            fas         \
            {input[0]}  \
            {output[0]}
        """

rule hhblits:
    input:
        "{data_dir}/our_input/sequences/{domain}.fasta",
        # "{data_dir}/pipeline_tools/uniclust/UniRef30_2020_01_a3m.ffdata",   # maybe add all files in the future
    output:
        temp("{data_dir}/our_input/temp/{domain}.hh"),
        temp("{data_dir}/our_input/temp/{domain}.a3m"),
    params:
        "{data_dir}/pipeline_tools/uniclust/UniRef30_2020_01",
        config["HHBLITS_EVALUE"],
        config["HHBLITS_ITER"],
    threads: 16
    conda:
        "envs/input_generation.yml",
    shell:
        """
        # LD_LIBRARY_PATH=/faststorage/project/deeply_thinking_potato/.snakemake/conda/36bdbca7/lib:$LD_LIBRARY_PATH

        # {wildcards.data_dir}/pipeline_tools/hh-suite/build/bin/hhblits_mpi

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
        "{data_dir}/our_input/temp/{domain}.pssm",
    output:
        "{data_dir}/our_input/pssm/{domain}_pssm.pt",
    conda:
        "envs/input_generation.yml",
    shell:
        """
        python ./scripts/pipeline_scripts/pssm.py \
            {input[0]} \
            {output[0]}
        """

rule psiblast:
    input:
        "{data_dir}/our_input/sequences/{domain}.fasta",
    output:
        temp("{data_dir}/our_input/temp/{domain}.out"),
        temp("{data_dir}/our_input/temp/{domain}.pssm"),
    params:
        "{data_dir}/pipeline_tools/nrdb/nr",
        config["PSIBLAST_ITER"],
        config["PSIBLAST_EVALUE"],
        config["PSIBLAST_ETHRESH"],
    threads: 16
    conda:
        "envs/input_generation.yml",
        
    shell:
        """
        psiblast                              \
            -query {input[0]}                 \
            -db {params[0]}                   \
            -out {output[0]}                 \
            -num_iterations {params[1]}       \
            -num_threads {threads}            \
            -evalue {params[2]}               \
            -inclusion_ethresh {params[3]}   \
            -save_pssm_after_last_round       \
            -out_ascii_pssm {output[1]} 
        """

