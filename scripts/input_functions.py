from glob import glob
import pandas as pd
import subprocess
import math
import re


# %%
def get_input(wildcards):
    # global variables defined by snakemake are config, rules, checkpoints...
    templates = [
        # "{data_dir}/our_input/tensors/{domain}_X.pt",
        "{data_dir}/our_input/Y_tensors/{domain}_Y.pt",
    ]
    data_dir = config["data_dir"]
    # domains = glob("/faststorage/project/deeply_thinking_potato/data"
    #                "/our_input/sequences/*.fasta")
    domains = glob("/faststorage/project/deeply_thinking_potato/data/our_input/torsion/psi/*_psi.pt")
    domains = [x.split('/')[-1].split('_')[0] for x in domains]
    results = []
    for template in templates:
        for domain in domains:
            results.append(template.format(data_dir=data_dir, domain=domain))

    return results


# %%
def binary_ceil(x):
    return 2**math.ceil(math.log2(x + 1))


def minute_upperbound(time):
    hours, minutes, seconds = [int(x) for x in time.split(':')]
    return binary_ceil(hours * 60 + minutes)


def memMB_upperbound(mem):
    size = float(mem[0:-1])
    multiplier = {'K': 2**-10, 'M': 2**0, 'G': 2**10, 'T': 2**20}[mem[-1]]
    return binary_ceil(size * multiplier)


def write_rule_config():
    output = subprocess.getoutput("""
        find ../logs/ -name "*.err" -print0 \
        | xargs -0 grep -l "Finished"
    """).split('\n')

    df = pd.DataFrame(columns=['jobid', 'rule', 'max_mem', 'max_time'])
    for i, file in enumerate(output[0:1000]):
        parts = re.split('\\/|-|\\.', file)
        df.loc[i, 'jobid'] = parts[-2]
        df.loc[i, 'rule'] = parts[-3]
        output2 = subprocess.getoutput(f"""
            jobinfo {df.loc[i, 'jobid']} \
            | grep -P "Used walltime|Max Mem used"
        """)
        output2 = re.split('\s+', output2)
        df.loc[i, 'max_time'] = output2[3]
        df.loc[i, 'max_mem'] = output2[8]

    df = df.query('max_time != "--" and max_mem != "--"')

    df.loc[:, 'MAX_time'] = df['max_time'].apply(minute_upperbound)
    df.loc[:, 'MAX_mem'] = df['max_mem'].apply(memMB_upperbound)

    result = df.groupby('rule').max().reset_index()

    rule_conf = """
    {}:
        time: {}
        mem: {}
    """

    for i, row in result.iterrows():
        print(rule_conf.format(row['rule'], row['MAX_time'], row['MAX_mem']))
