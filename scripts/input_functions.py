from glob import glob

# %%
def get_input(wildcards):
    # global variables defined by snakemake are config, rules, checkpoints...
    templates = [
        "{data_dir}/our_input/tensors/{domain}_X.pt"
    ]
    data_dir = config["data_dir"]
    domains = glob("/faststorage/project/deeply_thinking_potato/data"
                   "/our_input/sequences/*.fasta")
    domains = [x.split('/')[-1].split('.')[0] for x in domains][0:10]
    results = []
    for template in templates:
        for domain in domains:
            results.append(template.format(data_dir=data_dir, domain=domain))

    return results

