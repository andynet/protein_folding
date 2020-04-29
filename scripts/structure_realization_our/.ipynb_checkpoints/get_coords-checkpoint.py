from Bio.PDB import PDBParser, protein_letters_3to1
import numpy as np

def str_to_int(s):
    nums = '0123456789'
    for i in s:
        if i in nums:
            pass
        else:
            return None
    return int(s)

domains = {}
with open('../../data/our_input/cath-domain-seqs-S35.fa') as f:
    while True:
        info = f.readline()
        if info == '':
            break
            
            # extract domain id and positions
        domain, pos = info.strip().split('|')[2].split('/')
        pos = pos.split('_') # in case the domain is not continuous
        
        if len(pos) == 1:  # we only allow continous domains 
            start, end = pos[0].strip('-').split('-')
            start, end = str_to_int(start), str_to_int(end) # check if there is a character in position
            if start is None or end is None:
                f.readline()
            else:
                domains[domain] = [
                    start, end,
                    f.readline().strip()]
        else:
            f.readline()


def get_coords(domain):
    
    domain_start, domain_end = domains[domain][0], domains[domain][1]
    domain_id = domain[:4]
    chain_id = domain[4]
    
    if chain_id == '0': # BioPython does not know 0, it has to translated to the space character
        chain_id = ' '
    
    # Extract Chain from the Domain
    structure = PDBParser().get_structure('', f'../../data/pdbfiles/{domain_id}.pdb')
    chain = structure[0][chain_id]
    
    coords_list = []
    domain_size = domain_end - domain_start + 1
    
    known_aminoacids = np.array(list(protein_letters_3to1.keys()))
    
    for i, residue in enumerate(chain.get_residues()):
        
        residue_name = residue.get_resname()
        
        if residue_name not in known_aminoacids: 
            break

        residue_oneletter = protein_letters_3to1[residue_name]
        residue_number = residue.child_list[0].get_full_id()[3][1]
        
        if residue_oneletter == 'G': # if Glycin -> C-alpha. Otherwise C-beta
            try:
                C = residue['C']
                N = residue['N']
                CA = residue['CA']
                
                for atom in [N, CA, C]:
                    x, y, z = atom.get_coord()
                    coords_list.append([residue_number,  
                                        residue_name,
                                        residue_oneletter,
                                        atom.fullname.strip(),
                                        x, y, z])
            except:
                print('Atom not found')
                return
                #if residue_number < domain_start:
                #    atom = residue.child_list[0] # just append any atom, it doesnt matter
                #else:
                #    print('Missing C-alpha atom')
                #    return
        else:
            try:
                C = residue['C']
                N = residue['N']
                CA = residue['CA']
                CB = residue['CB']
                
                for atom in [N, CA, CB, C]:
                    x, y, z = atom.get_coord()
                    coords_list.append([residue_number,  
                                        residue_name,
                                        residue_oneletter,
                                        atom.fullname.strip(),
                                        x, y, z])                  
            except:
                print('Atom not found')
                return
                #if residue_number < domain_start:
                #    atom = residue.child_list[0] # just append any atom, it doesnt matter
                #else:
                #    print('Missing C-beta atom')
                #    return
        
        if residue_number == domain_end: # because we need to include also that residue
            break
            
    coords_list = np.array(coords_list, dtype='O')
    return coords_list
    # in case the domain_start is not included in the coords indices
    #try:
    #    start = np.where(coords_list[:, 0] == domain_start)[0][0]
    #    end = np.where(coords_list[:, 0] == domain_end)[0][0]
    #except:
    #    print('domain_start or domain_end index not found in pdb file')
    #    return None
    
    #if (end - start) == (domain_end - domain_start):
    #    return coords_list[start:(end+1)]
    #else:
        #print(f'Domain {domain} has missing data. PDB indices:{start, end}, CATH indices: {domain_start, domain_end}')
        #return None
        #return coords_list[start:(end+1)]