import numpy as np
from Bio.PDB import PDBParser, protein_letters_3to1


def align(smaller, larger):
    '''Align smaller Sequence to larger sequence and return the start and end indices'''

    start = -1
    max_score = 0
    for i in range(len(larger) - len(smaller) + 1):
        temp_score = 0
        for j in range(len(smaller)):
            if smaller[j] == larger[i + j]:
                temp_score += 1

        if temp_score > max_score:
            max_score = temp_score
            start = i

    return start


def get_coords(target, casp_sequences, casp_targets, chainid=0):

    L = len(casp_sequences[target])
    protein = casp_targets[target]

    # Get PDB structure
    try:
        structure = PDBParser().get_structure('', f'../../data/casp13_test_data/pdbfiles/{protein}.pdb')
    except (IndexError, ValueError):
        print('IndexError or ValueError')
        return

    # Get the first chain

    chain = structure[0].child_list[chainid]

    coords_list = []

    known_aminoacids = np.array(list(protein_letters_3to1.keys()))

    for i, residue in enumerate(chain.get_residues()):
        residue_name = residue.get_resname()

        if residue_name not in known_aminoacids:
            break

        residue_oneletter = protein_letters_3to1[residue_name]
        residue_number = residue.child_list[0].get_full_id()[3][1]

        if residue_oneletter == 'G':  # if Glycin -> C-alpha. Otherwise C-beta
            try:
                atom = residue['CA']
            except KeyError:
                print('Missing C-alpha atom')
                return
        else:
            try:
                atom = residue['CB']
            except KeyError:
                print('Missing C-beta atom')
                return

        x, y, z = atom.get_coord()
        coords_list.append([residue_number,
                            residue_name,
                            residue_oneletter,
                            x, y, z])

    coords_list = np.array(coords_list, dtype='O')
    coords_start, coords_end = coords_list[0][0], coords_list[-1][0]

    # Fill missing data in the PDB coordinates
    missing = np.setdiff1d(np.arange(coords_list[:, 0][0], 1 + coords_list[:, 0][-1]), coords_list[:, 0]).astype(np.int)
    if len(missing) > 0:
        missingarr = np.zeros((len(missing), 6), dtype='O')
        missingarr[:, 0] = missing
        missingarr[:, 2] = np.repeat('X', len(missing))

        coords_list = np.concatenate((coords_list, missingarr))
        coords_list = coords_list[coords_list[:, 0].argsort()]

    pdbseq = ''.join(coords_list[:, 2])
    caspseq = casp_sequences[target]

    # Align the PDB file with the casp Sequence
    if pdbseq != caspseq:

        fill_before = min(L, coords_start)
        pdbseqX = 'X' * fill_before + pdbseq + 'X' * L

        if fill_before > 0:
            coords_list = np.concatenate((
                np.array([[coords_start - fill_before + i, 0, 'X', 0, 0, 0] for i in range(fill_before)], dtype='O'),
                coords_list,
                np.array([[coords_end + i + 1, 0, 'X', 0, 0, 0] for i in range(L)], dtype='O')
            ))
        else:
            coords_list = np.concatenate((
                coords_list,
                np.array([[coords_end + i + 1, 0, 'X', 0, 0, 0] for i in range(L)], dtype='O')
            ))

        start = align(caspseq, pdbseqX)
        coords_list = coords_list[start:(start + L)]
        for i in range(len(coords_list)):
            if coords_list[i, 2] != 'X':
                if caspseq[i] != coords_list[i, 2]:
                    coords_list[i] = [coords_list[i, 0], 0, 'X', 0, 0, 0]

    return coords_list


def dist_mat(target, casp_sequences, casp_targets):
    """ Generates distance matrix with 64 and 32 bins as well as the sequence"""

    coords = get_coords(target, casp_sequences, casp_targets)

    L = coords.shape[0]

    dist_mat = np.zeros((L, L))

    for i in range(L):
        for j in range(L):
            if coords[i, 2] == 'X' or coords[j, 2] == 'X':
                dist_mat[i, j] = -1
            else:
                dist_mat[i, j] = np.sqrt(np.sum((coords[i, 3:] - coords[j, 3:]) ** 2))

    bins64 = np.concatenate(([0], np.linspace(2, 22, 62), [1000]))
    bins32 = np.concatenate(([0], np.linspace(2, 22, 30), [1000]))

    dist_mat64 = np.digitize(dist_mat, bins64)
    dist_mat32 = np.digitize(dist_mat, bins32)

    return dist_mat64, dist_mat32


def make_fasta(target, casp_sequences):
    sequence = casp_sequences[target]
    with open(f'../../data/casp13_test_data/sequences/{target}.fasta', 'w') as s:
        s.write(f'>{target}\n{sequence}')
