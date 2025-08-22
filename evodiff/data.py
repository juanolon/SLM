import os
from tqdm import tqdm
from scipy.spatial.distance import hamming, cdist

import numpy as np
from torch.utils.data import Dataset
import pandas as pd

from evodiff.utils import Tokenizer
from sequence_models.utils import parse_fasta
from sequence_models.constants import PROTEIN_ALPHABET, trR_ALPHABET, PAD, GAP
from collections import Counter
import torch
import os
from os import path
import json
import pickle as pkl

from torch.utils.data import Subset
from sequence_models.gnn import bins_to_vals
from sequence_models.pdb_utils import process_coords


DIST_BINS = np.concatenate([np.array([np.nan]), np.linspace(2, 20, 37)])
THETA_BINS = np.concatenate([np.array([np.nan]), np.linspace(0, 2 * np.pi, 25)])
PHI_BINS = np.concatenate([np.array([np.nan]), np.linspace(0, np.pi, 13)])
OMEGA_BINS = np.concatenate([np.array([np.nan]), np.linspace(0, 2 * np.pi, 25)])

def subsample_msa(path_to_msa, n_sequences=64, max_seq_len=512, selection_type='random'):
    alphabet = PROTEIN_ALPHABET
    tokenizer = Tokenizer(alphabet)
    alpha = np.array(list(alphabet))
    gap_idx = tokenizer.alphabet.index(GAP)

    if not os.path.exists(path_to_msa):
        print("PATH TO MSA DOES NOT EXIST")
    path = path_to_msa
    parsed_msa = parse_fasta(path)

    aligned_msa = [[char for char in seq if (char.isupper() or char == '-') and not char == '.'] for seq in
                   parsed_msa]
    aligned_msa = [''.join(seq) for seq in aligned_msa]

    # aligned_msa_length = [len(seq) for seq in aligned_msa]
    # print(f"length of aligned msa = [{aligned_msa_length[0]}] {aligned_msa_length}")
    # aligned_msa = aligned_msa[:-1]

    tokenized_msa = [tokenizer.tokenizeMSA(seq) for seq in aligned_msa]
    tokenized_msa = np.array([l.tolist() for l in tokenized_msa])
    msa_seq_len = len(tokenized_msa[0])

    if msa_seq_len > max_seq_len:
        slice_start = np.random.choice(msa_seq_len - max_seq_len + 1)
        seq_len = max_seq_len
    else:
        slice_start = 0
        seq_len = msa_seq_len

    # Slice to 512
    sliced_msa_seq = tokenized_msa[:, slice_start: slice_start + max_seq_len]
    anchor_seq = sliced_msa_seq[0]  # This is the query sequence in MSA

    # slice out all-gap rows
    sliced_msa = [seq for seq in sliced_msa_seq if (list(set(seq)) != [gap_idx])]
    msa_num_seqs = len(sliced_msa)

    if msa_num_seqs < n_sequences:
        output = np.full(shape=(n_sequences, seq_len), fill_value=tokenizer.pad_id)
        output[:msa_num_seqs] = sliced_msa
        raise Exception("msa num_seqs < self.n_sequences, indicates dataset not filtered properly")
    elif msa_num_seqs > n_sequences:
        if selection_type == 'random':
            random_idx = np.random.choice(msa_num_seqs - 1, size=n_sequences - 1, replace=False) + 1
            anchor_seq = np.expand_dims(anchor_seq, axis=0)
            output = np.concatenate((anchor_seq, np.array(sliced_msa)[random_idx.astype(int)]), axis=0)
        elif selection_type == "MaxHamming":
            output = [list(anchor_seq)]
            msa_subset = sliced_msa[1:]
            msa_ind = np.arange(msa_num_seqs)[1:]
            random_ind = np.random.choice(msa_ind)
            random_seq = sliced_msa[random_ind]
            output.append(list(random_seq))
            random_seq = np.expand_dims(random_seq, axis=0)
            msa_subset = np.delete(msa_subset, (random_ind - 1), axis=0)
            m = len(msa_ind) - 1
            distance_matrix = np.ones((n_sequences - 2, m))

            for i in range(n_sequences - 2):
                curr_dist = cdist(random_seq, msa_subset, metric='hamming')
                curr_dist = np.expand_dims(np.array(curr_dist), axis=0)  # shape is now (1,msa_num_seqs)
                distance_matrix[i] = curr_dist
                col_min = np.min(distance_matrix, axis=0)  # (1,num_choices)
                max_ind = np.argmax(col_min)
                random_ind = max_ind
                random_seq = msa_subset[random_ind]
                output.append(list(random_seq))
                random_seq = np.expand_dims(random_seq, axis=0)
                msa_subset = np.delete(msa_subset, random_ind, axis=0)
                distance_matrix = np.delete(distance_matrix, random_ind, axis=1)
    else:
        output = sliced_msa

    output = [''.join(seq) for seq in alpha[output]]
    return output, output[0]

def read_openfold_files(data_dir, filename):
    """
    Helper function to read the openfold files

    inputs:
        data_dir : path to directory with data
        filename: MSA name

    outputs:
        path: path to .a3m file
    """
    if os.path.exists(data_dir + filename + '/a3m/uniclust30.a3m'):
        path = data_dir + filename + '/a3m/uniclust30.a3m'
    elif os.path.exists(data_dir + filename + '/a3m/bfd_uniclust_hits.a3m'):
        path = data_dir + filename + '/a3m/bfd_uniclust_hits.a3m'
    else:
        raise Exception("Missing filepaths")
    return path

def read_idr_files(data_dir, filename):
    """
    Helper function to read the idr files

    inputs:
        data_dir : path to directory with data
        filename: IDR name

    outputs:
        path: path to IDR file
    """
    if os.path.exists(data_dir + filename):
        path = data_dir + filename
    else:
        raise Exception("Missing filepaths")
    return path

def get_msa_depth_lengths(data_dir, all_files, save_depth_file, save_length_file, idr=False):
    """
    Function to compute openfold and IDR dataset depths

    inputs:
        data_dir : path to directory with data
        all_files: all filenames
        save_depth_file: file to save depth values in
        save_length_file: file to save length values in
    """
    msa_depth = []
    msa_lengths = []
    for filename in tqdm(all_files):
        if idr:
            path = read_idr_files(data_dir, filename)
        else:
            path = read_openfold_files(data_dir, filename)
        parsed_msa = parse_fasta(path)
        msa_depth.append(len(parsed_msa))
        msa_lengths.append(len(parsed_msa[0]))  # all seq in MSA are same length
    np.savez_compressed(data_dir+save_depth_file, np.asarray(msa_depth))
    np.savez_compressed(data_dir + save_length_file, np.asarray(msa_lengths))


def get_valid_msas(data_top_dir, data_dir='openfold/', selection_type='MaxHamming', n_sequences=64, max_seq_len=512,
                   out_path='../DMs/ref/'):
    assert data_dir=='openfold/', "get_valid_msas only works on OPENFOLD"
    _ = torch.manual_seed(1) # same seeds used for training
    np.random.seed(1)

    dataset = A3MMSADataset(selection_type, n_sequences, max_seq_len, data_dir=os.path.join(data_top_dir,data_dir), min_depth=64)

    train_size = len(dataset)
    random_ind = np.random.choice(train_size, size=(train_size - 10000), replace=False)
    val_ind = np.delete(np.arange(train_size), random_ind)
    ds_valid = Subset(dataset, val_ind)

    return ds_valid


def get_idr_query_index(data_dir, all_files, save_file):
    """
    Function to get IDR query index

    inputs:
        data_dir : path to directory with data
        all_files: all filenames
        save_file: file to save query indexes in
    """
    query_idxs = []
    for filename in tqdm(all_files):
        msa_data, msa_names = parse_fasta(data_dir + filename, return_names=True)
        query_idx = [i for i, name in enumerate(msa_names) if name == filename.split('_')[0]][0]  # get query index
        query_idxs.append(query_idx)
    np.savez_compressed(data_dir + save_file, np.asarray(query_idxs))

def get_sliced_gap_depth_openfold(data_dir, all_files, save_file, max_seq_len=512):
    """
    Function to compute make sure every MSA has 64 sequences

    inputs:
        data_dir : path to directory with data
        all_files: all filenames
        save_file: file to save data to
    """
    sliced_depth = []
    for filename in tqdm(all_files):
        path=read_openfold_files(data_dir, filename)
        parsed_msa = parse_fasta(path)
        sliced_msa_depth = [seq for seq in parsed_msa if (Counter(seq)[GAP]) <= max_seq_len] # Only append seqs with gaps<512
        sliced_depth.append(len(sliced_msa_depth))

    np.savez_compressed(data_dir + save_file, np.asarray(sliced_depth))


class TRRMSADataset(Dataset):
    """Build dataset for trRosetta data: MSA Absorbing Diffusion model"""

    def __init__(self, selection_type, n_sequences, max_seq_len, data_dir=None):
        """
        Args:
            selection_type: str,
                MSA selection strategy of random or MaxHamming
            n_sequences: int,
                number of sequences to subsample down to
            max_seq_len: int,
                maximum MSA sequence length
            data_dir: str,
                if you have a specified npz directory
        """

        # Get npz_data dir
        if data_dir is not None:
            self.data_dir = data_dir
        else:
            raise FileNotFoundError(data_dir)

        # MSAs should be in the order of npz_dir
        all_files = os.listdir(self.data_dir)
        if 'trrosetta_lengths.npz' in all_files:
            all_files.remove('trrosetta_lengths.npz')
        all_files = sorted(all_files)
        self.filenames = all_files  # IDs of samples to include

        # Number of sequences to subsample down to
        self.n_sequences = n_sequences
        self.max_seq_len = max_seq_len
        self.selection_type = selection_type

        alphabet = trR_ALPHABET + PAD
        self.tokenizer = Tokenizer(alphabet)
        self.alpha = np.array(list(alphabet))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        data = np.load(self.data_dir + filename)
        # Grab sequence info
        msa = data['msa']

        msa_seq_len = len(msa[0])
        if msa_seq_len > self.max_seq_len:
            slice_start = np.random.choice(msa_seq_len - self.max_seq_len + 1)
            seq_len = self.max_seq_len
        else:
            slice_start = 0
            seq_len = msa_seq_len

        sliced_msa = msa[:, slice_start: slice_start + seq_len]
        anchor_seq = sliced_msa[0]  # This is the query sequence in MSA

        sliced_msa = [list(seq) for seq in sliced_msa if (list(set(seq)) != [self.tokenizer.alphabet.index(GAP)])]
        sliced_msa = np.asarray(sliced_msa)
        msa_num_seqs = len(sliced_msa)

        # If fewer sequences in MSA than self.n_sequences, create sequences padded with PAD token based on 'random' or
        # 'MaxHamming' selection strategy
        if msa_num_seqs < self.n_sequences:
            output = np.full(shape=(self.n_sequences, seq_len), fill_value=self.tokenizer.pad_id)
            output[:msa_num_seqs] = sliced_msa
        elif msa_num_seqs > self.n_sequences:
            if self.selection_type == 'random':
                random_idx = np.random.choice(msa_num_seqs - 1, size=self.n_sequences - 1, replace=False) + 1
                anchor_seq = np.expand_dims(anchor_seq, axis=0)
                output = np.concatenate((anchor_seq, sliced_msa[random_idx]), axis=0)
            elif self.selection_type == 'non-random':
                output = sliced_msa[:self.n_sequences]
            elif self.selection_type == "MaxHamming":
                output = [list(anchor_seq)]
                msa_subset = sliced_msa[1:]
                msa_ind = np.arange(msa_num_seqs)[1:]
                random_ind = np.random.choice(msa_ind)
                random_seq = sliced_msa[random_ind]
                output.append(list(random_seq))
                random_seq = np.expand_dims(random_seq, axis=0)
                msa_subset = np.delete(msa_subset, (random_ind - 1), axis=0)
                m = len(msa_ind) - 1
                distance_matrix = np.ones((self.n_sequences - 2, m))

                for i in range(self.n_sequences - 2):
                    curr_dist = cdist(random_seq, msa_subset, metric='hamming')
                    curr_dist = np.expand_dims(np.array(curr_dist), axis=0)  # shape is now (1,msa_num_seqs)
                    distance_matrix[i] = curr_dist
                    col_min = np.min(distance_matrix, axis=0) # (1,num_choices)
                    max_ind = np.argmax(col_min)
                    random_ind = max_ind
                    random_seq = msa_subset[random_ind]
                    output.append(list(random_seq))
                    random_seq = np.expand_dims(random_seq, axis=0)
                    msa_subset = np.delete(msa_subset, random_ind, axis=0)
                    distance_matrix = np.delete(distance_matrix, random_ind, axis=1)
        else:
            output = sliced_msa
        output = [''.join(seq) for seq in self.alpha[output]]
        print("shape of msa", len(output), len(output[0]))
        print(output) # check that there are no all-msa rows
        #import pdb; pdb.set_trace()
        return output


class A3MMSADataset(Dataset):
    """Build dataset for A3M data: MSA Absorbing Diffusion model"""

    def __init__(self, selection_type, n_sequences, max_seq_len, data_dir=None, min_depth=None):
        """
        Args:
            selection_type: str,
                MSA selection strategy of random or MaxHamming
            n_sequences: int,
                number of sequences to subsample down to
            max_seq_len: int,
                maximum MSA sequence length
            data_dir: str,
                if you have a specified data directory
        """
        alphabet = PROTEIN_ALPHABET
        self.tokenizer = Tokenizer(alphabet)
        self.alpha = np.array(list(alphabet))
        self.gap_idx = self.tokenizer.alphabet.index(GAP)

        # Get npz_data dir
        if data_dir is not None:
            self.data_dir = data_dir
        else:
            raise FileNotFoundError(data_dir)

        [print("Excluding", x) for x in os.listdir(self.data_dir) if x.endswith('.npz')]
        all_files = [x for x in os.listdir(self.data_dir) if not x.endswith('.npz')]
        all_files = sorted(all_files)
        print("unfiltered length", len(all_files))

        ## Filter based on depth (keep > 64 seqs/MSA)
        if not os.path.exists(data_dir + 'openfold_lengths.npz'):
            raise Exception("Missing openfold_lengths.npz in openfold/")
        if not os.path.exists(data_dir + 'openfold_depths.npz'):
            #get_msa_depth_openfold(data_dir, sorted(all_files), 'openfold_depths.npz')
            raise Exception("Missing openfold_depths.npz in openfold/")
        if min_depth is not None: # reindex, filtering out MSAs < min_depth
            _depths = np.load(data_dir+'openfold_depths.npz')['arr_0']
            depths = pd.DataFrame(_depths, columns=['depth'])
            depths = depths[depths['depth'] >= min_depth]
            keep_idx = depths.index

            _lengths = np.load(data_dir+'openfold_lengths.npz')['ells']
            lengths = np.array(_lengths)[keep_idx]
            all_files = np.array(all_files)[keep_idx]
            print("filter MSA depth > 64", len(all_files))

        # Re-filter based on high gap-contining rows
        if not os.path.exists(data_dir + 'openfold_gap_depths.npz'):
            #get_sliced_gap_depth_openfold(data_dir, all_files, 'openfold_gap_depths.npz', max_seq_len=max_seq_len)
            raise Exception("Missing openfold_gap_depths.npz in openfold/")
        _gap_depths = np.load(data_dir + 'openfold_gap_depths.npz')['arr_0']
        gap_depths = pd.DataFrame(_gap_depths, columns=['gapdepth'])
        gap_depths = gap_depths[gap_depths['gapdepth'] >= min_depth]
        filter_gaps_idx = gap_depths.index
        lengths = np.array(lengths)[filter_gaps_idx]
        all_files = np.array(all_files)[filter_gaps_idx]
        print("filter rows with GAPs > 512", len(all_files))

        self.filenames = all_files  # IDs of samples to include
        self.lengths = lengths # pass to batch sampler
        self.n_sequences = n_sequences
        self.max_seq_len = max_seq_len
        self.selection_type = selection_type

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        path = read_openfold_files(self.data_dir, filename)
        parsed_msa = parse_fasta(path)

        aligned_msa = [[char for char in seq if (char.isupper() or char == '-') and not char == '.'] for seq in parsed_msa]
        aligned_msa = [''.join(seq) for seq in aligned_msa]

        tokenized_msa = [self.tokenizer.tokenizeMSA(seq) for seq in aligned_msa]
        tokenized_msa = np.array([l.tolist() for l in tokenized_msa])
        msa_seq_len = len(tokenized_msa[0])

        if msa_seq_len > self.max_seq_len:
            slice_start = np.random.choice(msa_seq_len - self.max_seq_len + 1)
            seq_len = self.max_seq_len
        else:
            slice_start = 0
            seq_len = msa_seq_len

        # Slice to 512
        sliced_msa_seq = tokenized_msa[:, slice_start: slice_start + self.max_seq_len]
        anchor_seq = sliced_msa_seq[0]  # This is the query sequence in MSA

        # slice out all-gap rows
        sliced_msa = [seq for seq in sliced_msa_seq if (list(set(seq)) != [self.gap_idx])]
        msa_num_seqs = len(sliced_msa)

        if msa_num_seqs < self.n_sequences:
            print("before for len", len(sliced_msa_seq))
            print("msa_num_seqs < self.n_sequences should not be called")
            print("tokenized msa shape", tokenized_msa.shape)
            print("tokenized msa depth", len(tokenized_msa))
            print("sliced msa depth", msa_num_seqs)
            print("used to set slice")
            print("msa_seq_len", msa_seq_len)
            print("self max seq len", self.max_seq_len)
            print(slice_start)
            import pdb; pdb.set_trace()
            output = np.full(shape=(self.n_sequences, seq_len), fill_value=self.tokenizer.pad_id)
            output[:msa_num_seqs] = sliced_msa
            raise Exception("msa num_seqs < self.n_sequences, indicates dataset not filtered properly")
        elif msa_num_seqs > self.n_sequences:
            if self.selection_type == 'random':
                random_idx = np.random.choice(msa_num_seqs - 1, size=self.n_sequences - 1, replace=False) + 1
                anchor_seq = np.expand_dims(anchor_seq, axis=0)
                output = np.concatenate((anchor_seq, np.array(sliced_msa)[random_idx.astype(int)]), axis=0)
            elif self.selection_type == "MaxHamming":
                output = [list(anchor_seq)]
                msa_subset = sliced_msa[1:]
                msa_ind = np.arange(msa_num_seqs)[1:]
                random_ind = np.random.choice(msa_ind)
                random_seq = sliced_msa[random_ind]
                output.append(list(random_seq))
                random_seq = np.expand_dims(random_seq, axis=0)
                msa_subset = np.delete(msa_subset, (random_ind - 1), axis=0)
                m = len(msa_ind) - 1
                distance_matrix = np.ones((self.n_sequences - 2, m))

                for i in range(self.n_sequences - 2):
                    curr_dist = cdist(random_seq, msa_subset, metric='hamming')
                    curr_dist = np.expand_dims(np.array(curr_dist), axis=0)  # shape is now (1,msa_num_seqs)
                    distance_matrix[i] = curr_dist
                    col_min = np.min(distance_matrix, axis=0)  # (1,num_choices)
                    max_ind = np.argmax(col_min)
                    random_ind = max_ind
                    random_seq = msa_subset[random_ind]
                    output.append(list(random_seq))
                    random_seq = np.expand_dims(random_seq, axis=0)
                    msa_subset = np.delete(msa_subset, random_ind, axis=0)
                    distance_matrix = np.delete(distance_matrix, random_ind, axis=1)
        else:
            output = sliced_msa

        output = [''.join(seq) for seq in self.alpha[output]]
        return output


class IDRDataset(Dataset):
    """Build dataset for IDRs"""

    def __init__(self, selection_type, n_sequences, max_seq_len, data_dir=None, min_depth=None):
        """
        Args:
            selection_type: str,
                MSA selection strategy of random or MaxHamming
            n_sequences: int,
                number of sequences to subsample down to
            max_seq_len: int,
                maximum MSA sequence length
            data_dir: str,
                if you have a specified data directory
        """
        alphabet = PROTEIN_ALPHABET
        self.tokenizer = Tokenizer(alphabet)
        self.alpha = np.array(list(alphabet))
        self.gap_idx = self.tokenizer.alphabet.index(GAP)

        # Get npz_data dir
        if data_dir is not None:
            self.data_dir = data_dir
        else:
            raise FileNotFoundError(data_dir)

        [print("Excluding", x) for x in os.listdir(self.data_dir) if x.endswith('.npz')]
        all_files = [x for x in os.listdir(self.data_dir) if not x.endswith('.npz')]
        all_files = sorted(all_files)
        print("unfiltered length", len(all_files))

        ## Filter based on depth (keep > 64 seqs/MSA)
        if not os.path.exists(data_dir + 'idr_lengths.npz'):
            raise Exception("Missing idr_lengths.npz in human_idr_alignments/human_protein_alignments/")
        if not os.path.exists(data_dir + 'idr_depths.npz'):
            #get_msa_depth_openfold(data_dir, sorted(all_files), 'openfold_depths.npz')
            raise Exception("Missing idr_depths.npz in human_idr_alignments/human_protein_alignments/")
        _depths = np.load(data_dir + 'idr_depths.npz')['arr_0']
        depths = pd.DataFrame(_depths, columns=['depth'])

        if min_depth is not None: # reindex, filtering out MSAs < min_depth
            raise Exception("MIN DEPTH CONSTRAINT NOT CURRENTLY WORKING ON IDRS")
        #    depths = depths[depths['depth'] >= min_depth]
        #keep_idx = depths.index

        _lengths = np.load(data_dir + 'idr_lengths.npz')['arr_0']
        lengths = pd.DataFrame(_lengths, columns=['length'])
        if max_seq_len is not None:
            lengths = lengths[lengths['length'] <= max_seq_len]
        keep_idx = lengths.index

        lengths = np.array(_lengths)[keep_idx]
        all_files = np.array(all_files)[keep_idx]
        print("filter MSA length >", max_seq_len, len(all_files))

        _query_idxs = np.load(data_dir+'idr_query_idxs.npz')['arr_0']
        query_idxs = np.array(_query_idxs)[keep_idx]

        self.filenames = all_files  # IDs of samples to include
        self.lengths = lengths # pass to batch sampler
        self.n_sequences = n_sequences
        self.max_seq_len = max_seq_len
        self.selection_type = selection_type
        self.query_idxs = query_idxs


    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        print(filename)
        path = read_idr_files(self.data_dir, filename)
        parsed_msa = parse_fasta(path)
        aligned_msa = [[char for char in seq if (char.isupper() or char == '-') and not char == '.'] for seq in parsed_msa]
        aligned_msa = [''.join(seq) for seq in aligned_msa]

        tokenized_msa = [self.tokenizer.tokenizeMSA(seq) for seq in aligned_msa]

        tokenized_msa = np.array([l.tolist() for l in tokenized_msa])
        msa_seq_len = len(tokenized_msa[0])
        print("msa_seq_len", msa_seq_len, "max seq len", self.max_seq_len)

        if msa_seq_len > self.max_seq_len:
            slice_start = np.random.choice(msa_seq_len - self.max_seq_len + 1)
            seq_len = self.max_seq_len
        else:
            slice_start = 0
            seq_len = msa_seq_len

        # Slice to 512
        sliced_msa_seq = tokenized_msa[:, slice_start: slice_start + self.max_seq_len]
        #print(sliced_msa_seq.shape)
        query_idx = self.query_idxs[idx]
        anchor_seq = tokenized_msa[query_idx]  # This is the query sequence
        print("anchor seq", len(anchor_seq))
        # Remove query from MSA?
        #del tokenized_msa[query_idx]

        # slice out all-gap rows
        sliced_msa = [seq for seq in sliced_msa_seq if (list(set(seq)) != [self.gap_idx])]
        msa_num_seqs = len(sliced_msa)

        # if msa_num_seqs < self.n_sequences:
        #     raise Exception("msa num_seqs < self.n_sequences, indicates dataset not filtered properly")
        if msa_num_seqs > self.n_sequences:
            if self.selection_type == 'random':
                random_idx = np.random.choice(msa_num_seqs - 1, size=self.n_sequences - 1, replace=False) + 1
                anchor_seq = np.expand_dims(anchor_seq, axis=0)
                output = np.concatenate((anchor_seq, np.array(sliced_msa)[random_idx.astype(int)]), axis=0)
            elif self.selection_type == "MaxHamming":
                output = [list(anchor_seq)]
                msa_subset = sliced_msa[1:]
                msa_ind = np.arange(msa_num_seqs)[1:]
                random_ind = np.random.choice(msa_ind)
                random_seq = sliced_msa[random_ind]
                output.append(list(random_seq))
                random_seq = np.expand_dims(random_seq, axis=0)
                msa_subset = np.delete(msa_subset, (random_ind - 1), axis=0)
                m = len(msa_ind) - 1
                distance_matrix = np.ones((self.n_sequences - 2, m))

                for i in range(self.n_sequences - 2):
                    curr_dist = cdist(random_seq, msa_subset, metric='hamming')
                    curr_dist = np.expand_dims(np.array(curr_dist), axis=0)  # shape is now (1,msa_num_seqs)
                    distance_matrix[i] = curr_dist
                    col_min = np.min(distance_matrix, axis=0)  # (1,num_choices)
                    max_ind = np.argmax(col_min)
                    random_ind = max_ind
                    random_seq = msa_subset[random_ind]
                    output.append(list(random_seq))
                    random_seq = np.expand_dims(random_seq, axis=0)
                    msa_subset = np.delete(msa_subset, random_ind, axis=0)
                    distance_matrix = np.delete(distance_matrix, random_ind, axis=1)
        else:
            output = sliced_msa

        output = [''.join(seq) for seq in self.alpha[output]]
        return output
    
def trr_bin(dist, omega, theta, phi):
    dist = torch.tensor(np.digitize(dist, DIST_BINS[1:]) % (len(DIST_BINS) - 1))
    idx = np.where(omega == omega)
    jdx = np.where(omega[idx] < 0)[0]
    idx = tuple(i[jdx] for i in idx)
    omega[idx] = 2 * np.pi + omega[idx]
    omega = torch.tensor(np.digitize(omega, OMEGA_BINS[1:]) % (len(OMEGA_BINS) - 1))
    idx = np.where(theta == theta)
    jdx = np.where(theta[idx] < 0)[0]
    idx = tuple(i[jdx] for i in idx)
    theta[idx] = 2 * np.pi + theta[idx]
    theta = torch.tensor(np.digitize(theta, THETA_BINS[1:]) % (len(THETA_BINS) - 1))
    phi = torch.tensor(np.digitize(phi, PHI_BINS[1:]) % (len(PHI_BINS) - 1))
    idx = torch.where(dist == 0)
    omega[idx] = 0
    theta[idx] = 0
    phi[idx] = 0
    return dist, omega, theta, phi


class UniRefDataset(Dataset):
    """
    Dataset that pulls from UniRef/Uniclust downloads.

    The data folder should contain the following:
    - 'consensus.fasta': consensus sequences, no line breaks in sequences
    - 'splits.json': a dict with keys 'train', 'valid', and 'test' mapping to lists of indices
    - 'lengths_and_offsets.npz': byte offsets for the 'consensus.fasta' and sequence lengths
    """

    def __init__(self, data_dir: str, split: str, structure=False, pdb=False, coords=False, bins=False,
                 p_drop=0.0, max_len=2048):
        self.data_dir = data_dir
        self.split = split
        self.structure = structure
        self.coords = coords
        with open(data_dir + 'splits.json', 'r') as f:
            self.indices = json.load(f)[self.split]
        metadata = np.load(self.data_dir + 'lengths_and_offsets.npz')
        self.offsets = metadata['seq_offsets']
        self.pdb = pdb
        self.bins = bins
        if self.pdb or self.bins:
            self.n_digits = 6
        else:
            self.n_digits = 8
        if self.coords:
            with open(data_dir + 'coords.pkl', 'rb') as f:
                self.structures = pkl.load(f)
        self.p_drop = p_drop
        self.max_len = max_len

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idx = self.indices[idx]
        offset = self.offsets[idx]
        with open(self.data_dir + 'consensus.fasta') as f:
            f.seek(offset)
            consensus = f.readline()[:-1]
        if len(consensus) - self.max_len > 0:
            start = np.random.choice(len(consensus) - self.max_len)
            stop = start + self.max_len
        else:
            start = 0
            stop = len(consensus)
        if self.coords:
            coords = self.structures[str(idx)]
            dist, omega, theta, phi = process_coords(coords)
            dist = torch.tensor(dist).float()
            omega = torch.tensor(omega).float()
            theta = torch.tensor(theta).float()
            phi = torch.tensor(phi).float()
        elif self.structure:
            sname = 'structures/{num:{fill}{width}}.npz'.format(num=idx, fill='0', width=self.n_digits)
            fname = self.data_dir + sname
            if path.isfile(fname):
                structure = np.load(fname)
            else:
                structure = None
            if structure is not None:
                if np.random.random() < self.p_drop:
                    structure = None
                elif self.pdb:
                    dist = torch.tensor(structure['dist']).float()
                    omega = torch.tensor(structure['omega']).float()
                    theta = torch.tensor(structure['theta']).float()
                    phi = torch.tensor(structure['phi']).float()
                    if self.bins:
                        dist, omega, theta, phi = trr_bin(dist, omega, theta, phi)
                else:
                    dist, omega, theta, phi = bins_to_vals(data=structure)
            if structure is None:
                dist, omega, theta, phi = bins_to_vals(L=len(consensus))
        if self.structure or self.coords:
            consensus = consensus[start:stop]
            dist = dist[start:stop, start:stop]
            omega = omega[start:stop, start:stop]
            theta = theta[start:stop, start:stop]
            phi = phi[start:stop, start:stop]
            return consensus, dist, omega, theta, phi
        consensus = consensus[start:stop]

        return (consensus,)
    
def _pad(tokenized, max_len, value, dim=2):

    tokenized = torch.tensor(tokenized)


    seq_len = tokenized.shape[0]
    bos_id, eos_id = tokenized[0], tokenized[-1]

    # cut off sequences that are too long
    if(seq_len > max_len):
        # randomly cut off the sequence
        start = np.random.randint(0, seq_len - max_len)
        end = start + max_len
        tokenized = tokenized[start:end]
        tokenized[0] = bos_id
        tokenized[-1] = eos_id

    output = torch.zeros((max_len,)) + value
    output[ :min(seq_len, max_len)] = tokenized

    return output

class WrappedUniRefDataset(Dataset):
    """
    Dataset that wraps a UniRefDataset and returns the consensus sequence and the sequence at a given index.
    """

    def __init__(self, dataset: UniRefDataset, tokenizer, max_len=1024, restrit=False):
        self.dataset = dataset
        # restrit dataset to 24000 samples
        self.new_dataset = []
        if restrit:
            for i in tqdm(range(24000) ):
                r_idx = np.random.choice(len(self.dataset))
                item = self.dataset[r_idx]
                self.new_dataset.append(item)
            self.dataset = self.new_dataset


        self.tokenizer = tokenizer
        self.max_length = max_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        consensus, *structure = self.dataset[idx]
        # print(f"consensus = {consensus}")
        tokenized = self.tokenizer.encode(consensus)
        tokenized = np.insert(tokenized, 0, self.tokenizer.bos_token_id)
        tokenized = np.append(tokenized, self.tokenizer.eos_token_id)

        tokenized = _pad(tokenized, self.max_length, self.tokenizer.pad_token_id)
        # print(f"tokenized item = {tokenized}")
        padded = (tokenized != self.tokenizer.pad_token_id)
        # padded = padded.to(torch.long)
        # print(f"attention mask = {padded}")

        return {
            'input_ids': tokenized.to(torch.long),
            'attention_mask': padded.to(torch.float32),
        }

    
    # # apply func to each item in the dataset
    # def map(self, function, batched=False, num_proc=1, load_from_cache_file=True, desc=None):

    #     # Apply the function to each element in the dataset
    #     new_dataset = []
    #     for item in tqdm(self.dataset, desc=desc):
    #         new_dataset.append( function(item) )

    #     self.dataset = new_dataset
    #     return self
        

