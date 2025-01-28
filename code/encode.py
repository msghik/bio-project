""" Encodes data in different formats """

from os.path import join, isfile
import argparse

import numpy as np
from sklearn.preprocessing import OneHotEncoder

import constants
import utils
##########################################
# Additional imports at the top of encode.py
##########################################
import torch
from transformers import AutoTokenizer, AutoModel

##########################################
# New function for PLM-based encoding
##########################################
def enc_plm_embeddings(char_seqs, model_name="Rostlab/prot_bert_bfd", device="cpu"):
    """
    Encodes protein sequences into embeddings using a pretrained protein language model.

    Parameters
    ----------
    char_seqs : list of str
        List of protein sequences (single-letter codes), e.g. ["MKT...", "ACDE..."].
    model_name : str
        Name of the pretrained model to load from Hugging Face.
    device : str
        Device to run the model on, e.g. "cpu" or "cuda".

    Returns
    -------
    embeddings : np.ndarray
        A 3D NumPy array of shape (num_sequences, max_seq_len, hidden_dim),
        containing the token-wise embeddings for each input sequence.
    """

    # 1) Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    # We’ll collect embeddings in a list, then stack them
    all_embeddings = []

    with torch.no_grad():
        for seq in char_seqs:
            # 2) Prepend special tokens as required by the model
            # ProtBert typically expects a sequence like: [CLS] A C D ... [SEP]
            # but the tokenizer handles that automatically if you pass just seq.

            # The model might expect spaces between each amino acid (depending on model).
            # For ProtBert, you often do: " ".join(list(seq)) if your seq is "ACDE..."
            spaced_seq = " ".join(list(seq))

            encoded_input = tokenizer(
                spaced_seq,
                return_tensors='pt',
                add_special_tokens=True
            ).to(device)

            # 3) Forward pass to get hidden states
            outputs = model(**encoded_input)
            # Typically, outputs.last_hidden_state has shape [batch_size, seq_len, hidden_dim]
            hidden_states = outputs.last_hidden_state

            # Remove batch dimension (batch_size=1 here)
            hidden_states = hidden_states.squeeze(0)  # shape: (seq_len, hidden_dim)

            # 4) Convert to NumPy
            emb_np = hidden_states.cpu().numpy()  # shape (seq_len, hidden_dim)
            all_embeddings.append(emb_np)

    # Now, we have a list of arrays with potentially different lengths (since sequences may differ).
    # If your sequences are the same length, you can stack them directly.
    # Otherwise, you might want to pad them to a uniform length or store them in a list.

    # For demonstration, let’s assume all sequences are the same length:
    embeddings = np.stack(all_embeddings, axis=0)  # shape (num_seqs, seq_len, hidden_dim)
    return embeddings


def enc_aa_index(int_seqs):
    """ encodes data in aa index properties format """
    aa_features = np.load("data/aaindex/pca-19.npy")
    # add all zero features for stop codon
    aa_features = np.insert(aa_features, 0, np.zeros(aa_features.shape[1]), axis=0)
    aa_features_enc = aa_features[int_seqs]
    return aa_features_enc


def enc_one_hot(int_seqs):
    enc = OneHotEncoder(categories=[range(constants.NUM_CHARS)] * int_seqs.shape[1], dtype=np.bool, sparse=False)
    one_hot = enc.fit_transform(int_seqs).reshape((int_seqs.shape[0], int_seqs.shape[1], constants.NUM_CHARS))
    return one_hot


def enc_int_seqs_from_char_seqs(char_seqs):
    seq_ints = []
    for char_seq in char_seqs:
        int_seq = [constants.C2I_MAPPING[c] for c in char_seq]
        seq_ints.append(int_seq)
    seq_ints = np.array(seq_ints)
    return seq_ints


def enc_int_seqs_from_variants(variants, wild_type_seq, wt_offset=0):
    # convert wild type seq to integer encoding
    wild_type_int = np.zeros(len(wild_type_seq), dtype=np.uint8)
    for i, c in enumerate(wild_type_seq):
        wild_type_int[i] = constants.C2I_MAPPING[c]

    seq_ints = np.tile(wild_type_int, (len(variants), 1))

    for i, variant in enumerate(variants):
        # special handling if we want to encode the wild-type seq
        # the seq_ints array is already filled with WT, so all we have to do is just ignore it
        # and it will be properly encoded
        if variant == "_wt":
            continue

        # variants are a list of mutations [mutation1, mutation2, ....]
        variant = variant.split(",")
        for mutation in variant:
            # mutations are in the form <original char><position><replacement char>
            position = int(mutation[1:-1])
            replacement = constants.C2I_MAPPING[mutation[-1]]
            seq_ints[i, position-wt_offset] = replacement

    return seq_ints


def encode_int_seqs(char_seqs=None, variants=None, wild_type_aa=None, wild_type_offset=None):
    single = False
    if variants is not None:
        if not isinstance(variants, list):
            single = True
            variants = [variants]

        int_seqs = enc_int_seqs_from_variants(variants, wild_type_aa, wild_type_offset)

    elif char_seqs is not None:
        if not isinstance(char_seqs, list):
            single = True
            char_seqs = [char_seqs]

        int_seqs = enc_int_seqs_from_char_seqs(char_seqs)

    return int_seqs, single


def encode(encoding, char_seqs=None, variants=None, ds_name=None, wt_aa=None, wt_offset=None):
    """
    the main encoding function that will encode the given sequences or variants and return the encoded data
    Now extended to include 'plm' embeddings using pretrained language models.
    """

    if variants is None and char_seqs is None:
        raise ValueError("must provide either variants or full sequences to encode")

    if variants is not None and ((ds_name is None) and ((wt_aa is None) or (wt_offset is None))):
        raise ValueError("if providing variants, must also provide (wt_aa and wt_offset) or ds_name so I can look up the WT sequence myself")

    # If the user wants plm encoding, we want raw character sequences (like 'ACDE...'),
    # so let's create them from variants or use char_seqs directly.
    if "plm" in encoding:
        # We'll gather the actual char sequences:
        if char_seqs is None:
            # Convert from variants to full char seq
            # e.g. build them from the wild_type_aa with the specified variant changes
            # Because enc_int_seqs_from_variants already returns an integer matrix,
            # we can map that back to characters, or we can replicate the logic to directly get char_seqs.
            # For simplicity, let's just replicate the logic to produce char strings.
            char_seqs_list = []
            int_seqs, _ = encode_int_seqs(variants=variants, wild_type_aa=wt_aa, wt_offset=wt_offset)
            
            # Build reverse mapping from integer -> char
            rev_map = {v: k for k, v in constants.C2I_MAPPING.items()}

            for row in int_seqs:
                seq_str = "".join(rev_map[int_val] for int_val in row)
                char_seqs_list.append(seq_str)

        else:
            if isinstance(char_seqs, str):
                char_seqs_list = [char_seqs]
            else:
                char_seqs_list = char_seqs

        # Now call the PLM embedding function
        plm_embs = enc_plm_embeddings(char_seqs_list, model_name="Rostlab/prot_bert_bfd", device="cpu")
        return plm_embs  # shape: (num_seqs, seq_len, hidden_dim)

    # Otherwise, proceed with the existing logic
    int_seqs, single = encode_int_seqs(
        char_seqs=char_seqs,
        variants=variants,
        wild_type_aa=wt_aa,
        wild_type_offset=wt_offset
    )

    encodings_list = encoding.split(",")
    encoded_data = []

    for enc in encodings_list:
        if enc == "one_hot":
            encoded_data.append(enc_one_hot(int_seqs))
        elif enc == "aa_index":
            encoded_data.append(enc_aa_index(int_seqs))
        elif enc == "plm":
            # If we get here, it means the user had "plm" among multiple encodings, e.g. "one_hot,plm"
            # There's some ambiguity how you want to combine them, since PLM is typically raw chars
            # For demonstration, let's skip or raise an error:
            raise ValueError("Combining 'plm' with other encodings is not implemented.")
                # Handling Combined Encodings
                # If you want to combine "one_hot" + "plm" in a single run (i.e., encoding="one_hot,plm"), you have to decide how to combine them. For example:
                # Produce the PLM embedding shape (N, L, D_plm)
                # Produce the one-hot shape (N, L, 21)
                # Possibly broadcast or pad the dimension to (N, L, D_plm + 21)
        else:
            raise ValueError(f"err: encountered unknown encoding: {enc}")

    # Concatenate if more than one encoding (excluding plm-only scenario)
    if len(encoded_data) > 1:
        encoded_data = np.concatenate(encoded_data, axis=-1)
    else:
        encoded_data = encoded_data[0]

    # If single sequence, remove the extra dimension
    if single:
        encoded_data = encoded_data[0]

    return encoded_data


def encode_full_dataset(ds_name, encoding):
    # load the dataset
    ds = utils.load_dataset(ds_name=ds_name)
    # encode the data
    encoded_data = encode(encoding=encoding, variants=ds["variant"].tolist(), ds_name=ds_name)
    return encoded_data


def encode_full_dataset_and_save(ds_name, encoding):
    """ encoding a full dataset """
    out_fn = join(constants.DATASETS[ds_name]["ds_dir"], "enc_{}_{}.npy".format(ds_name, encoding))
    if isfile(out_fn):
        print("err: encoded data already exists: {}".format(out_fn))
        return
    encoded_data = encode_full_dataset(ds_name, encoding)
    np.save(out_fn, encoded_data)
    return encoded_data


def main(args):

    if args.ds_name == "all":
        ds_names = constants.DATASETS.keys()
    else:
        ds_names = [args.ds_name]

    if args.encoding == "all":
        encodings = ["one_hot", "aa_index"]
    else:
        encodings = [args.encoding]

    for ds_name in ds_names:
        for encoding in encodings:
            encode_full_dataset_and_save(ds_name, encoding)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ds_name",
                        help="name of the dataset",
                        type=str)
    parser.add_argument("encoding",
                        help="what encoding to use",
                        type=str,
                        choices=["one_hot", "aa_index", "all"])
    main(parser.parse_args())
