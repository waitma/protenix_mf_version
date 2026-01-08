# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Union

import biotite.structure.io as strucio
import numpy as np
import pandas as pd
import torch
from biotite.structure import AtomArray

from protenix.data.msa_featurizer import MSAFeaturizer
from protenix.data.parser import DistillationMMCIFParser, MMCIFParser
from protenix.data.tokenizer import AtomArrayTokenizer, TokenArray
from protenix.utils.cropping import CropData
from protenix.utils.file_io import load_gzip_pickle

torch.multiprocessing.set_sharing_strategy("file_system")


def get_cdr_indices_from_sequence(
    sequence: str,
    scheme: str = "imgt",
) -> set:
    """
    Get CDR indices from antibody sequence using ANARCI/abnumber.
    
    Args:
        sequence: Amino acid sequence (1-letter code)
        scheme: Numbering scheme ('imgt', 'chothia', 'kabat')
    
    Returns:
        Set of CDR indices (0-based positions in sequence)
    """
    try:
        from abnumber import Chain as AbChain
        use_anarci = False
    except ImportError:
        try:
            from anarci import run_anarci
            use_anarci = True
        except ImportError:
            raise ImportError("Please install abnumber or anarci: pip install abnumber")
    
    cdr_indices = set()
    
    if not use_anarci:
        try:
            ab_chain = AbChain(sequence, scheme=scheme)
            fr1_len = len(ab_chain.fr1_seq)
            cdr1_len = len(ab_chain.cdr1_seq)
            fr2_len = len(ab_chain.fr2_seq)
            cdr2_len = len(ab_chain.cdr2_seq)
            fr3_len = len(ab_chain.fr3_seq)
            cdr3_len = len(ab_chain.cdr3_seq)
            
            # CDR1
            cdr1_start = fr1_len
            cdr_indices.update(range(cdr1_start, cdr1_start + cdr1_len))
            # CDR2
            cdr2_start = fr1_len + cdr1_len + fr2_len
            cdr_indices.update(range(cdr2_start, cdr2_start + cdr2_len))
            # CDR3
            cdr3_start = fr1_len + cdr1_len + fr2_len + cdr2_len + fr3_len
            cdr_indices.update(range(cdr3_start, cdr3_start + cdr3_len))
        except Exception:
            pass
    else:
        try:
            results = run_anarci([('seq', sequence)], scheme=scheme, allowed_species=['human', 'mouse'])
            if results and results[0] and results[0][0]:
                numbering = results[0][0][0]
                for pos_idx, (pos, aa) in enumerate(numbering):
                    pos_num = pos[0]
                    if scheme == 'imgt':
                        if 27 <= pos_num <= 38:  # CDR1
                            cdr_indices.add(pos_idx)
                        elif 56 <= pos_num <= 65:  # CDR2
                            cdr_indices.add(pos_idx)
                        elif 105 <= pos_num <= 117:  # CDR3
                            cdr_indices.add(pos_idx)
        except Exception:
            pass
    
    return cdr_indices


def generate_cdr_mask(
    token_array: TokenArray,
    atom_array: AtomArray,
    cdr_chain_ids: list[str] = None,
    sequences: dict[str, str] = None,
    scheme: str = "imgt",
) -> np.ndarray:
    """
    Generate CDR mask for tokens using ANARCI antibody numbering.
    
    Args:
        token_array: TokenArray object
        atom_array: AtomArray object
        cdr_chain_ids: List of chain IDs for antibody chains to identify CDRs.
                      If None, returns all False mask.
        sequences: Dict mapping label_entity_id to canonical_sequence (from bioassembly_dict).
                  If provided, uses these sequences instead of extracting from tokens.
        scheme: Numbering scheme for ANARCI ('imgt', 'chothia', 'kabat', etc.)
    
    Returns:
        np.ndarray: Boolean mask of shape [N_token], True for CDR positions.
    """
    from protenix.data.constants import PRO_STD_RESIDUES
    
    n_tokens = len(token_array)
    cdr_mask = np.zeros(n_tokens, dtype=bool)
    
    if cdr_chain_ids is None or len(cdr_chain_ids) == 0:
        return cdr_mask
    
    # Get chain_id and label_entity_id for each token
    centre_atom_indices = token_array.get_annotation("centre_atom_index")
    token_chain_ids = atom_array[centre_atom_indices].chain_id
    
    # Token value to 1-letter mapping (fallback if sequences not provided)
    id_to_3letter = {v: k for k, v in PRO_STD_RESIDUES.items()}
    aa_3to1 = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
        'UNK': 'X'
    }
    
    for chain_id in cdr_chain_ids:
        chain_token_mask = token_chain_ids == chain_id
        chain_token_indices = np.where(chain_token_mask)[0]
        
        if len(chain_token_indices) == 0:
            continue
        
        # Get sequence for this chain
        sequence = None
        
        # Try to get sequence from bioassembly_dict sequences
        if sequences is not None:
            # Get label_entity_id for this chain
            chain_atoms = atom_array[atom_array.chain_id == chain_id]
            if len(chain_atoms) > 0 and hasattr(chain_atoms, 'label_entity_id'):
                entity_id = str(chain_atoms.label_entity_id[0])
                sequence = sequences.get(entity_id, None)
        
        # Fallback: extract sequence from token values
        if sequence is None:
            sequence_chars = []
            for idx in chain_token_indices:
                token = token_array[idx]
                token_value = token.value
                three_letter = id_to_3letter.get(token_value, 'UNK')
                one_letter = aa_3to1.get(three_letter, 'X')
                sequence_chars.append(one_letter)
            sequence = ''.join(sequence_chars)
        
        if len(sequence) == 0 or all(c == 'X' for c in sequence):
            continue
        
        # Get CDR indices from sequence
        cdr_indices_in_chain = get_cdr_indices_from_sequence(sequence, scheme)
        
        # Map chain indices to global token indices
        for local_idx in cdr_indices_in_chain:
            if local_idx < len(chain_token_indices):
                global_idx = chain_token_indices[local_idx]
                cdr_mask[global_idx] = True
    
    return cdr_mask


class DataPipeline(object):
    """
    DataPipeline class provides static methods to handle various data processing tasks related to bioassembly structures.
    """

    @staticmethod
    def get_data_from_mmcif(
        mmcif: Union[str, Path],
        pdb_cluster_file: Union[str, Path, None] = None,
        dataset: str = "WeightedPDB",
        interface_radius: float = 5,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """
        Get raw data from mmcif with tokenizer and a list of chains and interfaces for sampling.

        Args:
            mmcif (Union[str, Path]): The raw mmcif file.
            pdb_cluster_file (Union[str, Path, None], optional): Cluster info txt file. Defaults to None.
            dataset (str, optional): The dataset type, either "WeightedPDB" or "Distillation". Defaults to "WeightedPDB".
            interface_radius (float, optional): The radius of the interface. Defaults to 5.
        Returns:
            tuple[list[dict[str, Any]], dict[str, Any]]:
                sample_indices_list (list[dict[str, Any]]): The sample indices list (each one is a chain or an interface).
                bioassembly_dict (dict[str, Any]): The bioassembly dict with sequence, atom_array, and token_array.
        """
        try:
            if dataset == "WeightedPDB":
                parser = MMCIFParser(mmcif_file=mmcif)
                bioassembly_dict = parser.get_bioassembly()
            elif dataset == "Distillation":
                parser = DistillationMMCIFParser(mmcif_file=mmcif)
                bioassembly_dict = parser.get_structure_dict()
            else:
                raise NotImplementedError(
                    'Unsupported "dataset", please input either "WeightedPDB" or "Distillation".'
                )

            sample_indices_list = parser.make_indices(
                bioassembly_dict=bioassembly_dict,
                pdb_cluster_file=pdb_cluster_file,
                interface_radius=interface_radius,
            )
            if len(sample_indices_list) == 0:
                # empty indices and AtomArray
                return [], bioassembly_dict

            atom_array = bioassembly_dict["atom_array"]
            atom_array.set_annotation(
                "resolution", [parser.resolution] * len(atom_array)
            )

            tokenizer = AtomArrayTokenizer(atom_array)
            token_array = tokenizer.get_token_array()
            bioassembly_dict["msa_features"] = None
            bioassembly_dict["template_features"] = None

            bioassembly_dict["token_array"] = token_array
            return sample_indices_list, bioassembly_dict

        except Exception as e:
            logging.warning("Gen data failed for %s due to %s", mmcif, e)
            return [], {}

    @staticmethod
    def get_label_entity_id_to_asym_id_int(atom_array: AtomArray) -> dict[str, int]:
        """
        Get a dictionary that associates each label_entity_id with its corresponding asym_id_int.

        Args:
            atom_array (AtomArray): AtomArray object

        Returns:
            dict[str, int]: label_entity_id to its asym_id_int
        """
        entity_to_asym_id = defaultdict(set)
        for atom in atom_array:
            entity_id = atom.label_entity_id
            entity_to_asym_id[entity_id].add(atom.asym_id_int)
        return entity_to_asym_id

    @staticmethod
    def get_data_bioassembly(
        bioassembly_dict_fpath: Union[str, Path],
    ) -> dict[str, Any]:
        """
        Get the bioassembly dict.

        Args:
            bioassembly_dict_fpath (Union[str, Path]): The path to the bioassembly dictionary file.

        Returns:
            dict[str, Any]: The bioassembly dict with sequence, atom_array and token_array.

        Raises:
            AssertionError: If the bioassembly dictionary file does not exist.
        """
        assert os.path.exists(
            bioassembly_dict_fpath
        ), f"File not exists {bioassembly_dict_fpath}"
        bioassembly_dict = load_gzip_pickle(bioassembly_dict_fpath)

        return bioassembly_dict

    @staticmethod
    def _map_ref_chain(
        one_sample: pd.Series, bioassembly_dict: dict[str, Any]
    ) -> list[int]:
        """
        Map the chain or interface chain_x_id to the reference chain asym_id.

        Args:
            one_sample (pd.Series): A dict of one chain or interface from indices list.
            bioassembly_dict (dict[str, Any]): The bioassembly dict with sequence, atom_array and token_array.

        Returns:
            list[int]: A list of asym_id_lnt of the chosen chain or interface, length 1 or 2.
        """
        atom_array = bioassembly_dict["atom_array"]
        ref_chain_indices = []
        for chain_id_field in ["chain_1_id", "chain_2_id"]:
            chain_id = one_sample[chain_id_field]
            assert np.isin(
                chain_id, np.unique(atom_array.chain_id)
            ), f"PDB {bioassembly_dict['pdb_id']} {chain_id_field}:{chain_id} not in atom_array"
            chain_asym_id = atom_array[atom_array.chain_id == chain_id].asym_id_int[0]
            ref_chain_indices.append(chain_asym_id)
            if one_sample["type"] == "chain":
                break
        return ref_chain_indices

    @staticmethod
    def get_msa_raw_features(
        bioassembly_dict: dict[str, Any],
        selected_indices: np.ndarray,
        msa_featurizer: Optional[MSAFeaturizer],
    ) -> Optional[dict[str, np.ndarray]]:
        """
        Get tokenized MSA features of the bioassembly

        Args:
            bioassembly_dict (Mapping[str, Any]): The bioassembly dict with sequence, atom_array and token_array.
            selected_indices (torch.Tensor): Cropped token indices.
            msa_featurizer (MSAFeaturizer): MSAFeaturizer instance.

        Returns:
            Optional[dict[str, np.ndarray]]: The tokenized MSA features of the bioassembly.
        """
        if msa_featurizer is None:
            return None

        entity_to_asym_id_int = dict(
            DataPipeline.get_label_entity_id_to_asym_id_int(
                bioassembly_dict["atom_array"]
            )
        )

        msa_feats = msa_featurizer(
            bioassembly_dict=bioassembly_dict,
            selected_indices=selected_indices,
            entity_to_asym_id_int=entity_to_asym_id_int,
        )

        return msa_feats

    @staticmethod
    def get_template_raw_features(
        bioassembly_dict: dict[str, Any],
        selected_indices: np.ndarray,
        template_featurizer: None,
    ) -> Optional[dict[str, np.ndarray]]:
        """
        Get tokenized template features of the bioassembly.

        Args:
            bioassembly_dict (dict[str, Any]): The bioassembly dict with sequence, atom_array and token_array.
            selected_indices (np.ndarray): Cropped token indices.
            template_featurizer (None): Placeholder for the template featurizer.

        Returns:
            Optional[dict[str, np.ndarray]]: The tokenized template features of the bioassembly,
                or None if the template featurizer is not provided.
        """
        if template_featurizer is None:
            return None

        entity_to_asym_id_int = dict(
            DataPipeline.get_label_entity_id_to_asym_id_int(
                bioassembly_dict["atom_array"]
            )
        )

        template_feats = template_featurizer(
            bioassembly_dict=bioassembly_dict,
            selected_indices=selected_indices,
            entity_to_asym_id_int=entity_to_asym_id_int,
        )
        return template_feats

    @staticmethod
    def crop(
        one_sample: pd.Series,
        bioassembly_dict: dict[str, Any],
        crop_size: int,
        msa_featurizer: Optional[MSAFeaturizer],
        template_featurizer: None,
        method_weights: list[float] = [0.2, 0.4, 0.4],
        contiguous_crop_complete_lig: bool = False,
        spatial_crop_complete_lig: bool = False,
        drop_last: bool = False,
        remove_metal: bool = False,
        antibody_chain_ids: list[int] = None,
        add_antigen: bool = True,
        min_neighborhood: int = 0,
        max_neighborhood: int = 40,
        cdr_mask: np.ndarray = None,
    ) -> tuple[str, TokenArray, AtomArray, dict[str, Any], dict[str, Any], int, np.ndarray]:
        """
        Crop data based on the crop size and reference chain indices.

        Args:
            one_sample (pd.Series): A dict of one chain or interface from indices list.
            bioassembly_dict (dict[str, Any]): A dict of bioassembly dict with sequence, atom_array and token_array.
            crop_size (int): the crop size.
            msa_featurizer (MSAFeaturizer): Default to an empty replacement for msa featurizer.
            template_featurizer (None): Placeholder for the template featurizer.
            method_weights (list[float]): The weights corresponding to these three cropping methods:
                                          ["ContiguousCropping", "SpatialCropping", "SpatialInterfaceCropping"].
            contiguous_crop_complete_lig (bool): Whether to crop the complete ligand in ContiguousCropping method.
            spatial_crop_complete_lig (bool): Whether to crop the complete ligand in SpatialCropping method.
            drop_last (bool): Whether to drop the last fragment in ContiguousCropping.
            remove_metal (bool): Whether to remove metal atoms from the crop.

        Returns:
            tuple[str, TokenArray, AtomArray, dict[str, Any], dict[str, Any]]:
                crop_method (str): The crop method.
                cropped_token_array (TokenArray): TokenArray after cropping.
                cropped_atom_array (AtomArray): AtomArray after cropping.
                cropped_msa_features (dict[str, Any]): The cropped msa features.
                cropped_template_features (dict[str, Any]): The cropped template features.
        """
        if crop_size <= 0:
            selected_indices = None
            # Prepare msa
            msa_features = DataPipeline.get_msa_raw_features(
                bioassembly_dict=bioassembly_dict,
                selected_indices=selected_indices,
                msa_featurizer=msa_featurizer,
            )
            # Prepare template
            template_features = DataPipeline.get_template_raw_features(
                bioassembly_dict=bioassembly_dict,
                selected_indices=selected_indices,
                template_featurizer=template_featurizer,
            )
            return (
                "no_crop",
                bioassembly_dict["token_array"],
                bioassembly_dict["atom_array"],
                msa_features or {},
                template_features or {},
                -1,
                cdr_mask if cdr_mask is not None else np.zeros(len(bioassembly_dict["token_array"]), dtype=bool),
            )

        ref_chain_indices = DataPipeline._map_ref_chain(
            one_sample=one_sample, bioassembly_dict=bioassembly_dict
        )

        crop = CropData(
            crop_size=crop_size,
            ref_chain_indices=ref_chain_indices,
            token_array=bioassembly_dict["token_array"],
            atom_array=bioassembly_dict["atom_array"],
            method_weights=method_weights,
            contiguous_crop_complete_lig=contiguous_crop_complete_lig,
            spatial_crop_complete_lig=spatial_crop_complete_lig,
            drop_last=drop_last,
            remove_metal=remove_metal,
            antibody_chain_ids=antibody_chain_ids,
            add_antigen=add_antigen,
            min_neighborhood=min_neighborhood,
            max_neighborhood=max_neighborhood,
            cdr_mask=cdr_mask,
        )
        # Get crop method
        crop_method = crop.random_crop_method()
        # Get crop indices based crop method
        selected_indices, reference_token_index = crop.get_crop_indices(
            crop_method=crop_method
        )
        # Prepare msa
        msa_features = DataPipeline.get_msa_raw_features(
            bioassembly_dict=bioassembly_dict,
            selected_indices=selected_indices,
            msa_featurizer=msa_featurizer,
        )
        # Prepare template
        template_features = DataPipeline.get_template_raw_features(
            bioassembly_dict=bioassembly_dict,
            selected_indices=selected_indices,
            template_featurizer=template_featurizer,
        )

        (
            cropped_token_array,
            cropped_atom_array,
            cropped_msa_features,
            cropped_template_features,
        ) = crop.crop_by_indices(
            selected_token_indices=selected_indices,
            msa_features=msa_features,
            template_features=template_features,
        )

        if crop_method == "ContiguousCropping":
            resovled_atom_num = cropped_atom_array.is_resolved.sum()
            # The criterion of "more than 4 atoms" is chosen arbitrarily.
            assert (
                resovled_atom_num > 4
            ), f"{resovled_atom_num=} <= 4 after ContiguousCropping"

        # Crop CDR mask
        if cdr_mask is not None:
            cropped_cdr_mask = cdr_mask[selected_indices]
        else:
            cropped_cdr_mask = np.zeros(len(cropped_token_array), dtype=bool)

        return (
            crop_method,
            cropped_token_array,
            cropped_atom_array,
            cropped_msa_features,
            cropped_template_features,
            reference_token_index,
            cropped_cdr_mask,
        )

    @staticmethod
    def save_atoms_to_cif(
        output_cif_file: str, atom_array: AtomArray, include_bonds: bool = False
    ) -> None:
        """
        Save atom array data to a CIF file.

        Args:
            output_cif_file (str): The output path for saving atom array in cif
            atom_array (AtomArray): The atom array to be saved
            include_bonds (bool): Whether to include bond information in the CIF file. Default is False.

        """
        strucio.save_structure(
            file_path=output_cif_file,
            array=atom_array,
            data_block=os.path.basename(output_cif_file).replace(".cif", ""),
            include_bonds=include_bonds,
        )
