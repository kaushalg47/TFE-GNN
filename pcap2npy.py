import os
import argparse
import logging
import torch
import dgl
import numpy as np
from typing import Dict, List
from concurrent.futures import ProcessPoolExecutor
from sklearn.model_selection import train_test_split

from utils import show_time, construct_graph, split_flow_Tor_nonoverlapping, split_flow_ISCX
from config import *

# Setting up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

generalConfig = Config()

def construct_dataset_from_bytes_ISCX(dir_path_dict: Dict[str, str], data_type: str, max_segments: int) -> None:
    """
    Constructs datasets from raw byte data for ISCX dataset.

    Args:
        dir_path_dict (Dict[str, str]): Dictionary with categories as keys and directory paths as values.
        data_type (str): Type of data ('payload' or 'header').
        max_segments (int): Maximum number of segments per class.
    """
    train = []
    train_label = []
    test = []
    test_label = []
    TRAIN_FLOW_COUNT = dict()
    TEST_FLOW_COUNT = dict()
    
    for category, dir_path in dir_path_dict.items():
        try:
            file_list = os.listdir(dir_path)
        except FileNotFoundError as e:
            logger.error(f"Directory not found: {dir_path}")
            raise e
        
        data_list = []
        for file in file_list:
            if not file.endswith('.npz'):
                continue
            file_path = os.path.join(dir_path, file)
            logger.info('{} {} Process Starting'.format(show_time(), file_path))
            
            if opt.dataset == 'iscx-tor':
                data_list.extend(
                    split_flow_Tor_nonoverlapping(
                        file_path, category, allow_empty=False, pad_trunc=True, config=config, type=data_type
                    )
                )
            else:
                data_list.extend(
                    split_flow_ISCX(
                        file_path, category, allow_empty=False, pad_trunc=True, config=config, type=data_type
                    )
                )
        
        data_list = data_list[:max_segments]
        split_ind = int(len(data_list) / 10)
        
        # Use train_test_split for flexibility
        data_list_train, data_list_test = train_test_split(
            data_list, test_size=0.1, random_state=42
        )

        train.extend(data_list_train)
        train_label.extend([category] * len(data_list_train))
        test.extend(data_list_test)
        test_label.extend([category] * len(data_list_test))

        TRAIN_FLOW_COUNT[category] = len(data_list_train)
        TEST_FLOW_COUNT[category] = len(data_list_test)
        logger.info(f"Category '{category}': {TRAIN_FLOW_COUNT[category]} training, {TEST_FLOW_COUNT[category]} testing samples")

    # Save the processed data
    if data_type == 'payload':
        save_data(train, train_label, config.TRAIN_DATA)
        save_data(test, test_label, config.TEST_DATA)
    elif data_type == 'header':
        save_data(train, train_label, config.HEADER_TRAIN_DATA)
        save_data(test, test_label, config.HEADER_TEST_DATA)

    logger.info(f"Training Flow Count: {TRAIN_FLOW_COUNT}")
    logger.info(f"Testing Flow Count: {TEST_FLOW_COUNT}")

def save_data(data: List, labels: List, file_path: str) -> None:
    """
    Saves the data and labels into a compressed .npz file.

    Args:
        data (List): List of data samples.
        labels (List): Corresponding labels for the data samples.
        file_path (str): Path to save the .npz file.
    """
    np.savez_compressed(file_path, data=np.array(data), label=np.array(labels))

def filter_relevant_byte_data(bytes_sequence: np.ndarray, data_type: str) -> np.ndarray:
    """
    Filters out unwanted data fields from the byte sequence.

    Args:
        bytes_sequence (np.ndarray): Original byte sequence.
        data_type (str): Type of data ('payload' or 'header').

    Returns:
        np.ndarray: Filtered byte sequence with only relevant fields.
    """
    if data_type == 'payload':
        # Example: Retain IP, Ports, and Packet Size information only
        # Adjust indices based on actual data structure
        filtered_data = bytes_sequence[:, [0, 1, 2, 3, 5]]  # Example indices to keep
    elif data_type == 'header':
        # Remove unnecessary header fields like version and header length
        filtered_data = bytes_sequence[:, [2, 3, 6, 7]]  # Example indices to keep
        
    return filtered_data

def construct_graph_format_data(file_path: str, save_path: str, data_type: str, w_size: int = 5, pmi: int = 1) -> None:
    """
    Constructs graph format data from byte data and saves it in a specified format.

    Args:
        file_path (str): Path to the input file.
        save_path (str): Path where the constructed graphs will be saved.
        data_type (str): Type of data ('payload' or 'header').
        w_size (int): Window size for PMI calculation. Default is 5.
        pmi (int): PMI value to use for graph construction. Default is 1.
    """
    file = np.load(file_path, allow_pickle=True)
    gs = []
    if data_type == 'payload':
        data = file['data'].reshape(-1, config.BYTE_PAD_TRUNC_LENGTH)
    elif data_type == 'header':
        data = file['data'].reshape(-1, config.HEADER_BYTE_PAD_TRUNC_LENGTH)
    label = file['label']

    # Apply filtering to the byte data
    filtered_data = filter_relevant_byte_data(data, data_type)

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(construct_graph, bytes=p, w_size=w_size, k=pmi)
            for p in filtered_data
        ]
        
        # Retrieve the constructed graphs
        for ind, future in enumerate(futures):
            gs.append(future.result())
            if ind % 500 == 0:
                logger.info('{} {} Graphs Constructed'.format(show_time(), ind))

    dgl.save_graphs(save_path, gs, {"glabel": torch.LongTensor(label)})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="dataset", required=True)
    parser.add_argument("--max-seg-per-class", type=int, default=1000, help="Max segments per class")
    parser.add_argument("--pmi-window-size", type=int, default=5, help="PMI window size")
    parser.add_argument("--data-type", type=str, choices=['payload', 'header'], default='payload', help="Type of data to process")
    opt = parser.parse_args()

    # Load the appropriate configuration
    if opt.dataset == 'iscx-vpn':
        config = ISCXVPNConfig()
    elif opt.dataset == 'iscx-nonvpn':
        config = ISCXNonVPNConfig()
    elif opt.dataset == 'iscx-tor':
        config = ISCXTorConfig()
    elif opt.dataset == 'iscx-nontor':
        config = ISCXNonTorConfig()
    else:
        raise Exception('Dataset Error')

    # Construct datasets and graphs
    construct_dataset_from_bytes_ISCX(
        dir_path_dict=config.DIR_PATH_DICT, 
        data_type=opt.data_type, 
        max_segments=opt.max_seg_per_class
    )
    construct_graph_format_data(
        file_path=config.TRAIN_DATA, 
        save_path=config.TRAIN_GRAPH_DATA, 
        data_type=opt.data_type, 
        w_size=opt.pmi_window_size
    )
    construct_graph_format_data(
        file_path=config.TEST_DATA, 
        save_path=config.TEST_GRAPH_DATA, 
        data_type=opt.data_type, 
        w_size=opt.pmi_window_size
    )
