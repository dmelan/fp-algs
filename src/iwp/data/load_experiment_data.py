import glob
import logging
import os
import re

import numpy as np

from iwp.data.load_from_dat import load_sparse_matrix, load_vector

logger = logging.getLogger("iwp")


def load_experiment_data(data_path: str) -> tuple:
    """
    Load the experiment data from the specified path.

    Args:
        data_path (str): Parsed command line arguments containing the data path.

    Returns:
        tuple: A tuple containing:
            - A (scipy.sparse.coo_matrix): Sparse matrix A.
            - B_list (list of scipy.sparse.coo_matrix): List of sparse matrices B
            - C (scipy.sparse.coo_matrix): Sparse matrix C.
            - d_list (list of numpy.ndarray): List of data vectors d_i corresponding to each B matrix.
            - m (numpy.ndarray): Vector m.
    """
    # Load A, C, and m
    A = load_sparse_matrix(os.path.join(data_path, "MatrixABorn.dat"), is_complex=True)
    C = load_sparse_matrix(os.path.join(data_path, "MatrixC.dat"), is_complex=False)
    m = load_vector(os.path.join(data_path, "Vectorm.dat"), is_complex=True)

    # Find all files starting with "MatrixB_" in the experiment path, ordered by
    # their numeric source index rather than lexicographically: plain sorting
    # gives MatrixB_0, MatrixB_1, MatrixB_10, ... so from I >= 10 onwards B_list[i]
    # would no longer correspond to the Vectord_{i}.dat loaded below.
    B_files = sorted(
        glob.glob(os.path.join(data_path, "MatrixB_*")),
        key=lambda fname: int(re.search(r"MatrixB_(\d+)", os.path.basename(fname)).group(1)),
    )
    logger.info(f"Found {len(B_files)} B matrices.")

    # Load all B matrices
    B_list = [load_sparse_matrix(fname, is_complex=True) for fname in B_files]

    # Load data vectors d_i (if available, else use zeros and raise a warning)
    d_list = []
    for i, B in enumerate(B_list):
        dfile = os.path.join(data_path, f"Vectord_{i}.dat")
        if os.path.exists(dfile):
            d_vec = load_vector(dfile, is_complex=True)
        else:
            logger.warning(f"Data vector for B matrix {i} not found, using zeros.")
            d_vec = np.zeros(B.shape[0], dtype=np.complex128)
        d_list.append(d_vec)

    logger.info(f"Loaded data.")

    try:
        assert A.shape[0] == A.shape[1], "Matrix A must be square."
        assert all(
            B.shape[0] == A.shape[0] for B in B_list
        ), "All B matrices must have the same number of rows as A."
        assert all(
            B.shape[1] == m.shape[0] for B in B_list
        ), "All B matrices must have the same number of columns as the dimension of m."
        assert all(
            d.shape[0] == C.shape[0] for d in d_list
        ), "All d vectors must have the same number of rows as C."
        assert (
            C.shape[1] == A.shape[0]
        ), "Matrix C must have the same number of columns as A."

    except AssertionError as ae:
        logger.error(f"Data validation failed: {ae}")
        raise ValueError(f"Data validation failed: {ae}")

    return A, B_list, C, d_list, m
