import numpy as np
import time
import unittest
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestEulerEquations(unittest.TestCase):
    """
    Test suite for the euler_equations module.
    """

    def setUp(self):
        self.node_tags = np.load("tests/nodetags.npy")
        self.nnode = len(self.node_tags)

    def test_compare(self):
        start_time_block1 = time.perf_counter()

        max_tag = np.max(self.node_tags)
        node_tag_map = np.full(max_tag + 1, -1, dtype=np.int32)
        node_tag_map[self.node_tags] = np.arange(self.nnode, dtype=np.int32)

        end_time_block1 = time.perf_counter()
        elapsed_time_block1 = end_time_block1 - start_time_block1
        print(f"\nTime taken for Block 1: {elapsed_time_block1:.6f} seconds")

        start_time_block2 = time.perf_counter()

        node_tag_to_index = {tag: i for i, tag in enumerate(self.node_tags)}

        end_time_block2 = time.perf_counter()
        elapsed_time_block2 = end_time_block2 - start_time_block2
        print(f"\nTime taken for Block 2: {elapsed_time_block2:.6f} seconds")


if __name__ == "__main__":
    unittest.main()
