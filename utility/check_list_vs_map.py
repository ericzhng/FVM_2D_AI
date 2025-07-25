import numpy as np
import time

node_tags = np.load("nodetags.npy")
nnode = len(node_tags)

start_time_block1 = time.perf_counter()

max_tag = np.max(node_tags)
node_tag_map = np.full(max_tag + 1, -1, dtype=np.int32)
node_tag_map[node_tags] = np.arange(nnode, dtype=np.int32)

end_time_block1 = time.perf_counter()
elapsed_time_block1 = end_time_block1 - start_time_block1
print(f"Time taken for Block 1: {elapsed_time_block1:.6f} seconds")


start_time_block2 = time.perf_counter()

node_tag_to_index = {tag: i for i, tag in enumerate(node_tags)}

end_time_block2 = time.perf_counter()
elapsed_time_block2 = end_time_block2 - start_time_block2
print(f"Time taken for Block 2: {elapsed_time_block2:.6f} seconds")
