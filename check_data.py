"""
Sanity checks on the data.
"""

import os
from hashlib import md5


def check_md5(filename: str, gt_hashcode: str) -> bool:
    """
    Check the MD5 of a file against the ground truth.
    """
    if not os.path.exists(filename):
        return False
    # The file could be large.
    # See https://stackoverflow.com/questions/48122798/oserror-errno-22-invalid-argument-when-reading-a-huge-file.
    inp = open(filename, "rb")
    hasher = md5()
    while True:
        block = inp.read(64 * (1 << 20))
        if not block:
            break
        hasher.update(block)
    return hasher.hexdigest() == gt_hashcode


files = {
    "data/entailment_trees_emnlp2021_data_v3/dataset/task_1/train.jsonl": "db9c55c3806712d1aac7ac228f9e3532",
    "data/entailment_trees_emnlp2021_data_v3/dataset/task_1/dev.jsonl": "c2fd696f2bf6b4f6e90a04a1c2d9cf24",
    "data/entailment_trees_emnlp2021_data_v3/dataset/task_1/test.jsonl": "3d4b63031e3f4e50be3dbe1c989b5d3a",
    "data/entailment_trees_emnlp2021_data_v3/dataset/task_2/train.jsonl": "7bb4468d4f0c61343f48768bce957f53",
    "data/entailment_trees_emnlp2021_data_v3/dataset/task_2/dev.jsonl": "25407a4e62dcc837672055babcd667b8",
    "data/entailment_trees_emnlp2021_data_v3/dataset/task_2/test.jsonl": "c0984382cdbb7f3e7a3627f10c288315",
    "data/entailment_trees_emnlp2021_data_v3/dataset/task_3/train.jsonl": "568ca63286add189eae0dd8f68b57f40",
    "data/entailment_trees_emnlp2021_data_v3/dataset/task_3/dev.jsonl": "65a483cb83918ebb7759bdd40993e472",
    "data/entailment_trees_emnlp2021_data_v3/dataset/task_3/test.jsonl": "5ee45454aa6a34d05c067440f24b7430",
    "data/proofwriter-dataset-V2020.12.3/OWA/depth-3ext/meta-train.jsonl": "2ab7bb67319ae0e90417a344af11cdf5",
    "data/proofwriter-dataset-V2020.12.3/OWA/depth-3ext/meta-dev.jsonl": "1eacbe38239b0ae4ccce5fc3ef0000db",
    "data/proofwriter-dataset-V2020.12.3/OWA/depth-3ext/meta-test.jsonl": "95a7ba872eb31f67308d085dca736382",
}

if __name__ == "__main__":
    for f, h in files.items():
        print(f"Checking {f}.")
        if not check_md5(f, h):
            print(f"{f} does not exist or has an incorrect MD5.")
            break
    else:
        print("The data looks good!")
