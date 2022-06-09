"""
Some utility functions.
"""
import re
import torch
import random
import unicodedata
from ete3 import Tree, TreeNode
from transformers import get_cosine_schedule_with_warmup
from typing import *

Example = Dict[str, Any]
Batch = Dict[str, Any]


def normalize(text: str) -> str:
    """
    Deal with unicode-related artifacts.
    """
    return unicodedata.normalize("NFD", text)


def normalize_sentence(text: str) -> str:
    """
    Convert sentences to lowercase and remove the trailing period.
    """
    text = normalize(text).lower().strip()
    if text.endswith("."):
        text = text[:-1].strip()
    return text


def extract_context(ctx: str) -> OrderedDict[str, str]:
    """
    Extract supporting facts from string to dict.
    """
    return OrderedDict(
        {
            ident.strip(): normalize_sentence(sent)
            for ident, sent in re.findall(
                r"(?P<ident>sent\d+): (?P<sent>.+?) (?=sent)", ctx + " sent"
            )
        }
    )


def deserialize(
    hypothesis: str, context: OrderedDict[str, str], proof: str, strict: bool = True
) -> Tree:
    """
    Construct a tree from a text sequence.
    """
    nodes = {}

    for proof_step in proof.split(";"):
        proof_step = proof_step.strip()
        if proof_step == "":
            continue

        if proof_step.count(" -> ") != 1:
            return None
        premises, conclusion = proof_step.split(" -> ")
        m = re.fullmatch(r"\((?P<score>.+?)\) (?P<concl>.+)", conclusion)
        score: Optional[str]
        if m is not None:
            score = m["score"]
            conclusion = m["concl"]
        else:
            score = None

        if conclusion == "hypothesis":
            ident = "hypothesis"
            sent = hypothesis
        else:
            m = re.match(r"(?P<ident>int\d+): (?P<sent>.+)", conclusion)
            if m is None:
                return None
            ident = m["ident"]
            sent = m["sent"]

        nodes[ident] = TreeNode(name=ident)
        nodes[ident].add_feature("sent", sent)
        if score is not None:
            nodes[ident].add_feature("score", score)

        for p in premises.split(" & "):
            if p == ident:
                return None
            elif p not in nodes:
                if re.fullmatch(r"sent\d+", p) is None:
                    return None
                nodes[p] = TreeNode(name=p)
                if p not in context:
                    if strict:
                        return None
                else:
                    nodes[p].add_feature("sent", context[p])
            if nodes[p] not in nodes[ident].children:
                nodes[ident].add_child(nodes[p])

    return nodes.get("hypothesis", None)


def serialize(tree: Tree) -> str:
    """
    Serialize a proof tree as a text sequence.
    """
    if tree is None:
        return "INVALID"
    elif tree.is_leaf():
        return tree.name  # type: ignore

    prev_steps = [serialize(child) for child in tree.children if not child.is_leaf()]
    random.shuffle(prev_steps)

    if len(prev_steps) == 0:
        seq = " & ".join(child.name for child in tree.children)
    else:
        seq = (
            " ".join(prev_steps)
            + " "
            + " & ".join(child.name for child in tree.children)
        )
    seq += " -> "
    if tree.name.startswith("int"):
        seq += f"{tree.name}: {tree.sent};"
    else:
        assert tree.name == "hypothesis"
        seq += f"hypothesis;"

    return seq


def get_internal_nodes(tree: Tree) -> List[TreeNode]:
    """
    Get the internal nodes of a tree as a list.
    """
    return [node for node in tree.traverse() if not node.is_leaf()]


def rename_ints(proof: str) -> str:
    """
    Rename the `int\d+` identifiers in a proof so that they increase from 1.
    """
    assert "INT" not in proof
    mapping: Dict[str, str] = dict()

    while True:
        m = re.search(r"int\d+", proof)
        if m is None:
            break
        s = m.group()
        assert s not in mapping
        dst = f"INT{1 + len(mapping)}"
        mapping[s] = dst
        proof = proof.replace(f"{s}:", f"{dst}:").replace(f"{s} ", f"{dst} ")

    return proof.replace("INT", "int")


def get_optimizers(
    parameters: Iterable[torch.nn.parameter.Parameter],
    lr: float,
    num_warmup_steps: int,
    num_training_steps: int,
) -> Dict[str, Any]:
    """
    Get an AdamW optimizer with linear learning rate warmup and cosine decay.
    """
    optimizer = torch.optim.AdamW(parameters, lr=lr)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "step",
        },
    }
