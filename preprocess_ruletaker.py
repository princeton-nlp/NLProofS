"""
Preprocess the RuleTaker dataset into a format similar to EntailmentBank.
"""
from common import *
from glob import glob
from lark import Lark
import argparse
import re
import os
import random
import shutil
import json


grammar = """
proof: TRIPLE | "(" conditions " -> (" RULE " % " CONCLUSION "))"

conditions: "(" proof+ ")"

TRIPLE: /triple\d+/

RULE: /rule\d+/

CONCLUSION: /int\d+/

%import common.WS
%ignore WS
"""

parser = Lark(grammar, start="proof")


def extract_proof(raw_proof: Dict[str, Any], name_map: Dict[str, str]) -> str:
    """
    Convert a proof from RuleTaker's format to EntailmentBank's format.
    """
    if re.fullmatch(r"(triple|rule)\d+", raw_proof["representation"]):
        return f"{name_map[raw_proof['representation']]} -> hypothesis;"
    tree = parser.parse(raw_proof["representation"])

    proof_steps = []
    for node in tree.iter_subtrees():
        if node.data == "proof" and len(node.children) == 3:
            assert node.children[1].type == "RULE"
            assert node.children[2].type == "CONCLUSION"
            premises = [name_map[node.children[1].value]]
            for child in node.children[0].children:
                assert child.data == "proof"
                if len(child.children) == 1:
                    premises.append(name_map[child.children[0].value])
                else:
                    assert (
                        len(child.children) == 3
                        and child.children[2].type == "CONCLUSION"
                    )
                    premises.append(child.children[2].value)
            premises_sorted = sorted(
                list(set(premises)),
                key=lambda p: (p.startswith("int"), int(re.search(r"\d+", p).group())),
            )
            conclusion_ident = node.children[2].value
            conclusion_sent = normalize_sentence(
                raw_proof["intermediates"][conclusion_ident]["text"]
            )
            ps = (
                " & ".join(premises_sorted)
                + f" -> {conclusion_ident}: {conclusion_sent};"
            )
            if ps not in proof_steps:
                proof_steps.append(ps)

    proof_steps[-1] = re.sub(r"-> int\d+: .+;$", "-> hypothesis;", proof_steps[-1])
    proof = rename_ints(" ".join(proof_steps))
    return proof


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess the RuleTaker dataset.")
    parser.add_argument(
        "--src",
        type=str,
        default="data/proofwriter-dataset-V2020.12.3/OWA",
        help="Directory of the original RuleTaker (OWA).",
    )
    parser.add_argument(
        "--dst",
        type=str,
        default="data/proofwriter-dataset-V2020.12.3/preprocessed_OWA",
        help="Directory for the data after preprocessing.",
    )
    args = parser.parse_args()
    print(args)

    if os.path.exists(args.dst):
        shutil.rmtree(args.dst)
    os.mkdir(args.dst)

    for src_dir in glob(os.path.join(args.src, "*")):
        print(f"Processing {src_dir}")
        dst_dir = os.path.join(args.dst, os.path.split(src_dir)[-1])
        os.mkdir(dst_dir)

        for split in ("train", "dev", "test"):
            print(f"\t{split}")
            data = []
            inp = os.path.join(src_dir, f"meta-{split}.jsonl")
            if not os.path.exists(inp):
                continue

            for line in open(inp):
                ex = json.loads(line)

                # Rules and facts in RuleTaker are supporting facts.
                supporting_facts = {}
                for ident, triple in list(ex["triples"].items()) + list(
                    ex["rules"].items()
                ):
                    supporting_facts[ident] = (
                        normalize_sentence(ex["sentences"][ex["mappings"][ident]])
                        if "sentences" in ex
                        else normalize_sentence(triple["text"])
                    )

                context_sents = list(set(supporting_facts.values()))
                random.shuffle(context_sents)

                context = " ".join(
                    f"sent{i+1}: {context_sents[i]}" for i in range(len(context_sents))
                )
                name_map = {
                    ident: f"sent{1 + context_sents.index(sent)}"
                    for ident, sent in supporting_facts.items()
                }
                triples = {_["text"] for _ in ex["triples"].values()}

                for question in ex["questions"].values():
                    answer = question["answer"]
                    q = normalize_sentence(question["question"])

                    if split == "train":
                        if answer == "Unknown":
                            # Discard training examples that are neither provable nor unprovable.
                            continue
                        elif answer == True:
                            hypothesis = q
                        else:
                            hypothesis = f"i don't think {q}"
                            answer = True

                        unique_proofs = set()
                        for raw_proof in question["proofsWithIntermediates"]:
                            if len(raw_proof["intermediates"]) > 0:
                                int_sents = {
                                    _["text"]
                                    for _ in raw_proof["intermediates"].values()
                                }
                                if len(triples.intersection(int_sents)) > 0:
                                    # Discard invalid examples with overlapping intermediate conclusions and supporting facts.
                                    continue
                            # Convert the proof from RuleTaker's format to EntailmentBank's format.
                            unique_proofs.add(extract_proof(raw_proof, name_map))

                        proofs = list(unique_proofs)
                        data.append(
                            {
                                "hypothesis": hypothesis,
                                "context": context,
                                "proofs": proofs,
                                "answer": answer,
                                "depth": question["QDep"],
                            }
                        )

                    else:
                        hypothesis = q
                        if "proofsWithIntermediates" not in question:
                            assert answer == "Unknown"
                            proofs = []
                        else:
                            proofs = list(
                                {
                                    extract_proof(raw_proof, name_map)
                                    for raw_proof in question["proofsWithIntermediates"]
                                }
                            )

                        data.append(
                            {
                                "hypothesis": hypothesis,
                                "context": context,
                                "proofs": proofs,
                                "answer": answer,
                                "depth": question["QDep"]
                                if answer != "Unknown"
                                else None,
                            }
                        )

            with open(os.path.join(dst_dir, f"meta-{split}.jsonl"), "wt") as oup:
                for d in data:
                    oup.write(json.dumps(d) + "\n")


if __name__ == "__main__":
    main()
