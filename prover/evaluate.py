"""
Utilities for evaluation.
"""
from common import *
import argparse
import itertools
from tqdm import tqdm
from collections import defaultdict
import datasets
import json
import tempfile
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def gather_descendants(tree: TreeNode) -> DefaultDict[str, Set[str]]:
    descendants = defaultdict(set)

    for node in tree.traverse("postorder"):
        if node.name.startswith("sent"):
            descendants[node.name].add(node.name)
        for child in node.children:
            descendants[node.name].update(descendants[child.name])

    return descendants


def intersection_over_union(a: Set[Any], b: Set[Any]) -> float:
    return len(a.intersection(b)) / len(a.union(b))


def align(tree_pred: TreeNode, tree_gt: TreeNode) -> Dict[str, str]:
    """
    Align nodes in the predicted tree to nodes in the ground truth tree.
    """
    alignment: Dict[str, str] = {}
    if tree_pred is None:
        return alignment

    descendants_pred = gather_descendants(tree_pred)
    descendants_gt = gather_descendants(tree_gt)

    for node in tree_pred.traverse():
        if node.name.startswith("sent"):
            alignment[node.name] = (
                node.name if tree_gt.get_leaves_by_name(node.name) != [] else "dummy"
            )
        else:
            max_iou = 0.0
            max_node = None
            for node_gt in tree_gt.traverse():
                if node_gt.name.startswith("sent"):
                    continue
                iou = intersection_over_union(
                    descendants_gt[node_gt.name], descendants_pred[node.name]
                )
                if iou > max_iou:
                    max_iou = iou
                    max_node = node_gt
            alignment[node.name] = max_node.name if max_node is not None else "dummy"

    return alignment


def calculate_f1(tp: float, fp: float, p: float) -> float:
    prec = 1.0 if tp + fp == 0 else tp / (tp + fp)
    rec = 1.0 if p == 0 else tp / p
    return 0.0 if prec + rec == 0.0 else 2 * prec * rec / (prec + rec)


def evaluate_leaves(tree_pred: TreeNode, tree_gt: TreeNode) -> Tuple[float, float]:
    if tree_pred is None:
        return 0.0, 0.0

    sents_gt = {node.name for node in tree_gt.get_leaves()}

    sents_pred = set()
    for node in tree_pred.get_leaves():
        if node.name not in sents_gt:
            node.add_feature("error", "leaf")
        sents_pred.add(node.name)

    tp = len(sents_pred.intersection(sents_gt))
    fp = len(sents_pred.difference(sents_gt))
    fn = len(sents_gt.difference(sents_pred))
    prec = 1.0 if tp + fp == 0 else tp / (tp + fp)
    rec = 1.0 if tp + fn == 0 else tp / (tp + fn)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0.0 else 0.0
    em = float(f1 == 1.0)

    return em, f1


def evaluate_steps(
    tree_pred: TreeNode, tree_gt: TreeNode, alignment: Dict[str, str]
) -> Tuple[float, float]:
    if tree_pred is None:
        return 0.0, 0.0

    steps_gt = set()
    for node in tree_gt.traverse():
        if node.is_leaf():
            continue
        steps_gt.add(
            (tuple(sorted([child.name for child in node.children])), node.name)
        )

    steps_pred = set()
    for node in tree_pred.traverse():
        if node.is_leaf():
            continue
        step = (
            tuple(sorted([alignment[child.name] for child in node.children])),
            alignment[node.name],
        )
        steps_pred.add(step)
        if step not in steps_gt:
            node.add_feature("error", "step")

    tp = len(steps_pred.intersection(steps_gt))
    fp = len(steps_pred.difference(steps_gt))
    fn = len(steps_gt.difference(steps_pred))
    prec = 1.0 if tp + fp == 0 else tp / (tp + fp)
    rec = 1.0 if tp + fn == 0 else tp / (tp + fn)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0.0 else 0.0
    em = float(f1 == 1.0)

    return em, f1


def evaluate_intermediates(
    tree_pred: TreeNode, tree_gt: TreeNode, alignment: Dict[str, str], bleurt: Any
) -> Tuple[float, float]:
    if tree_pred is None:
        return 0.0, 0.0

    ints_gt = {node.name for node in tree_gt.traverse() if not node.is_leaf()}

    tp = fp = 0
    nodes_pred = []
    sents_pred = []
    nodes_gt = []
    sents_gt = []

    for node in tree_pred.traverse():
        if node.is_leaf():
            continue
        if alignment[node.name] == "dummy":
            fp += 1
            node.add_feature("error", "intermediate")
        else:
            nodes_pred.append(node)
            nodes_gt.append(tree_gt.search_nodes(name=alignment[node.name])[0])
            sents_pred.append(node.sent)
            sents_gt.append(nodes_gt[-1].sent)

    similarities = bleurt.compute(predictions=sents_pred, references=sents_gt)["scores"]

    ints = set()

    for k, s in enumerate(similarities):
        if s >= 0.28:
            tp += 1
            ints.add(nodes_gt[k].name)
        else:
            fp += 1
            nodes_pred[k].add_feature("error", "intermediate")

    prec = 1.0 if tp + fp == 0 else tp / (tp + fp)
    rec = 1.0 if len(ints) == 0 else len(ints) / len(ints_gt)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0.0 else 0.0
    em = float(f1 == 1.0)

    return em, f1


def highlight_errors(tree: TreeNode) -> None:
    "Highlight errors in the tree."
    for node in tree.traverse():
        name = TextFace(node.name)
        sent = TextFace(node.sent)
        if hasattr(node, "error"):
            if node.error == "step":
                node.set_style(NodeStyle(vt_line_width=10, vt_line_color="LightSalmon"))
                for child in node.children:
                    child.set_style(
                        NodeStyle(hz_line_width=10, hz_line_color="LightSalmon")
                    )
            else:
                sent.background.color = name.background.color = "LightSalmon"
        node.add_face(name, column=0)
        node.add_face(sent, column=0)
        if hasattr(node, "score"):
            node.add_face(TextFace(node.score), column=0)


def get_tree_style(proof: str, score: Optional[float], is_gt: bool) -> TreeStyle:
    style = TreeStyle()
    style.branch_vertical_margin = 100
    style.show_leaf_name = False
    style.show_scale = False
    style.title.add_face(TextFace("Ground truth" if is_gt else "Predicted"), column=0)
    style.title.add_face(TextFace(proof), column=0)
    if score is not None:
        style.title.add_face(TextFace(f"Verifier score: {score}"), column=0)
    return style


def evaluate_example(ex: Example, bleurt: Any, output_pdf: bool) -> Any:
    hypothesis = normalize(ex["hypothesis"]).strip()
    context = ex["context"]
    proof_gt = normalize(ex["proof_gt"]).strip()
    tree_gt = deserialize(hypothesis, context, proof_gt, strict=False)
    proof_gt = serialize(tree_gt)
    assert tree_gt is not None
    em = {}
    f1 = {}

    with tempfile.TemporaryDirectory() as dir_path:
        if output_pdf:
            highlight_errors(tree_gt)
            file_path = os.path.join(dir_path, "tree_gt.pdf")
            style = get_tree_style(
                proof_gt, ex.get("verifier_score_gt", None), is_gt=True
            )
            tree_gt.render(file_path, tree_style=style)
            pdf_pages = [PdfFileReader(open(file_path, "rb")).getPage(0)]

        tree_pred = deserialize(hypothesis, context, ex["proof_pred"])
        proof_pred = serialize(tree_pred)
        alignment = align(tree_pred, tree_gt)

        em_leaves, f1_leaves = evaluate_leaves(tree_pred, tree_gt)
        em["leaves"] = em_leaves
        f1["leaves"] = f1_leaves

        em_steps, f1_steps = evaluate_steps(tree_pred, tree_gt, alignment)
        em["steps"] = em_steps
        f1["steps"] = f1_steps

        if bleurt is not None:
            em_intermediates, f1_intermediates = evaluate_intermediates(
                tree_pred, tree_gt, alignment, bleurt
            )
            em["intermediates"] = em_intermediates
            f1["intermediates"] = f1_intermediates

        correct = (em["leaves"] == 1.0) and (em["steps"] == 1.0)
        if bleurt is not None:
            correct = correct and (em["intermediates"] == 1.0)
        em["proof"] = f1["proof"] = 1.0 if correct else 0.0

        if output_pdf and tree_pred is not None:
            highlight_errors(tree_pred)
            file_path = os.path.join(dir_path, f"tree_pred.pdf")
            style = get_tree_style(
                proof_pred,
                ex["verifier_scores_pred"] if "verifier_scores_pred" in ex else None,
                is_gt=False,
            )
            tree_pred.render(file_path, tree_style=style)
            pdf_pages.append(PdfFileReader(file_path).getPage(0))

    tree_depth = int(tree_gt.get_farthest_leaf()[1])
    tree_size = 1 + len(tree_gt.get_descendants())

    if output_pdf:
        margin = 50
        total_width = np.max([page.mediaBox.upperRight[0] for page in pdf_pages])
        total_height = np.sum(
            [page.mediaBox.upperRight[1] + margin for page in pdf_pages]
        )
        combined_page = PageObject.createBlankPage(None, total_width, total_height)
        offset = 0
        for page in pdf_pages[::-1]:
            combined_page.mergeTranslatedPage(page, 0, offset)
            offset += page.mediaBox.upperRight[1] + margin
        return em, f1, tree_depth, tree_size, combined_page
    else:
        return em, f1, tree_depth, tree_size, None


def evaluate_entailmentbank(
    results: List[Any],
    eval_intermediates: bool = True,
    output_pdf: Optional[str] = None,
) -> Any:
    """
    Evaluate predicted proof trees on EntailmentBank.

    The implementation has minor differences from EntailmentBank's official evaluation code.
    DO NOT use it for reporting results in papers. Use the official code instead.
    """
    measures = ["leaves", "steps", "proof"]
    if eval_intermediates:
        measures.append("intermediates")
    em = defaultdict(list)
    f1 = defaultdict(list)
    depth = []
    size = []
    bleurt = (
        datasets.load_metric("bleurt", "bleurt-large-512")
        if eval_intermediates
        else None
    )
    if output_pdf is not None:
        pdf_oup = PdfFileWriter()

    for ex in tqdm(results):
        em_i, f1_i, depth_i, size_i, page = evaluate_example(
            ex, bleurt, output_pdf is not None
        )

        depth.append(depth_i)
        size.append(size_i)
        for m in measures:
            em[m].append(em_i[m])
            f1[m].append(f1_i[m])

        if output_pdf:
            pdf_oup.addPage(page)

    if output_pdf is not None:
        pdf_oup.write(open(output_pdf, "wb"))

    depth_arr = np.array(depth)
    print("Performance by depth:")
    for d in np.unique(depth_arr):
        mask = depth_arr == d
        n = mask.sum()
        print(f"{n} trees have depth {d}")
        for m in measures:
            print(
                f"\t{m}: {(np.array(em[m]) * mask).sum() / n}\t{(np.array(f1[m]) * mask).sum() / n}"
            )

    size_arr = np.array(size)
    print("Performance by size:")
    for s in np.unique(size_arr):
        mask = size_arr == s
        n = mask.sum()
        print(f"{n} trees have size {s}")
        for m in measures:
            print(
                f"\t{m}: {(np.array(em[m]) * mask).sum() / n}\t{(np.array(f1[m]) * mask).sum() / n}"
            )

    return (
        {m: np.mean(values) for m, values in em.items()},
        {m: np.mean(values) for m, values in f1.items()},
    )


def split_steps(proof: str) -> Set[Any]:
    steps = set()
    for s in proof.split(";"):
        s = s.strip()
        if s == "":
            continue
        if s.count(" -> ") != 1:
            raise ValueError
        premises, conclusion = s.split(" -> ")
        steps.add((tuple(sorted(premises.split(" & "))), conclusion == "hypothesis"))
    return steps


def check_ruletaker_proof(proof_pred: str, proofs_gt: str) -> bool:
    """
    Check whether two RukeTaker proofs are equivalent.
    """
    try:
        proof_steps_pred = split_steps(proof_pred)
    except ValueError:
        return False
    proofs_steps_gt = [split_steps(_) for _ in proofs_gt]
    return proof_steps_pred in proofs_steps_gt


def process_results(results: List[Any]) -> Any:
    assert len(results) % 2 == 0
    n = len(results) // 2
    scores = []
    labels = []
    depths = []
    all_proofs = []

    for i in range(n):
        r_orig = results[2 * i]
        r_neg = results[2 * i + 1]
        assert r_orig["proof_gt"] == r_neg["proof_gt"]
        assert r_orig["depth"] == r_neg["depth"]
        assert r_orig["all_proofs"] == r_neg["all_proofs"]
        assert r_neg["hypothesis"] == f"i don't think {r_orig['hypothesis']}"
        scores.append([r_orig["score"], r_neg["score"]])
        if r_orig["answer"] == True:
            labels.append(0)
        elif r_orig["answer"] == False:
            labels.append(1)
        else:
            assert r_orig["answer"] == "Unknown"
            labels.append(2)
        depths.append(r_orig["depth"])
        all_proofs.append(r_orig["all_proofs"])

    return scores, labels, depths, all_proofs


def calculate_metrics(
    y_pred: Any, y: Any, depths: Any, all_proofs: Any, results: Any
) -> Any:
    """
    Calculate the answer accuracy and proof accuracy for RuleTaker.
    """
    answer_is_correct = defaultdict(list)
    n = len(y_pred)
    for i in range(n):
        d = depths[i]
        answer_is_correct[d].append(y[i] == y_pred[i])
    answer_accuracies = {k: np.mean(v) for k, v in answer_is_correct.items()}
    answer_accuracies["overall"] = accuracy_score(y, y_pred)

    proof_is_correct = defaultdict(list)
    for i in range(n):
        d = depths[i]
        if y_pred[i] == 2:
            proof_is_correct[d].append(y[i] == 2)
            continue
        if y_pred[i] == 0:
            proof_pred = results[2 * i]["proof_pred"]
        else:
            proof_pred = results[2 * i + 1]["proof_pred"]
        proof_is_correct[d].append(check_ruletaker_proof(proof_pred, all_proofs[i]))
    proof_accuracies = {k: np.mean(v) for k, v in proof_is_correct.items()}
    proof_accuracies["overall"] = np.mean(
        list(itertools.chain.from_iterable(proof_is_correct.values()))
    )

    return answer_accuracies, proof_accuracies


def evaluate_ruletaker(
    results_val: List[Any], results_test: Optional[List[Any]] = None
) -> Any:
    """
    Evaluate on RuleTaker.
    """
    scores_val, labels_val, depths_val, all_proofs_val = process_results(results_val)

    scaler = StandardScaler()
    X_val = scaler.fit_transform(np.array(scores_val))
    y_val = np.array(labels_val, dtype=np.int64)
    clf = LogisticRegression()
    clf.fit(X_val, y_val)
    y_val_pred = clf.predict(X_val)

    answer_accuracies_val, proof_accuracies_val = calculate_metrics(
        y_val_pred, y_val, depths_val, all_proofs_val, results_val
    )

    if results_test is None:
        return answer_accuracies_val, proof_accuracies_val

    scores_test, labels_test, depths_test, all_proofs_test = process_results(
        results_test
    )
    X_test = scaler.transform(np.array(scores_test))
    y_test = np.array(labels_test, dtype=np.int64)
    y_test_pred = clf.predict(X_test)
    answer_accuracies_test, proof_accuracies_test = calculate_metrics(
        y_test_pred, y_test, depths_test, all_proofs_test, results_test
    )
    return (
        answer_accuracies_val,
        proof_accuracies_val,
        answer_accuracies_test,
        proof_accuracies_test,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, choices=["entailmentbank", "ruletaker"])
    parser.add_argument("--path", type=str)
    parser.add_argument("--path-val", type=str)
    parser.add_argument("--path-test", type=str)
    parser.add_argument("--skip-intermediates", action="store_true")
    parser.add_argument("--output-pdf", type=str, help="Path for outputing the PDF")
    args = parser.parse_args()
    print(args)

    if args.output_pdf is not None:
        from ete3 import TextFace, TreeStyle, NodeStyle
        from PyPDF3 import PdfFileWriter, PdfFileReader
        from PyPDF3.pdf import PageObject

    if args.dataset == "entailmentbank":
        results = json.load(open(args.path))
        em, f1 = evaluate_entailmentbank(
            results, not args.skip_intermediates, args.output_pdf
        )
        print("Exact match: ", em)
        print("F1: ", f1)
    else:
        results_val = json.load(open(args.path_val))
        results_test = json.load(open(args.path_test))
        (
            answer_accuracies_val,
            proof_accuracies_val,
            answer_accuracies_test,
            proof_accuracies_test,
        ) = evaluate_ruletaker(results_val, results_test)
        print("Validation results:")
        print("Answer: ", answer_accuracies_val)
        print("Proof: ", proof_accuracies_val)
        print("Testing results:")
        print("Answer: ", answer_accuracies_test)
        print("Proof: ", proof_accuracies_test)
