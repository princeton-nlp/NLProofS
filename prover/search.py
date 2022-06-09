"""
Proof graph for proof search.
"""
from common import *
from prover.proof import ProofStep
import networkx as nx


class ProofGraph:
    def __init__(
        self, context: OrderedDict[str, str], hypothesis: str, eps: float = 1e-7
    ) -> None:
        self.graph = nx.DiGraph()
        self.graph.add_node("hypothesis", score=0.0, step_score=None, sent=hypothesis)
        self.graph.add_nodes_from(
            [(k, {"score": 1.0, "sent": v}) for k, v in context.items()]
        )
        self.assumptions = set(context.keys())
        self.intermediates = set()
        self.hypothesis = hypothesis
        self.sent2node = {v: k for k, v in context.items()}
        self.sent2node[hypothesis] = "hypothesis"
        assert nx.is_directed_acyclic_graph(self.graph)
        self.eps = eps

    def initialize(self, proof_steps: List[ProofStep], scores: List[float]):
        if len(proof_steps) == 0:
            return
        assert len(proof_steps) == len(scores)

        for step, score in zip(proof_steps, scores):
            self.expand(step, score)

    def sample_proof_tree(self, exclude: Set[str]) -> Optional[str]:
        """
        Sample a new partial proof tree not in `exclude`.
        """
        reachable_nodes = set()
        for node in self.assumptions:
            reachable_nodes.update(nx.descendants(self.graph, node))
        reachable_component = nx.subgraph(self.graph, reachable_nodes)

        hypothesis_score = self.graph.nodes["hypothesis"]["score"]

        for _ in range(10):
            partial_proof = ""
            nodes_included = set()
            for node in reversed(list(nx.topological_sort(reachable_component))):
                if (
                    node == "hypothesis"
                    or node in nodes_included
                    or self.graph.nodes[node]["score"] <= hypothesis_score + self.eps
                ):
                    continue

                subproof = self.extract_proof(node, nodes_proved=nodes_included) + " "
                if random.random() < 0.5:  # Include node.
                    partial_proof += subproof
                    nodes_included.add(node)
                    nodes_included.update(nx.ancestors(self.graph, node))
            prf = rename_ints(partial_proof.strip())
            if prf not in exclude:
                return prf

        return None

    def extract_proof(
        self, node: str, rename: bool = False, nodes_proved: Set[str] = set()
    ) -> str:
        ancestors = nx.ancestors(self.graph, node)
        ancestors.add(node)
        ancestors -= nodes_proved
        ancestors_subgraph = nx.subgraph(self.graph, ancestors)
        proof = ""

        for x in nx.topological_sort(ancestors_subgraph):
            if x not in self.assumptions:
                try:
                    proof += self.extract_proof_step(x)
                except ValueError:
                    return "INVALID_PROOF"

        if rename:
            proof = rename_ints(proof)
        return proof.strip()

    def extract_proof_step(self, node: str) -> str:
        assert node == "hypothesis" or node in self.intermediates
        predecessors = list(self.graph.predecessors(node))
        if len(predecessors) == 0:
            raise ValueError
        premises = " & ".join(predecessors)
        if node == "hypothesis":
            return f"{premises} -> hypothesis; "
        else:
            return f"{premises} -> {node}: {self.graph.nodes[node]['sent']}; "

    def expand(self, proof_step: ProofStep, score: float) -> bool:
        premises = []
        for ident, sent in zip(proof_step.premise_idents, proof_step.premise_sents):
            if re.fullmatch(r"sent\d+", ident):
                premises.append(ident)
            else:
                premises.append(self.sent2node[sent])
        dst_score = self.calculate_score(score, premises)
        if dst_score <= self.graph.nodes["hypothesis"]["score"] + self.eps:  # Prune.
            return False

        # Create a node for the conclusion if necessasry.
        if proof_step.conclusion_ident == "hypothesis":
            dst = "hypothesis"
            self.remove_inbound_edges(dst)
            self.graph.nodes["hypothesis"]["score"] = dst_score
            self.graph.nodes["hypothesis"]["step_score"] = score
        else:
            sent = proof_step.conclusion_sent
            dst = None

            if sent in self.sent2node:
                dst = self.sent2node[sent]

            if dst is not None:  # The node exists.
                if (
                    dst in self.assumptions
                    or dst_score <= self.graph.nodes[dst]["score"] + self.eps
                ):  # Prune.
                    return False
                self.graph.nodes[dst]["score"] = dst_score
                self.graph.nodes[dst]["step_score"] = score
                self.remove_inbound_edges(dst)
                self.propagate_score(dst)
            else:  # Need to create a new node.
                dst = f"int{1 + len(self.intermediates)}"
                self.graph.add_node(dst, score=dst_score, step_score=score, sent=sent)
                self.intermediates.add(dst)
                self.sent2node[sent] = dst

        # Add edges from the premises.
        for p in premises:
            if p.startswith("sent"):
                assert p in self.assumptions
            else:  # int
                assert p in self.intermediates
            self.graph.add_edge(p, dst)

        assert nx.is_directed_acyclic_graph(self.graph)
        return True

    def remove_inbound_edges(self, node):
        self.graph.remove_edges_from(list(self.graph.in_edges(node)))

    def propagate_score(self, node):
        for u in self.graph.successors(node):
            score_new = self.agg_op(
                self.graph.nodes[u]["step_score"],
                [self.graph.nodes[v]["score"] for v in self.graph.predecessors(u)],
            )
            if score_new > self.graph.nodes[u]["score"] + self.eps:
                self.graph.nodes[u]["score"] = score_new
                self.propagate_score(u)

    def calculate_score(self, step_score: float, premises: str) -> None:
        return self.agg_op(step_score, [self.graph.nodes[p]["score"] for p in premises])

    def agg_op(self, step_score: float, input_scores: List[float]) -> float:
        return min([step_score] + input_scores)

    def to_pydot(self):
        nodes_to_visualize = {
            node
            for node in self.graph.nodes
            if node not in self.assumptions or self.graph.out_degree(node) > 0
        }

        graph = nx.nx_pydot.to_pydot(nx.subgraph(self.graph, nodes_to_visualize))

        for node in graph.get_nodes():
            name = node.get_name()
            if not name.startswith("sent"):
                node.set_shape("box")
            nx_node = self.graph.nodes[name]
            label = f"{name}: {nx_node['sent']}"
            if name not in self.assumptions:
                label += f"\nstep_score: {nx_node['step_score']}"
            label += f"\nscore: {nx_node['score']}"
            node.set_label(label)

        return graph

    def visualize(self, fname: str) -> None:
        graph = self.to_pydot()

        # Highlight the ancestors of the hypothesis.
        ancestors = nx.ancestors(self.graph, "hypothesis")
        ancestors.add("hypothesis")
        for nx_node in ancestors:
            graph.get_node(nx_node)[0].set_color("red")

        graph.write_png(f"{fname}.png")

    def visualize_proof_tree(self, proof, fname):
        graph = self.to_pydot()

        for proof_step in proof.split(";"):
            proof_step = proof_step.strip()
            if proof_step == "":
                continue
            premises, conclusion = proof_step.split(" -> ")
            for p in premises.split(" & "):
                if re.fullmatch(r"sent\d+", p):
                    graph.get_node(p)[0].set_color("red")
            m = re.fullmatch(r"int\d+: (?P<sent>.+)", conclusion)
            assert m is not None
            name = self.sent2node[m["sent"]]
            graph.get_node(name)[0].set_color("red")

        graph.write_png(f"{fname}.png")
