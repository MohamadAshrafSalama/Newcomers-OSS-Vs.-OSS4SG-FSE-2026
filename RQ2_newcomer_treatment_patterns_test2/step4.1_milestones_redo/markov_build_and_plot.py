#!/usr/bin/env python3
"""
Build Markov graphs from sequences_all.csv and plot Overall/OSS/OSS4SG
for the All cohort. Prune edges with p <= 0.05. Extract top-5 flows using
negative log probabilities and shortest_simple_paths (Yen-style).

Outputs (under figures/):
- overall_markov.png, oss_markov.png, oss4sg_markov.png
- top_flows_overall.csv, top_flows_oss.csv, top_flows_oss4sg.csv
- transitions_overall.csv, transitions_oss.csv, transitions_oss4sg.csv
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

BASE = Path("/Users/mohamadashraf/Desktop/Projects/Newcomers OSS Vs. OSS4SG FSE 2026")
DIR = BASE / "RQ2_newcomer_treatment_patterns_test2/step4.1_milestones_redo"
SEQ_FILE_ALL = DIR / "sequences_all.csv"
SEQ_FILE_MIN15 = DIR / "sequences_min15.csv"
FIG_DIR = DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def build_transitions(seqs: List[List[str]]) -> Dict[Tuple[str, str], int]:
    counts: Dict[Tuple[str, str], int] = {}
    for seq in seqs:
        # ensure START...END exists
        if not seq or seq[0] != 'START' or seq[-1] != 'END':
            continue
        for a, b in zip(seq[:-1], seq[1:]):
            counts[(a, b)] = counts.get((a, b), 0) + 1
    return counts


def to_probabilities(counts: Dict[Tuple[str, str], int]) -> Dict[Tuple[str, str], float]:
    out: Dict[Tuple[str, str], float] = {}
    by_src: Dict[str, int] = {}
    for (a, b), c in counts.items():
        by_src[a] = by_src.get(a, 0) + c
    for (a, b), c in counts.items():
        denom = by_src.get(a, 0)
        if denom > 0:
            out[(a, b)] = c / denom
    return out


def prune_edges(probs: Dict[Tuple[str, str], float], threshold: float = 0.05) -> Dict[Tuple[str, str], float]:
    return {k: p for k, p in probs.items() if p > threshold}


def build_graph(probs: Dict[Tuple[str, str], float]) -> nx.DiGraph:
    G = nx.DiGraph()
    for (a, b), p in probs.items():
        G.add_edge(a, b, p=p, w=-math.log(max(p, 1e-12)))
    return G


def top_k_paths(G: nx.DiGraph, source: str, target: str, k: int = 5) -> List[Tuple[List[str], float]]:
    paths = []
    try:
        for path in nx.shortest_simple_paths(G, source, target, weight='w'):
            # compute path probability as product of edge p
            prob = 1.0
            for a, b in zip(path[:-1], path[1:]):
                prob *= G[a][b]['p']
            paths.append((path, prob))
            if len(paths) >= k:
                break
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        pass
    return paths


def plot_graph(G: nx.DiGraph, title: str, outfile: Path) -> None:
    if G.number_of_edges() == 0:
        plt.figure(figsize=(6, 4))
        plt.title(title)
        plt.text(0.5, 0.5, 'No edges after pruning', ha='center')
        plt.savefig(outfile, dpi=300, bbox_inches='tight')
        plt.close()
        return
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 7))
    # Node colors
    colors = []
    for n in G.nodes:
        if n == 'START' or n == 'END':
            colors.append('#e74c3c')
        else:
            colors.append('#3498db')
    # Edge widths by p
    widths = [max(1.0, G[u][v]['p'] * 10.0) for u, v in G.edges()]
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=900, alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=9)
    nx.draw_networkx_edges(G, pos, width=widths, arrows=True, alpha=0.7)
    edge_labels = {(u, v): f"{G[u][v]['p']:.2f}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()


def run_for_subset(df: pd.DataFrame, name: str) -> None:
    # Remove sequences with no milestones (START->END only)
    df = df[df['sequence'].astype(str) != 'START,END'].copy()
    if len(df) == 0:
        # Write empty artifacts for consistency
        pd.DataFrame().to_csv(DIR / f"transitions_{name}.csv", index=False)
        pd.DataFrame().to_csv(DIR / f"top_flows_{name}.csv", index=False)
        plt.figure(figsize=(6, 4)); plt.title(f"Markov Graph ({name})\nNo sequences after filtering"); plt.axis('off')
        plt.savefig(FIG_DIR / f"{name}_markov.png", dpi=300, bbox_inches='tight'); plt.close()
        return

    seqs = [s.split(',') for s in df['sequence'].astype(str).tolist()]
    counts = build_transitions(seqs)
    probs = to_probabilities(counts)
    probs = prune_edges(probs, threshold=0.05)
    # Save transitions
    rows = [{'src': a, 'dst': b, 'p': p} for (a, b), p in probs.items()]
    pd.DataFrame(rows).to_csv(DIR / f"transitions_{name}.csv", index=False)
    # Graph
    G = build_graph(probs)
    # Compute only the single best path (highest probability)
    paths = top_k_paths(G, 'START', 'END', k=1)
    # Base plot (grey edges) with clearer style
    if G.number_of_edges() > 0:
        # Start from spring layout for spread, then pin START far left and END far right
        pos = nx.spring_layout(G, seed=42, k=1.2)
        # Find extents
        xs = [p[0] for p in pos.values()]
        minx, maxx = min(xs), max(xs)
        # Place START/END with margin
        pos['START'] = (minx - 1.5, 0.0)
        pos['END'] = (maxx + 1.5, 0.0)

        plt.figure(figsize=(14, 8))
        # Nodes - use circles instead of squares for better readability
        start_end = [n for n in G.nodes if n in ('START','END')]
        others = [n for n in G.nodes if n not in ('START','END')]
        nx.draw_networkx_nodes(G, pos, nodelist=start_end, node_color='#e74c3c', node_size=1800, alpha=0.95, node_shape='o')
        nx.draw_networkx_nodes(G, pos, nodelist=others, node_color='#5dade2', node_size=1600, alpha=0.95, node_shape='o')
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', 
                                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.3'))
        
        # Identify edges that are part of the best path only
        best_edges = set()
        if paths:
            best_path, _p = paths[0]
            best_edges = set(zip(best_path[:-1], best_path[1:]))
        
        # Draw grey edges for non-top-5 transitions
        grey_edges = [(u, v) for u, v in G.edges() if (u, v) not in best_edges]
        if grey_edges:
            grey_widths = [max(1.0, G[u][v]['p'] * 6.0) for u, v in grey_edges]
            nx.draw_networkx_edges(
                G, pos, edgelist=grey_edges, width=grey_widths, arrows=True, 
                alpha=0.4, edge_color='#bdc3c7', arrowsize=25, arrowstyle='->', 
                connectionstyle='arc3,rad=0.12', min_target_margin=15
            )
        
        # Draw red edges for the best path only
        if best_edges:
            red_widths = [max(3.0, G[u][v]['p'] * 12.0) for u, v in best_edges]
            nx.draw_networkx_edges(
                G, pos, edgelist=list(best_edges), width=red_widths, arrows=True, 
                edge_color='#e74c3c', arrowsize=30, arrowstyle='->', 
                connectionstyle='arc3,rad=0.12', min_target_margin=15
            )
        plt.title(f"Markov Graph ({name})")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"{name}_markov.png", dpi=300, bbox_inches='tight')
        plt.close()
    pd.DataFrame([{ 'rank': i+1, 'path': ' → '.join(p), 'probability': prob } for i, (p, prob) in enumerate(paths)]).to_csv(
        DIR / f"top_flows_{name}.csv", index=False
    )

    # PyVis interactive HTML using provided style (box nodes, barnes_hut + physics off, best path highlighted)
    try:
        from pyvis.network import Network
        net = Network(height="900px", width="100%", directed=True)
        net.barnes_hut(gravity=-2000, central_gravity=0.2, spring_length=200, spring_strength=0.01)

        # Build highlight edge set from the best path
        highlight_edges = set()
        if paths:
            best_path, _prob = paths[0]
            highlight_edges.update(zip(best_path[:-1], best_path[1:]))

        # Add nodes
        for node in G.nodes:
            net.add_node(
                node,
                label=node,
                color="#97C2FC" if node not in {"START", "END"} else "#FB7E81",
                shape="box",
                font={"size": 24, "color": "black"},
                margin=15,
            )

        # Add edges
        for u, v, data in G.edges(data=True):
            prob = float(data.get('p', data.get('prob', 0.0)))
            net.add_edge(
                u,
                v,
                title=f"{prob:.2f}",
                value=prob,
                color="red" if (u, v) in highlight_edges else "gray",
                width=5 if (u, v) in highlight_edges else 2,
                font={"size": 18, "color": "gray"},
            )

        net.set_options("""
        var options = {
            "configure": {"enabled": true},
            "nodes": {"font": {"size": 28}},
            "edges": {"smooth": {"type": "continuous"}},
            "physics": {"enabled": false}
        }
        """)
        net.write_html(str(FIG_DIR / f"{name}_markov.html"))
    except Exception:
        pass


def process_sequences_file(path: Path, suffix: str) -> None:
    if not path.exists():
        return
    df = pd.read_csv(path)
    # Overall
    run_for_subset(df, f"overall{suffix}")
    # OSS
    run_for_subset(df[df['project_type'] == 'OSS'], f"oss{suffix}")
    # OSS4SG
    run_for_subset(df[df['project_type'] == 'OSS4SG'], f"oss4sg{suffix}")


def main() -> None:
    process_sequences_file(SEQ_FILE_ALL, '')
    process_sequences_file(SEQ_FILE_MIN15, '_min15')
    print("Done: Markov graphs and top flows (PNG + HTML) written.")


if __name__ == '__main__':
    main()


