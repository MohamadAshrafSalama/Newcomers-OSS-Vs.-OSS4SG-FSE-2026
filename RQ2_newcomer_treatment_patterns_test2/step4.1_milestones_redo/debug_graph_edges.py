#!/usr/bin/env python3
"""
Debug script to understand why all edges appear red.
Check how many edges are in graphs vs how many are covered by top-5 paths.
"""

import pandas as pd
from pathlib import Path

DIR = Path(__file__).parent

def analyze_graph_coverage(name):
    print(f"\n=== Analysis for {name} ===")
    
    # Load transitions (these are the graph edges after pruning)
    transitions_file = DIR / f"transitions_{name}.csv"
    if not transitions_file.exists():
        print(f"No transitions file for {name}")
        return
        
    transitions_df = pd.read_csv(transitions_file)
    total_edges = len(transitions_df)
    print(f"Total edges in graph: {total_edges}")
    
    # Load top flows
    flows_file = DIR / f"top_flows_{name}.csv"
    if not flows_file.exists():
        print(f"No flows file for {name}")
        return
        
    flows_df = pd.read_csv(flows_file)
    
    # Extract edges from top-5 paths
    top5_edges = set()
    for _, row in flows_df.iterrows():
        path = row['path'].split(' → ')
        edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
        top5_edges.update(edges)
    
    print(f"Unique edges in top-5 paths: {len(top5_edges)}")
    print(f"Coverage: {len(top5_edges)}/{total_edges} = {len(top5_edges)/total_edges*100:.1f}%")
    
    # Show which edges are NOT in top-5
    graph_edges = set(zip(transitions_df['src'], transitions_df['dst']))
    non_top5_edges = graph_edges - top5_edges
    if non_top5_edges:
        print(f"Edges NOT in top-5 paths: {len(non_top5_edges)}")
        for edge in sorted(non_top5_edges):
            print(f"  {edge[0]} → {edge[1]}")
    else:
        print("ALL edges are covered by top-5 paths! (That's why they're all red)")

if __name__ == "__main__":
    for name in ['overall', 'oss', 'oss4sg']:
        analyze_graph_coverage(name)
        
    # Check sequences counts for min15 vs all
    print(f"\n=== Sequence Count Analysis ===")
    
    for cohort in ['all', 'min15']:
        seq_file = DIR / f"sequences_{cohort}.csv"
        if seq_file.exists():
            df = pd.read_csv(seq_file)
            print(f"\nSequences {cohort}: {len(df)} total")
            type_counts = df['project_type'].value_counts()
            print(f"  OSS: {type_counts.get('OSS', 0)}")
            print(f"  OSS4SG: {type_counts.get('OSS4SG', 0)}")

            # FMPR presence by type
            df['has_FMPR'] = df['sequence'].astype(str).str.contains('FirstMergedPullRequest')
            for t in ['OSS', 'OSS4SG']:
                sub = df[df['project_type'] == t]
                if len(sub) == 0:
                    continue
                fmpr_n = int(sub['has_FMPR'].sum())
                print(f"  {t}: FMPR present in {fmpr_n}/{len(sub)} ({fmpr_n/len(sub)*100:.1f}%)")
            
            # Check if sequences are identical between min15 and all
            if cohort == 'min15':
                all_df = pd.read_csv(DIR / "sequences_all.csv")
                # Check overlap
                all_contributors = set(zip(all_df['project_name'], all_df['contributor_email']))
                min15_contributors = set(zip(df['project_name'], df['contributor_email']))
                overlap = len(min15_contributors & all_contributors)
                print(f"  Contributors in both all and min15: {overlap}/{len(min15_contributors)} = {overlap/len(min15_contributors)*100:.1f}%")
