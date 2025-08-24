import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi

def create_timing_radar_plot():
    """Create a radar plot comparing OSS vs OSS4SG milestone timing (median weeks)"""

    # Load milestone data
    results_path = "RQ2_newcomer_treatment_patterns_test2/step4_milestones/results/milestone_summary_statistics.csv"
    df = pd.read_csv(results_path)

    # Filter for the 5 working milestones (exclude Trusted Reviewer and Community Helper)
    working_milestones = [
        'First Accepted',
        'Sustained Participation',
        'Returning Contributor',
        'Cross Boundary',
        'Failure Recovery'
    ]

    milestone_labels = [
        'First Accepted',
        'Sustained\nParticipation',
        'Returning\nContributor',
        'Cross\nBoundary',
        'Failure\nRecovery'
    ]

    # Filter the dataframe to only working milestones
    df_filtered = df[df['Milestone'].isin(working_milestones)]

    # Get median timing (using the clean versions that handle 0-week corrections)
    oss_timing = df_filtered['OSS_Median_Clean'].tolist()
    oss4sg_timing = df_filtered['OSS4SG_Median_Clean'].tolist()

    # Create radar plot
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

    # Number of variables
    categories = milestone_labels
    N = len(categories)

    # Calculate angles for each axis
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Close the plot

    # Add OSS data
    oss_values = oss_timing + oss_timing[:1]  # Close the plot
    ax.plot(angles, oss_values, 'o-', linewidth=2, label='OSS', color='blue')
    ax.fill(angles, oss_values, alpha=0.25, color='blue')

    # Add OSS4SG data
    oss4sg_values = oss4sg_timing + oss4sg_timing[:1]  # Close the plot
    ax.plot(angles, oss4sg_values, 'o-', linewidth=2, label='OSS4SG', color='orange')
    ax.fill(angles, oss4sg_values, alpha=0.25, color='orange')

    # Add milestone labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=12, fontweight='bold')

    # Set y-axis limits to show timing clearly (0-16 weeks based on data)
    max_timing = max(max(oss_timing), max(oss4sg_timing))
    ax.set_ylim(0, max_timing + 2)
    ax.set_yticks(range(0, int(max_timing) + 3, 2))
    ax.set_yticklabels([f'{i} weeks' for i in range(0, int(max_timing) + 3, 2)], size=10)

    # Add title and legend
    ax.set_title('OSS vs OSS4SG Milestone Achievement Timing\n(Median Weeks to Achieve Each Milestone)',
                size=14, fontweight='bold', pad=20)

    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    # Add grid
    ax.grid(True, alpha=0.3)

    # Save the plot
    plt.tight_layout()
    plt.savefig('RQ2_newcomer_treatment_patterns_test2/step4_milestones/visualizations/milestone_timing_radar_plot.png',
                dpi=300, bbox_inches='tight')
    print("Saved: RQ2_newcomer_treatment_patterns_test2/step4_milestones/visualizations/milestone_timing_radar_plot.png")

    # Show the plot
    plt.show()

if __name__ == "__main__":
    create_timing_radar_plot()
