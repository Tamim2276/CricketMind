"""The execution-trace figure: one row per superstep, one box per node run.

Kept out of the paper to fit six pages, but it is the clearest picture of a
correction loop actually happening, so it belongs in slides and the thesis.
"""
from ..agents import AGENTS_BY_ID

NODE_LABELS = {
    # perception and grounding
    "n1_video_input":        "N1 Video input",
    "n2_vision_encoder":     "N2 Vision encoder",
    "n3_text_encoder":       "N3 Text encoder",
    "n4_temporal_fusion":    "N4 Temporal fusion",
    "n5_classifier_head":    "N5 Classifier",
    "n6_shot_dna":           "N6 Shot DNA",
    # retrieval, analytics, routing
    "n7a_stats_retrieval":   "N7a Stats retrieval",
    "n7b_context_retrieval": "N7b Context retrieval",
    "n8_stats_engine":       "N8 Stats engine",
    "n9_whatif_simulator":   "N9 What-if simulator",
    "n10_supervisor_router": "N10 Router",
    # the twelve stakeholder agents
    "n11a_club_operations":          "N11a Franchises",
    "n11b_tactical_analysis":        "N11b Coaches",
    "n11c_personal_performance":     "N11c Players",
    "n11d_global_scouting":          "N11d Scouts",
    "n11e_fan_concierge":            "N11e Fans",
    "n11f_stadium_operations":       "N11f Stadiums",
    "n11g_sponsorship_intelligence": "N11g Sponsors",
    "n11h_content_production":       "N11h Broadcasters",
    "n11i_compliance":               "N11i Governing bodies",
    "n11j_talent_development":       "N11j Academies",
    "n11k_injury_management":        "N11k Medical teams",
    "n11l_player_representation":    "N11l Player agents",
    # gates and correction loops
    "retry":           "Retry",
    "degrade":         "Degrade",
    "faith_check":     "Faith Check",
    "regen_prep":      "Regenerate",
    "drop_unfaithful": "Drop unverified",
    # synthesis and output
    "n12_supervisor_synthesis": "N12 Synthesis",
    "n13_final_output":         "N13 Output",
}

CHECK_NODES = {"retry", "degrade", "faith_check", "regen_prep", "drop_unfaithful"}

ROLE_STYLE = {                        # group -> (fill colour, border colour)
    "pipeline": ("#f0efec", "#c3c2b7"),   # neutral grey
    "agent":    ("#cde2fb", "#2a78d6"),   # blue
    "check":    ("#fbded2", "#eb6834"),   # orange
}
ROLE_NAMES = {
    "pipeline": "Pipeline node",
    "agent":    "Stakeholder agent",
    "check":    "Gate or correction loop",
}

INK_TEXT = "#0b0b0b"     # box labels
INK_NOTE = "#52514e"     # notes beside the boxes
INK_MUTED = "#898781"    # step numbers


def node_role(node: str) -> str:
    """Which colour group a node belongs to in the trace figure."""
    if node in AGENTS_BY_ID:
        return "agent"
    if node in CHECK_NODES:
        return "check"
    return "pipeline"


def draw_timeline(timeline: list, notes: dict = None, hatched: set = None,
                  save_to=None):
    """Draw a recorded run: one row per step, one box per node.

    timeline  the (step, node) list from record_run
    notes     {step: text}, a short note written beside that step's box
    hatched   {(step, node)}, boxes to hatch, e.g. a rejected agent run
    save_to   file path to save to; a .pdf path gives a vector figure
    Returns the matplotlib figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Patch

    notes = notes or {}
    hatched = hatched or set()

    steps = sorted({step for step, node in timeline})
    per_step = {step: [node for s, node in timeline if s == step] for step in steps}
    n_cols = max(len(nodes) for nodes in per_step.values())

    # page geometry, in inches
    WIDTH = 3.35                   # one column of a two-column paper
    LEFT = 0.34                    # room for the step numbers
    TOP = 0.24                     # room for the "step" heading
    ROW = 0.21                     # distance from one row to the next
    BOX_H = 0.16
    GAP = 0.06                     # space between boxes in the same row
    LEGEND = 0.42                  # room for the legend underneath
    col_w = (WIDTH - LEFT - 0.04 - GAP * (n_cols - 1)) / n_cols
    height = TOP + ROW * len(steps) + LEGEND

    with plt.rc_context({"pdf.fonttype": 42, "font.size": 6.5,
                         "hatch.linewidth": 0.6}):
        fig = plt.figure(figsize=(WIDTH, height))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, WIDTH)
        ax.set_ylim(height, 0)     # y grows downwards, like reading a page
        ax.axis("off")

        ax.text(LEFT - 0.12, TOP - 0.08, "step", ha="right", va="center",
                color=INK_MUTED)

        for row, step in enumerate(steps):
            y = TOP + row * ROW
            ax.text(LEFT - 0.12, y + BOX_H / 2, str(step), ha="right",
                    va="center", color=INK_MUTED)

            for col, node in enumerate(per_step[step]):
                x = LEFT + col * (col_w + GAP)
                fill, edge = ROLE_STYLE[node_role(node)]
                is_hatched = (step, node) in hatched
                ax.add_patch(FancyBboxPatch(
                    (x, y), col_w, BOX_H,
                    boxstyle="round,pad=0,rounding_size=0.03",
                    facecolor=fill, edgecolor=edge, linewidth=0.8,
                    hatch="//////" if is_hatched else None))
                # on a hatched box, give the label a plain patch behind it
                label_bg = (dict(facecolor=fill, edgecolor="none", pad=0.8)
                            if is_hatched else None)
                ax.text(x + 0.06, y + BOX_H / 2, NODE_LABELS[node], va="center",
                        color=INK_TEXT, bbox=label_bg)

            if step in notes:
                assert len(per_step[step]) < n_cols, \
                    f"step {step} has no free space for a note"
                x = LEFT + len(per_step[step]) * (col_w + GAP)
                ax.text(x + 0.02, y + BOX_H / 2, notes[step], va="center",
                        color=INK_NOTE, style="italic")

        # legend: one entry per colour group, plus the hatching if it was used
        handles = [Patch(facecolor=ROLE_STYLE[r][0], edgecolor=ROLE_STYLE[r][1],
                         linewidth=0.8, label=ROLE_NAMES[r]) for r in ROLE_STYLE]
        if hatched:
            fill, edge = ROLE_STYLE["agent"]
            handles.append(Patch(facecolor=fill, edgecolor=edge, linewidth=0.8,
                                 hatch="//////", label="Rejected agent run"))
        ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0),
                  ncol=2, frameon=False, fontsize=6, handlelength=1.6,
                  handleheight=1.0, columnspacing=1.2)

        if save_to is not None:
            fig.savefig(save_to)
    return fig
