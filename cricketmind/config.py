"""Constants and the run Settings object.

Everything that used to be a module-level knob the notebook reassigned
(`STUB_CONFIDENCE`, `TRACE`, `HALLUCINATE_ONCE`, `HALLUCINATE_ALWAYS`,
`ROUTER_MODE`, `OLLAMA_URL`) now lives on `Settings`, which travels inside the
graph state. Cell execution order can therefore no longer change a result.

The true constants below are different: they are properties of the
architecture, not of a run, and nothing reassigns them.
"""
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping, Sequence, Union

# --- the architecture's fixed parameters ------------------------------------
SHOT_CLASSES = [
    "Cover Drive", "Defensive", "Down The Wicket", "Flick", "Hook",
    "Late Cut", "Lofted Legside", "Lofted Offside", "Pull",
    "Reverse Sweep", "Scoop", "Square Cut", "Straight Drive",
    "Sweep", "Upper Cut",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(SHOT_CLASSES)}
NUM_CLASSES = len(SHOT_CLASSES)
NUM_FRAMES = 15
FRAME_SIZE = 224
FEATURE_DIM = 1280                 # EfficientNetV2-S output width

CONF_THRESHOLD = 0.70              # N5 -> Conf. Check gate
MAX_RETRIES = 2                    # bound on the retry loop
MAX_REGENS = 2                     # bound on the regenerate loop
MAX_ACTIVE_AGENTS = 4              # bound on how many agents answer

SIM_TRIALS = 20_000                # N9 Monte Carlo runs
SIM_DELIVERIES = 6                 # "what if the bowler sends down six of these?"

SEED = 42

# Woken when nothing else scores -- the general-purpose pair that can say
# something useful about any delivery.
DEFAULT_AGENTS = ["n11b_tactical_analysis", "n11c_personal_performance"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The running example, used by the execution profile and the routing tables.
QUERY = ("Compare Kohli's cover drive vs his pull shot against fast bowlers "
         "in the powerplay")
CLIP = "clips/kohli_01.mp4"


@dataclass(frozen=True)
class Settings:
    """Everything about a run that is not the query itself.

    Passed into `new_state` and carried in the graph state, so two runs with
    different settings cannot interfere with each other and running a cell
    twice cannot leave a knob changed. This replaces the module-level globals
    the notebook used to reassign.

    label       what the stub classifier predicts
    confidence  what it scores -- one number for every attempt, or a sequence
                with one number per attempt, e.g. (0.40, 0.92)
    lie_once    agent ids that fabricate a cited value on their first attempt
                and correct themselves afterwards
    lie_always  agent ids that never correct themselves, which exercises the
                regeneration budget
    router_mode "keyword" (the rule-based scorer) or "llm" (a local model)
    trace       print each node's own log line
    template_overrides  {agent id: replacement sentence template}, used by the
                Faith Check boundary experiment to change what an agent *says*
                without changing what it *cites*
    """
    label: str = "Cover Drive"
    confidence: Union[float, Sequence[float], str] = 0.92
    lie_once: frozenset = frozenset()
    lie_always: frozenset = frozenset()
    router_mode: str = "keyword"
    trace: bool = False
    ollama_url: str = "http://localhost:11434"
    router_model: str = "qwen2.5:7b"
    template_overrides: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self):
        # Normalise so equality and hashing behave, and so callers may pass a
        # plain list or set as the notebook always did.
        if isinstance(self.confidence, list):
            object.__setattr__(self, "confidence", tuple(self.confidence))
        if not isinstance(self.lie_once, frozenset):
            object.__setattr__(self, "lie_once", frozenset(self.lie_once))
        if not isinstance(self.lie_always, frozenset):
            object.__setattr__(self, "lie_always", frozenset(self.lie_always))
        assert self.router_mode in ("keyword", "llm"), \
            f"unknown router_mode {self.router_mode!r}"

    def with_(self, **changes) -> "Settings":
        """A copy with some fields changed: `settings.with_(trace=True)`."""
        return replace(self, **changes)


DEFAULT_SETTINGS = Settings()
