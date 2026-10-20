"""The gates: the seven checks in order, cheapest first, and two older gates."""

from cricketmind_v3.domain.faith.c1_registry import check_c1
from cricketmind_v3.domain.faith.c2_values import check_c2
from cricketmind_v3.domain.faith.c3_evidence import check_c3
from cricketmind_v3.domain.faith.c4_wording import check_c4
from cricketmind_v3.domain.faith.c5_numbers import check_c5
from cricketmind_v3.domain.faith.c6_names import check_c6
from cricketmind_v3.domain.faith.c7_tone import check_c7
from cricketmind_v3.domain.faith.v2_mode import check_v2

# the Faith Check of the v4 design
ALL_CHECKS = (check_c1, check_c2, check_c3, check_c4, check_c5, check_c6, check_c7)

# the v3 design document's gate: no tone check
V3_DESIGN_CHECKS = (check_c1, check_c2, check_c3, check_c4, check_c5, check_c6)

# the ICCIT (v2) gate: cited values only
V2_MODE = (check_v2,)
