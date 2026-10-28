import numpy as np
import pandas as pd
import pytest

from quakes import config


@pytest.fixture
def toy_df():
    rng = np.random.default_rng(0)
    n = 300
    d = {c: rng.normal(size=n) for c in config.NUMERIC}
    for c in config.NOMINAL:
        d[c] = rng.choice(["a", "b", "c"], n)
    d["mag"] = rng.uniform(0, 7, n)
    d["big_quake"] = (rng.random(n) < 0.3).astype(int)
    return pd.DataFrame(d)
