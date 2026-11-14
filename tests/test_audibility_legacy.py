import numpy as np
import pandas as pd

from ras.audibility import audibility_sir


def test_audibility_sir():
    family = pd.DataFrame({"Audible":[1,0,1,1]})
    # observed 3 / expected 4*0.5 = 2 => 1.5
    assert audibility_sir(family, 0.5) == 1.5
