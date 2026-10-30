import numpy as np
import pandas as pd

from artguide.models.baselines import aspect_ratio_features


def test_aspect_ratio_is_symmetric_and_follows_the_ids_order():
    metadata = pd.DataFrame({"id": ["a", "b", "c"], "orig_width": [100, 200, 300], "orig_height": [200, 100, 300]})
    x = aspect_ratio_features(["c", "a", "b"], metadata)
    assert x.shape == (3, 1)
    np.testing.assert_allclose(x[:, 0], [0.0, np.log(2), -np.log(2)])
