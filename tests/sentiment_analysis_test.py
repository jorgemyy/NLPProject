import pandas as pd
from package.models.model_factory import ModelFactory

def test_get_data():
    '''test if the data is successfully loaded from kaggles with the correct labels'''
    model_factory = ModelFactory()
    sem_model = model_factory.createSemModel()
    full_df = sem_model.get_data()

    assert type(full_df) == pd.DataFrame
    assert not full_df.isna().values.any()

    assert all(x <= 2 for x in set(full_df["label"]))