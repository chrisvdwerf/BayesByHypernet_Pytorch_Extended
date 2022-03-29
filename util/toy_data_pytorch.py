import pandas as pd


def train_and_predict(mode: str, data_x, data_y) -> (pd.DataFrame, dict):
    """
    mode - string of the mode
    data_x - datapoints x for training
    data_y - datapoints y for training
    linspace - prediction samples x
    --- output ---
    dataframe - all predictions in linspace
    weight_dict - dictionary with one key for the weights of the model used defined by mode
    """

    raise NotImplementedError("")
