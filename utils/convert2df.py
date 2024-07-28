def convert_df(df):
    """
    Converts a DataFrame to a CSV format and encodes it in UTF-8.
    Caches the conversion to prevent computation on every rerun.
    """
    return df.to_csv().encode('utf-8')
