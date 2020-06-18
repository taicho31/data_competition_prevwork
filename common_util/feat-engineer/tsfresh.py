from tsfresh import extract_features, extract_relevant_features
from tsfresh.feature_extraction import settings
# https://tsfresh.readthedocs.io/en/latest/api/tsfresh.feature_extraction.html#module-tsfresh.feature_extraction.feature_calculators
fc_parameters = {
    "large_standard_deviation": [{ "r": 0.1}],   
    "ar_coefficient": [{"coeff": 3, "k": 10}], 
    "cid_ce" :[{"normalize": True}],
    "autocorrelation": [{"lag": 30}],
    "fft_aggregated" : [{"aggtype": "kurtosis"}],
    "fft_coefficient": [{"coeff": 3, "attr": "real"}],
}
ts_df = extract_features(spec_df, column_id="spectrum_filename", column_sort="wavelength", n_jobs=8, 
                         default_fc_parameters=fc_parameters) #settings.EfficientFCParameters())

ts_df = ts_df.reset_index()
ts_df = ts_df.rename(columns={"id":"spectrum_filename"})
