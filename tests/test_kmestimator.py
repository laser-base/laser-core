"""
Unit tests for the KaplanMeierEstimator class from the laser.core.demographics module.

This module contains a suite of unit tests to validate the functionality and robustness of the
KaplanMeierEstimator class. The tests cover various initialization scenarios, prediction limits,
and statistical validation using the Kolmogorov-Smirnov test.

Classes:
    TestKaplanMeierEstimator: A unittest.TestCase subclass containing all the test methods.

Functions:
    _compare_estimators(etest, eexpected, max_year=100): Compare two estimators using the Kolmogorov-Smirnov test.

Test Methods in TestKaplanMeierEstimator:
    setUpClass(cls): Set up the test class with necessary data and estimator instance.
    test_init_with_numpy_array(self): Test initialization with a NumPy array.
    test_init_with_python_list(self): Test initialization with a Python list.
    test_init_with_string_filename(self): Test initialization with a string filename.
    test_init_with_missing_file(self): Test initialization with a missing file.
    test_init_with_invalid_source(self): Test initialization with an invalid source type.
    test_predict_year_of_death_limits_with_default(self): Test prediction limits for year of death with default settings.
    test_predict_year_of_death_limits_with_maximum(self): Test prediction limits for year of death with a specified maximum.
    test_predict_year_of_death_kstest(self): Test prediction for year of death using the Kolmogorov-Smirnov test.
    test_predict_age_at_death_limits_default(self): Test prediction limits for age at death with default settings.
    test_predict_age_at_death_limits_with_maximum(self): Test prediction limits for age at death with a specified maximum.
    test_predict_age_at_death_kstest(self): Test prediction for age at death using the Kolmogorov-Smirnov test.
    test_predict_year_of_death_with_types(self): Test prediction for year of death with various data types.
    test_predict_age_at_death_with_types(self): Test prediction for age at death with various data types.
"""

import re
import unittest
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import kstest

from laser.core.demographics import AliasedDistribution
from laser.core.demographics import KaplanMeierEstimator
from laser.core.demographics import load_pyramid_csv


class TestKaplanMeierEstimator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.filepath = Path(__file__).parent / "data" / "us-life-tables-nvs-2003.csv"
        cls.estimator = KaplanMeierEstimator(cls.filepath)
        cls.cumulative_deaths = cls.estimator.cumulative_deaths  # Note: Does not include a leading 0.

        return

    def test_init_with_numpy_array(self):
        estimator = KaplanMeierEstimator(self.cumulative_deaths)  # Should _definitely_ match...

        test = _compare_estimators(estimator, self.estimator)
        assert test.pvalue > 0.99, f"Estimator from NumPy array failed KS test ({test.pvalue=})"

        return

    def test_init_with_python_list(self):
        estimator = KaplanMeierEstimator(list(self.cumulative_deaths))  # Should _definitely_ match...

        test = _compare_estimators(estimator, self.estimator)
        assert test.pvalue > 0.99, f"Estimator from NumPy array failed KS test ({test.pvalue=})"

        return

    def test_init_with_string_filename(self):
        estimator = KaplanMeierEstimator(str(self.filepath))

        test = _compare_estimators(estimator, self.estimator)
        assert test.pvalue > 0.99, f"Estimator from NumPy array failed KS test ({test.pvalue=})"

        return

    def test_init_with_missing_file(self):
        missing = self.filepath.parent / "definitely_missing_file.csv"
        # Windows path uses "\" which needs to be escaped.
        with pytest.raises(FileNotFoundError, match=re.escape(f"File not found: {missing}")):
            KaplanMeierEstimator(missing)

        return

    def test_init_with_invalid_source(self):
        with pytest.raises(TypeError, match="Invalid source type: <class 'dict'>"):
            KaplanMeierEstimator({"A": 1, "B": 2, "C": 3})

        return

    # Implicitly tested with all the other functions...
    # def test_init_with_path(self):
    #     estimator = KaplanMeierEstimator(self.filepath)

    #     test = _compare_estimators(estimator, self.estimator)
    #     assert test.pvalue > 0.99, f"Estimator from NumPy array failed KS test ({test.pvalue=})"

    #     return

    def test_predict_year_of_death_limits_with_default(self):
        ages_years = np.random.randint(100, size=1024, dtype=np.int32)
        result = self.estimator.predict_year_of_death(ages_years)
        assert all(result >= ages_years), f"pysod should be >= current age (in years) {result=}, {ages_years=}"
        assert all(result <= 100), f"pysod should be <= 100 ({result.max()=})"

        return

    def test_predict_year_of_death_limits_with_maximum(self):
        max_year = 80
        ages_years = np.random.randint(max_year, size=1024, dtype=np.int32)
        result = self.estimator.predict_year_of_death(ages_years, max_year)
        assert all(result >= ages_years), f"pysod should be >= current age (in years) {result=}, {ages_years=}"
        assert all(result <= max_year), f"pysod should be <= max year ({max_year=}, {result.max()=})"

        return

    def test_predict_year_of_death_kstest(self):
        ages_years = np.zeros(100_000, dtype=np.int32)
        max_year = 100
        predictions = self.estimator.predict_year_of_death(ages_years, max_year)
        counts = np.zeros(max_year + 1, dtype=np.int32)
        np.add.at(counts, predictions, 1)
        f_of_x = counts.cumsum()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="ks_2samp: Exact calculation unsuccessful.*",
                category=RuntimeWarning,
            )
            test = kstest(f_of_x, self.cumulative_deaths)
        assert test.pvalue > 0.80, f"Kolmogorov-Smirnov test failed {test.pvalue=}"

        return

    def test_predict_age_at_death_limits_default(self):
        ages_years = np.random.randint(100, size=1024, dtype=np.int32)
        ages_days = ages_years * 365 + np.random.randint(365, size=ages_years.shape[0], dtype=ages_years.dtype)
        result = self.estimator.predict_age_at_death(ages_days)
        assert all(result >= ages_days), f"predicted age at death should be >= current age (in days) {result=}, {ages_days=}"
        assert all(result < ((100 + 1) * 365)), f"predicted age at death should be < (100 + 1) (in days) ({result.max()=})"

        return

    def test_predict_age_at_death_limits_with_maximum(self):
        max_year = 80
        ages_years = np.random.randint(max_year, size=1024, dtype=np.int32)
        ages_days = ages_years * 365 + np.random.randint(365, size=ages_years.shape[0], dtype=ages_years.dtype)

        result = self.estimator.predict_age_at_death(ages_days, max_year)
        assert all(result >= ages_days), f"predicted age at death should be >= current age (in days) {result=}, {ages_days=}"
        assert all(
            result < ((max_year + 1) * 365)
        ), f"predicted age at death should be < (max_year + 1) (in days) ({max_year=}, {result.max()=})"

        return

    def test_predict_age_at_death_kstest(self):
        # pvalues = []    # debugging
        # We could just do a uniform draw from 0-100 years, but that would be boring.
        pyramid = load_pyramid_csv(Path(__file__).parent / "data" / "us-pyramid-2023.csv")
        both = pyramid[:, 2] + pyramid[:, 3]  # males and females combined
        ad = AliasedDistribution(both)
        ages_years = ad.sample(10_000_000)
        ages_days = ages_years * 365 + np.random.randint(365, size=ages_years.shape[0], dtype=ages_years.dtype)

        cfailures = 0
        ntrials = 4
        for itrial in range(ntrials):
            predictions = self.estimator.predict_age_at_death(ages_days, 100)
            predicted_years = np.floor_divide(predictions, 365, dtype=np.int32)  # convert age (days) at death to age (years) at death
            assert predicted_years.max() <= 100, f"predicted years of death should be <= 100 ({predicted_years.max()=})"
            # for each starting age, check against the (remaining) survival curve
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="ks_2samp: Exact calculation unsuccessful.*",
                    category=RuntimeWarning,
                )
                expected_deaths = np.concatenate(([self.cumulative_deaths[0]], np.diff(self.cumulative_deaths)))
                failed = False
                for age in range(ages_years.max() + 1):
                    individuals = np.nonzero(ages_years == age)[0]  # indices of individuals who are age `age` or older
                    counts = np.zeros(len(self.cumulative_deaths), dtype=np.int32)  # zeroed array
                    np.add.at(counts, predicted_years[individuals], 1)  # histogram of deaths at age i
                    f_of_x = counts[age:]
                    g_of_x = expected_deaths[age:]  # survival curve for years >= age
                    factor = f_of_x.sum() / g_of_x.sum()  # comparison/normalization factor
                    g_of_x = (factor * g_of_x).astype(g_of_x.dtype)  # use same total number of deaths
                    test = kstest(f_of_x, g_of_x)
                    # pvalues.append(test.pvalue) # debugging
                    # assert test.pvalue >= 0.90, f"Kolmogorov-Smirnov test failed for {age=} ({test.pvalue=})"
                    if test.pvalue < 0.90:
                        failed = True
                        print(f"Kolmogorov-Smirnov test failed for {age=}, {test.pvalue=} ({itrial=})")
                if failed:
                    cfailures += 1
        assert cfailures < ntrials, f"Kolmogorov-Smirnov test failed too many times ({cfailures} out of {ntrials})."

        # print(f"minimum p-value: {min(pvalues)}")   # debugging

        return

    def test_predict_year_of_death_with_types(self):
        for type in [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64, np.float32, np.float64]:
            max_year = 100
            ages_years = np.random.randint(max_year + 1, size=1024).astype(type)
            result = self.estimator.predict_year_of_death(ages_years, max_year=max_year)
            assert result.dtype == np.uint16, f"Expected {type=}, got {result.dtype=}"

        return

    def test_predict_age_at_death_with_types(self):
        for type in [np.int16, np.int32, np.int64, np.uint16, np.uint32, np.uint64, np.float32, np.float64]:
            # Use 88 as maximum age to avoid overflow with int16
            max_year = 88 if type == np.int16 else 100
            ages_years = np.random.randint(max_year + 1, size=1024).astype(type)
            ages_days = ages_years * 365 + np.random.randint(365, size=ages_years.shape[0]).astype(ages_years.dtype)
            result = self.estimator.predict_age_at_death(ages_days, max_year=max_year)
            assert result.dtype == type, f"Expected {type=}, got {result.dtype=}"

        return

    def test_invalid_input_scenario1(self):
        input = np.array(self.cumulative_deaths)
        imiddle = input.shape[0] // 2
        input[imiddle], input[imiddle + 1] = input[imiddle + 1], input[imiddle]
        with pytest.raises(ValueError, match=re.escape("Input values should be monotonically non-decreasing:")):
            _ = KaplanMeierEstimator(input)

        return

    def test_invalid_input_scenario2(self):
        input = np.array(self.cumulative_deaths, dtype=np.int32)
        imiddle = input.shape[0] // 2
        input -= input[imiddle]
        with pytest.raises(ValueError, match=re.escape("Input values should be >= 0:")):
            _ = KaplanMeierEstimator(input)

        return

    def test_simple_sample(self):
        cumulative = np.full(100, 1000, dtype=np.uint32).cumsum().astype(np.uint32)
        estimator = KaplanMeierEstimator(cumulative)
        ages = np.array([0, 1, 2, 3, 5, 8], dtype=np.uint32)
        predicted = estimator.sample(ages)
        assert np.all(predicted >= ages), f"predicted ages at death should be >= current ages {predicted=}, {ages=}"
        assert np.all(predicted < len(cumulative)), f"predicted ages at death should be < {len(cumulative)} ({predicted.max()=})"

        return

    def test_limited_sample(self):
        cumulative = np.full(100, 1000, dtype=np.uint32).cumsum().astype(np.uint32)
        estimator = KaplanMeierEstimator(cumulative)
        OLDEST = 50
        ages = np.random.randint(0, OLDEST, 10_000, dtype=np.uint32)
        MAXIMUM = 80
        predicted = estimator.sample(ages, max_index=MAXIMUM)
        assert np.all(predicted >= ages), f"predicted ages at death should be >= current ages {predicted=}, {ages=}"
        assert np.all(predicted <= MAXIMUM), f"predicted ages at death should be < {MAXIMUM} ({predicted.max()=})"

        return

    def test_max_year_out_of_range(self):
        estimator = KaplanMeierEstimator(self.cumulative_deaths)  # Should _definitely_ match...
        ages_years = np.array([0, 1, 1, 2, 3, 5, 8], dtype=np.uint32)
        with pytest.raises(ValueError, match=re.escape("max_year=115 must be less than len(self.cumulative_deaths)=101")):
            estimator.predict_year_of_death(ages_years, max_year=115)

        return

    def test_max_index_out_of_range(self):
        estimator = KaplanMeierEstimator(self.cumulative_deaths)  # Should _definitely_ match...
        ages_years = np.array([0, 1, 1, 2, 3, 5, 8], dtype=np.uint32)
        with pytest.raises(ValueError, match=re.escape("max_index=115 must be less than len(self.cumulative_deaths)=101")):
            estimator.sample(ages_years, max_index=115)

        return

    def test_input_years_out_of_range(self):
        estimator = KaplanMeierEstimator(self.cumulative_deaths)
        ages_years = np.array([0, 1, 1, 2, 3, 5, 80, 85, 90, 95, 100], dtype=np.uint32)
        with pytest.raises(
            ValueError,
            match=re.escape("all current ages must be less than or equal to max_year=79 (ages_years.max()=100)"),
        ):
            estimator.predict_year_of_death(ages_years, max_year=79)

        return

    def test_input_indices_out_of_range(self):
        estimator = KaplanMeierEstimator(self.cumulative_deaths)
        age_indices = np.array([0, 1, 1, 2, 3, 5, 80, 85, 90, 95, 100], dtype=np.uint32)
        with pytest.raises(
            ValueError,
            match=re.escape("all current indices must be less than or equal to max_index=79 (current.max()=100)"),
        ):
            estimator.sample(age_indices, max_index=79)

        return


def _compare_estimators(etest, eexpected, max_year=100):
    """
    Compare two estimators using the Kolmogorov-Smirnov test.
    This function generates random ages, uses the provided estimators to predict
    the year of death, and then compares the cumulative distribution functions
    (CDFs) of the predictions using the Kolmogorov-Smirnov test.
    Args:
        etest: The estimator to be tested. Must have a method `predict_year_of_death`.
        eexpected: The expected estimator to compare against. Must have a method `predict_year_of_death`.
        max_year (int, optional): The maximum year to consider for predictions. Defaults to 100.
    Returns:
        KstestResult: The result of the Kolmogorov-Smirnov test comparing the two CDFs.
    """

    ages_years = np.random.randint(max_year + 1, size=10_000_000, dtype=np.int32)  # randint() = [0, max) so add 1
    test = etest.predict_year_of_death(ages_years, max_year)
    expected = eexpected.predict_year_of_death(ages_years, max_year)

    ctest = np.zeros(max_year + 1, dtype=np.int32)
    np.add.at(ctest, test, 1)  # get histogram
    f_of_x = ctest.cumsum()
    cexpected = np.zeros(max_year + 1, dtype=np.int32)
    np.add.at(cexpected, expected, 1)  # get histogram
    g_of_x = cexpected.cumsum()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="ks_2samp: Exact calculation unsuccessful.*",
            category=RuntimeWarning,
        )
        test = kstest(f_of_x, g_of_x)

    return test


if __name__ == "__main__":
    unittest.main()
