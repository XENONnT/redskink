from pydantic import BaseModel, validator
from typing import Dict, Optional, Any, Literal, Callable
import numpy as np
from scipy import stats
from copy import deepcopy
from multihist import Hist1d


def energy_res(energy, a=25.8, b=1.429):
    """Return energy resolution in keV.

    :param energy: true energy in keV :return: energy resolution in keV

    """
    # Reference for the values of a,b:
    # xenon:xenonnt:analysis:ntsciencerun0:g1g2_update#standard_gaussian_vs_skew-gaussian_yue
    return (np.sqrt(energy) * a + energy * b) / 100


def smearing_mono_gaussian(
    hist: Any,
    smearing_a: float,
    smearing_b: float,
    peak_energy: float,
    bins: Optional[np.ndarray] = None,
):
    """Smear a mono-energetic peak with a Gaussian."""

    if bins is None:
        # create an emptyzero histogram with the same binning as the input histogram
        data = stats.norm.pdf(
            hist.bin_centers,
            loc=peak_energy,
            scale=energy_res(peak_energy, smearing_a, smearing_b),
        )
        hist_smeared = Hist1d(data=np.zeros_like(data), bins=hist.bin_edges)
        hist_smeared.histogram = data
    else:
        # use the bins that set by the user
        bins = np.array(bins)
        if bins.size <= 1:
            raise ValueError("bins must have at least 2 elements")
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        data = stats.norm.pdf(
            bin_centers,
            loc=peak_energy,
            scale=energy_res(peak_energy, smearing_a, smearing_b),
        )
        # create an empty histogram with the user-defined binning
        hist_smeared = Hist1d(data=np.zeros_like(data), bins=bins)
        hist_smeared.histogram = data

    return hist_smeared


def smearing_hist_gaussian(
    hist: Any,
    smearing_a: float,
    smearing_b: float,
    bins: Optional[np.ndarray] = None,
):
    """Smear a histogram with Gaussian. This allows for non-uniform histogram binning.

    :param hist: the spectrum we want to smear :param bins: bin edges of the returned spectrum
    :return: smeared histogram in the same unit as input spectrum

    """
    assert isinstance(hist, Hist1d), "Only Hist1d object is supported"
    if bins is None:
        # set the bins to the bin edges of the input histogram
        bins = hist.bin_edges
    elif bins.size <= 1:
        raise ValueError("bins must have at least 2 elements")
    bins = np.array(bins)

    e_true_s, rates, bin_volumes = hist.bin_centers, hist.histogram, hist.bin_volumes()
    mask = np.where(e_true_s > 0)
    e_true_s = e_true_s[mask]
    rates = rates[mask]
    bin_volumes = bin_volumes[mask]

    e_smeared_s = 0.5 * (bins[1:] + bins[:-1])
    smeared = np.zeros_like(e_smeared_s)

    for idx, e_smeared in enumerate(e_smeared_s):
        probs = (
            stats.norm.pdf(
                e_smeared,
                loc=e_true_s,
                scale=energy_res(e_true_s, smearing_a, smearing_b),
            )
            * bin_volumes
        )
        smeared[idx] = np.sum(probs * rates)

    hist_smeared = Hist1d.from_histogram(smeared, bins)

    return hist_smeared


def biasing_hist_arctan(hist: Any, A: float = 0.01977, k: float = 0.01707):
    """Apply a constant bias to a histogram.

    :param hist: the spectrum we want to apply the bias to :param bias: the bias to apply to the
    spectrum :return: the spectrum with the bias applied

    """
    assert isinstance(hist, Hist1d), "Only Hist1d object is supported"
    true_energy = hist.bin_centers
    h_bias = deepcopy(hist)
    bias_derivative = A * k / (1 + k**2 * true_energy**2)
    h_bias.histogram *= 1 / (1 + bias_derivative)
    return h_bias


def efficiency_hist_constant(hist: Any, efficiency_constant: float):
    """Apply a constant efficiency to a histogram.

    :param hist: the spectrum we want to apply the efficiency to :param efficiency: the efficiency
    to apply to the spectrum :return: the spectrum with the efficiency applied

    """
    assert isinstance(hist, Hist1d), "Only Hist1d object is supported"
    assert 0 <= efficiency_constant <= 1, "Efficiency must be between 0 and 1"
    hist.histogram = hist.histogram * efficiency_constant
    return hist


MODELS: Dict[str, Dict[str, Callable]] = {
    "smearing": {
        "gaussian": smearing_hist_gaussian,
        "mono_gaussian": smearing_mono_gaussian,
    },
    "bias": {"arctan": biasing_hist_arctan},
    "efficiency": {
        "constant": efficiency_hist_constant,
    },
}


# input: model name, parameters, transformation mode
class Transformation(BaseModel):
    parameters: Dict[str, float]
    action: Literal["bias", "smearing", "efficiency"]
    model: str

    @validator("model")
    @classmethod
    def check_model(cls, v, values):
        """Check if the model exists for the given action."""
        if v not in MODELS[values["action"]]:
            raise ValueError(f"Model {v} not found for action {values['action']}")
        return v

    def apply_transformation(self, histogram: Hist1d):
        chosen_model = MODELS[self.action][self.model]
        return chosen_model(histogram, **self.parameters)
