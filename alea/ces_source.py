from typing import Dict, Literal
import numpy as np
from copy import deepcopy
from scipy.interpolate import interp1d

from inference_interface import template_to_multihist
from blueice import HistogramPdfSource, Source
from blueice.exceptions import PDFNotComputedException

from multihist import Hist1d
from alea.ces_transformation import Transformation

MINIMAL_ENERGY_RESOLUTION = 0.05


class CESTemplateSource(HistogramPdfSource):
    def __init__(self, config: Dict, *args, **kwargs):
        """Initialize the TemplateSource."""
        # override the default interpolation method
        if "pdf_interpolation_method" not in config:
            config["pdf_interpolation_method"] = "piecewise"
        super().__init__(config, *args, **kwargs)

    def _load_inputs(self):
        """Load the inputs needed for a histogram source from the config."""
        self.ces_space = self.config["analysis_space"][0][1]
        self.max_e = np.max(self.ces_space)
        self.min_e = np.min(self.ces_space)
        self.templatename = self.config["templatename"]
        self.histname = self.config["histname"]

    def _load_true_histogram(self):
        """Load the true spectrum from the template (no transformation applied)"""
        h = template_to_multihist(self.templatename, self.histname, hist_to_read=Hist1d)
        if self.config.get("zero_filling_for_outlier", False):
            bins = h.bin_centers
            inter = interp1d(bins, h.histogram, bounds_error=False, fill_value=0)
            width = self.ces_space[1] - self.ces_space[0]
            new_hist = Hist1d(bins=np.arange(0, self.max_e + width, width))
            new_hist.histogram = inter(new_hist.bin_centers)
            return new_hist
        else:
            return h

    def _check_histogram(self, h: Hist1d):
        """Check if the histogram has expected binning."""
        # We only take 1d histogram in the ces axes
        if not isinstance(h, Hist1d):
            raise ValueError("Only Hist1d object is supported")
        if self.ces_space.ndim != 1:
            raise ValueError("Only 1d analysis space is supported")
        if np.min(h.histogram) < 0:
            raise AssertionError(
                f"There are bins for source {self.templatename} with negative entries."
            )

        # check if the histogram overlaps the analysis space.
        histogram_max = np.max(h.bin_edges)
        histogram_min = np.min(h.bin_edges)
        if self.min_e > histogram_max or self.max_e < histogram_min:
            raise ValueError(
                f"The histogram edge ({histogram_min},{histogram_max}) \
                does not overlap with the analysis space ({self.min_e},{self.max_e}) \
                remove this background please:)"
            )

    def _create_transformation(
        self, transformation_type: Literal["smearing", "bias", "efficiency"]
    ):
        """Create a transformation object based on the transformation type."""
        if self.config.get(f"apply_{transformation_type}", True):
            parameters_key = f"{transformation_type}_parameters"
            model_key = f"{transformation_type}_model"

            if model_key not in self.config:
                raise ValueError(f"{transformation_type.capitalize()} model is not provided")

            if parameters_key not in self.config:
                raise ValueError(f"{transformation_type.capitalize()} parameters are not provided")
            else:
                parameter_list = self.config[parameters_key]
                # to get the values we need to iterate over the list and use self.config.get
                combined_parameter_dict = {k: self.config.get(k) for k in parameter_list}

            # Also take the peak_energy parameter if it is a mono smearing model
            if "mono" in self.config[model_key]:
                combined_parameter_dict["peak_energy"] = self.config["peak_energy"]

            return Transformation(
                parameters=combined_parameter_dict,
                action=transformation_type,
                model=self.config[model_key],
            )
        return None

    def _transform_histogram(self, h: Hist1d):
        """Apply the transformations to the histogram."""
        # Create transformations for efficiency, smearing, and bias
        smearing_transformation = self._create_transformation("smearing")
        bias_transformation = self._create_transformation("bias")
        efficiency_transformation = self._create_transformation("efficiency")

        # Apply the transformations to the histogram
        if smearing_transformation is not None:
            h = smearing_transformation.apply_transformation(h)
        if bias_transformation is not None:
            h = bias_transformation.apply_transformation(h)
        if efficiency_transformation is not None:
            h = efficiency_transformation.apply_transformation(h)
        return h

    def _normalize_histogram(self, h: Hist1d):
        """Normalize the histogram and calculate the rate of the source."""
        # To avoid confusion, we always normalize the histogram, regardless of the bin volume
        # So the unit is always events/year/keV, the rate multipliers are always in terms of that
        total_integration = np.sum(h.histogram * h.bin_volumes())
        h.histogram = h.histogram.astype(np.float64)
        total_integration = total_integration.astype(np.float64)

        h.histogram /= total_integration

        # Apply the transformations to the histogram
        h = self._transform_histogram(h)

        # Calculate the integration of the histogram after all transformations
        # Only from min_e to max_e
        left_edges = h.bin_edges[:-1]
        right_edges = h.bin_edges[1:]
        outside_index = np.where((left_edges < self.min_e) | (right_edges > self.max_e))
        h.histogram[outside_index] = 0

        self._bin_volumes = h.bin_volumes()
        self._n_events_histogram = h.similar_blank_histogram()

        # Note that it already does what "fraction_in_roi" does in the old code
        # So no need to do again
        integration_after_transformation_in_roi = np.sum(h.histogram * h.bin_volumes())

        self.events_per_year = (
            integration_after_transformation_in_roi * self.config["rate_multiplier"]
        )
        self.events_per_day = self.events_per_year / 365

        # For pdf, we need to normalize the histogram to 1 again
        return h, integration_after_transformation_in_roi

    def build_histogram(self):
        """Build the histogram of the source.

        It's always called during the initialization of the source. So the attributes are set here.

        """
        print("Building histogram")
        self._load_inputs()
        h = self._load_true_histogram()
        self._check_histogram(h)
        h, frac_in_roi = self._normalize_histogram(h)
        h_pdf = h
        h_pdf /= frac_in_roi
        self._pdf_histogram = h_pdf
        self.set_dtype()

    def simulate(self, n_events: int):
        """Simulate events from the source.

        Args:
            n_events (int): The number of events to simulate.

        Returns:
            numpy.ndarray: The simulated events.

        """
        dtype = [
            ("ces", float),
            ("source", int),
        ]
        ret = np.zeros(n_events, dtype=dtype)
        ret["ces"] = self._pdf_histogram.get_random(n_events)
        return ret

    def compute_pdf(self):
        """Compute the PDF of the source."""
        self.build_histogram()
        Source.compute_pdf(self)

    def pdf(self, *args):
        """Interpolate the PDF of the source to return a function."""
        # override the default interpolation method in blueice (RegularGridInterpolator)
        if not self.pdf_has_been_computed:
            raise PDFNotComputedException(
                "%s: Attempt to call a PDF that has not been computed" % self
            )

        method = self.config["pdf_interpolation_method"]

        if method == "linear":
            if not hasattr(self, "_pdf_interpolator"):
                # First call:
                # Construct a linear interpolator between the histogram bins
                self._pdf_interpolator = interp1d(
                    self._pdf_histogram.bin_centers,
                    self._pdf_histogram.histogram,
                )
            bcs = self._pdf_histogram.bin_centers
            clipped_data = np.clip(args, bcs.min(), bcs.max())

            return self._pdf_interpolator(np.transpose(clipped_data))

        elif method == "piecewise":
            return self._pdf_histogram.lookup(*args)

        else:
            raise NotImplementedError("PDF Interpolation method %s not implemented" % method)

    def set_dtype(self):
        """Set the data type of the source."""
        self.dtype = [
            ("ces", float),
            ("source", int),
        ]


class CESMonoenergySource(CESTemplateSource):
    def _load_inputs(self):
        """Load needed inputs for a monoenergetic source from the config."""
        self.ces_space = self.config["analysis_space"][0][1]
        self.max_e = np.max(self.ces_space)
        self.min_e = np.min(self.ces_space)
        try:
            self.mu = self.config["peak_energy"]
        except KeyError:
            raise ValueError("peak_energy is not provided in the config")

    def _load_true_histogram(self):
        """Create a fake histogram with a single peak at the peak energy."""
        number_of_bins = int((self.max_e - self.min_e) / MINIMAL_ENERGY_RESOLUTION)
        h = Hist1d(
            data=np.repeat(self.mu, 1),
            bins=number_of_bins,
            range=(self.min_e, self.max_e),
        )
        h.histogram = h.histogram.astype(np.float64)
        self.config["smearing_model"] = "mono_" + self.config["smearing_model"]
        return h


class CESFlatSource(CESTemplateSource):
    def _load_inputs(self):
        """Load needed inputs for a flat source from the config."""
        self.ces_space = self.config["analysis_space"][0][1]
        self.max_e = np.max(self.ces_space)
        self.min_e = np.min(self.ces_space)

    def _load_true_histogram(self):
        """Create a histogram for the flat source."""
        number_of_bins = int((self.max_e - self.min_e) / MINIMAL_ENERGY_RESOLUTION)
        h = Hist1d(
            data=np.linspace(self.min_e, self.max_e, number_of_bins),
            bins=number_of_bins,
            range=(self.min_e, self.max_e),
        )
        h.histogram = h.histogram.astype(np.float64)
        return h
