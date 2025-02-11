import warnings
from typing import List, Dict, Callable, Optional, Union, cast
from pydoc import locate
import itertools
from copy import deepcopy

from tqdm import tqdm
import numpy as np
import scipy.stats as stats
from blueice.likelihood import LogAncillaryLikelihood, LogLikelihoodSum
from inference_interface import dict_to_structured_array, structured_array_to_dict

from alea.model import StatisticalModel
from alea.parameters import Parameters
from alea.simulators import BlueiceDataGenerator
from alea.utils import ReadOnlyDict
from alea.utils import adapt_likelihood_config_for_blueice, get_template_folder_list, load_yaml


class BlueiceExtendedModel(StatisticalModel):
    """A statistical model based on blueice likelihoods.

    This class extends the ``StatisticalModel`` class and provides methods
    for generating data and computing likelihoods based on blueice.

    Attributes:
        parameters (Parameters): Parameters object containing the parameters of the model.
        data (dict): Data of the statistical model.
        is_data_set (bool): Whether data is set.
        _likelihood (LogLikelihoodSum): A blueice LogLikelihoodSum instance.
        likelihood_names (list): List of likelihood names.
        livetime_parameter_names (list): List of the name of the livetime of each term,
            None if not specified
        data_generators (list): List of data generators for each likelihood term.

    Args:
        parameter_definition (dict): A dictionary defining the model parameters.
        likelihood_config (dict): A dictionary defining the likelihood.

    Todo:
        analysis_space could be inferred from the data
        (assert that all sources have the same analysis_space)

    """

    def __init__(self, parameter_definition: dict, likelihood_config: dict, **kwargs):
        """Initializes the statistical model.

        Args:
            parameter_definition (dict): A dictionary defining the model parameters.
            likelihood_config (dict): A dictionary defining the likelihood.

        """
        super().__init__(parameter_definition=parameter_definition, **kwargs)
        self._likelihood = self._build_ll_from_config(
            likelihood_config, template_path=kwargs.get("template_path", None)
        )
        self.likelihood_names = [t["name"] for t in likelihood_config["likelihood_terms"]]
        self.likelihood_names.append("ancillary")
        self.livetime_parameter_names = [
            t.get("livetime_parameter", None) for t in likelihood_config["likelihood_terms"]
        ]
        self.livetime_parameter_names += [None]  # ancillary likelihood
        self.data_generators = self._build_data_generators()
        self._set_default_ptype()

    @classmethod
    def from_config(cls, config_file_path: str, **kwargs) -> "BlueiceExtendedModel":
        """Initializes the statistical model from a yaml config file.

        Args:
            config_file_path (str): Path to the yaml config file.

        Returns:
            BlueiceExtendedModel: Statistical model.

        """
        config = load_yaml(config_file_path)
        return cls(**{**config, **kwargs})

    @property
    def data(self) -> Union[dict, list]:
        """Return the data of the statistical model."""
        return super().data

    @data.setter
    def data(self, data: Union[dict, list]):
        """Overrides default setter. Will also set the data of the blueice ll. Data-sets are
        expected to be in the form of a list of one or more structured arrays representing the data-
        sets of one or more likelihood terms.

        Args:
            data (dict or list): Data of the statistical model.
                If data is a list, it must be a list of length len(self.likelihood_names) + 1.

        Raises:
            Warning: If data is not a list of length len(self.likelihood_names) + 1.

        Caution:
            The self._data is read-only, so you can not change the data after it is set.
            In order to change the data, you have to set the data again, using self.data = ***.

        """
        # iterate through all likelihood terms and set the science data in the blueice ll
        # last entry in data are the generate_values
        if isinstance(data, list):
            if len(data) != len(self.likelihood_names) + 1:
                if len(data) == len(self.likelihood_names):
                    warnings.warn(
                        f"If data is not a list of length {len(self.likelihood_names) + 1}, "
                        f"only the science data and ancillary will be set."
                    )
                else:
                    raise ValueError(
                        "You should at least provide data for all likelihood terms, "
                        "including science data and ancillary."
                    )
            data = dict(zip((self.likelihood_names + ["generate_values"])[: len(data)], data))
        for i, (dataset_name, d) in enumerate(data.items()):
            if dataset_name != "generate_values":
                ll_term = self.likelihood_list[i]
                if dataset_name != ll_term.pdf_base_config["name"]:
                    raise ValueError("Likelihood names do not match.")
                ll_term.set_data(d)

        self._data = ReadOnlyDict(data)
        self.is_data_set = True

    def get_source_name_list(self, likelihood_name: str) -> list:
        """Return a list of source names for a given likelihood term. The order is the same as used
        in the ``source`` column of the data, so this can be used to map the indices provided in the
        data to a source name.

        Args:
            likelihood_name (str): Name of the likelihood.

        Returns:
            list: List of source names.

        """
        ll_index = self.likelihood_names.index(likelihood_name)
        return self.likelihood_list[ll_index].source_name_list

    @property
    def all_source_names(self) -> list:
        """Return a set of possible source names from all likelihood terms.

        Args:
            likelihood_name (str): Name of the likelihood.
        Returns:
            set: set of source names.

        """
        source_names = set(
            itertools.chain.from_iterable([ll.source_name_list for ll in self.likelihood_list[:-1]])
        )
        return sorted(source_names)

    @property
    def likelihood_list(self) -> List:
        """Return a list of likelihood terms."""
        return self._likelihood.likelihood_list

    @property
    def likelihood_parameters(self) -> List:
        """Return a list of likelihood parameters."""
        return self._likelihood.likelihood_parameters

    def get_expectation_values(self, per_likelihood_term=False, **kwargs) -> dict:
        """Return total expectation values (summed over all likelihood terms with the same name)
        given a number of named parameters (kwargs)

        Args:
            per_likelihood_term (bool): If True, return expectation values
                per likelihood term. Otherwise, sum each source over all likelihood terms.
            kwargs: Named parameters

        Returns:
            dict: Dictionary of expectation values. If per_likelihood_term is True, the dictionary
                has the form {likelihood_name: {source_name: expectation_value, ...}, ...}.

        Todo:
            Make a self.likelihood_temrs dict with the likelihood names as keys and
            the corresponding likelihood terms as values.

        """
        ret = cast(Dict[str, Dict[str, float]], {})
        # prepare generate_values
        if not self.parameters.values_in_fit_limits(**kwargs):
            raise ValueError("Values are not within fit limits")
        generate_values = self.parameters(**kwargs)

        # ancillary likelihood does not contribute
        for ll_name, lt_name in zip(
            self.likelihood_names[:-1],
            self.livetime_parameter_names,
        ):
            ret[ll_name] = {}
            ll_index = self.likelihood_names.index(ll_name)
            lt = generate_values.get(lt_name, None)
            # compute the mus
            self.data_generators[ll_index].compute_pdfs_and_mus(**generate_values, livetime_days=lt)
            mus = self.data_generators[ll_index].mus
            for n, mu in zip(self.likelihood_list[ll_index].source_name_list, mus):
                ret[ll_name][n] = mu
            # sort by source name
            ret[ll_name] = dict(sorted(ret[ll_name].items(), key=lambda item: item[0]))
        if not per_likelihood_term:
            # sum over sources with same names of all likelihood terms
            ret = {
                n: sum([ret[ll_name].get(n, 0.0) for ll_name in ret.keys()])  # type: ignore
                for n in self.all_source_names
            }

        return ret

    def get_source_histograms(self, likelihood_name: str, expected_events=False, **kwargs) -> dict:
        """Return the pdfs or histograms of all sources for a given likelihood term.

        Args:
            likelihood_name (str): Name of the likelihood term.
            expected_events (bool): If True, return the histograms containing
                the number of expected events.
            kwargs: Named parameters.

        Returns:
            dict: Dictionary containing a multihist object for each source.

        """
        if likelihood_name not in self.likelihood_names:
            raise ValueError(f"Likelihood {likelihood_name} not found.")
        elif likelihood_name == "ancillary":
            raise ValueError("No source histograms for ancillary likelihood.")

        ll_index = self.likelihood_names.index(likelihood_name)

        # prepare generate_values
        if not self.parameters.values_in_fit_limits(**kwargs):
            raise ValueError("Values are not within fit limits")
        generate_values = self.parameters(**kwargs)
        lt_name = self.livetime_parameter_names[ll_index]
        # change keyof lt_name to "livetime_days" if it is in the generate_values
        if lt_name in generate_values:
            generate_values["livetime_days"] = generate_values.pop(lt_name)

        # compute the pdfs
        self.data_generators[ll_index].compute_pdfs_and_mus(**generate_values)
        source_histograms = deepcopy(self.data_generators[ll_index].source_histograms)

        if expected_events:
            mus = self.data_generators[ll_index].mus
            for source_name, hist in source_histograms.items():
                source_index = self.get_source_name_list(likelihood_name).index(source_name)
                hist.histogram *= mus[source_index]
        # for unbinned likelihoods we need to divide by the bin volumes
        elif not expected_events and not self.data_generators[ll_index].binned:
            for hist in source_histograms.values():
                hist.histogram /= hist.bin_volumes()

        # sort the source_histograms by source name
        source_histograms = dict(sorted(source_histograms.items(), key=lambda item: item[0]))

        return source_histograms

    def _process_blueice_config(self, config, template_folder_list):
        """Process the blueice config from config."""
        pdf_base_config = adapt_likelihood_config_for_blueice(config, template_folder_list)
        pdf_base_config["livetime_days"] = self.parameters[
            pdf_base_config["livetime_parameter"]
        ].nominal_value
        for p in self.parameters:
            # adding the nominal rate values will screw things up in blueice!
            # So here we're just adding the nominal values of all other parameters
            if p.ptype != "rate":
                pdf_base_config[p.name] = pdf_base_config.get(p.name, p.nominal_value)

        # sanity checks
        for source in config["sources"]:
            if "name" not in source:
                raise ValueError("No name specified for source.")
            if "parameters" not in source:
                raise ValueError(f"No parameters specified for source {source['name']}.")
            if set(source.get("named_parameters", [])) - set(source["parameters"]):
                raise ValueError(
                    f"Named parameters {source['named_parameters']} are not all in the "
                    f"parameter list {source['parameters']} of source {source['name']}."
                )

        # add all parameters to extra_dont_hash for each source unless it is used:
        for i, source in enumerate(config["sources"]):
            parameters_to_ignore = self._get_parameters_to_ignore(source)
            # ignore all shape parameters known to this model not named specifically
            # in the source:
            pdf_base_config["sources"][i]["extra_dont_hash_settings"] = parameters_to_ignore

        # get blueice likelihood_config if it's given
        likelihood_config = config.get("likelihood_config", None)

        source_wise_interpolation = config.get("source_wise_interpolation", True)

        if source_wise_interpolation and likelihood_config:
            if likelihood_config.get("morpher") == "IndexMorpher":
                raise ValueError("Source-wise interpolation is not yet supported for IndexMorpher.")

        blueice_config = {
            "pdf_base_config": pdf_base_config,
            "likelihood_config": likelihood_config,
            "source_wise_interpolation": source_wise_interpolation,
        }
        return blueice_config

    def _get_parameters_to_ignore(self, source):
        parameters_to_ignore: List[str] = [
            p.name
            for p in self.parameters
            if (p.ptype in ["shape", "index", "needs_reinit"])
            and (p.name not in source["parameters"])
        ]
        # no efficiency affects PDF:
        parameters_to_ignore += [p.name for p in self.parameters if (p.ptype == "efficiency")]
        parameters_to_ignore += source.get("extra_dont_hash_settings", [])
        return parameters_to_ignore

    def _build_ll_from_config(
        self, likelihood_config: dict, template_path: Optional[str] = None
    ) -> "LogLikelihoodSum":
        """Iterate through all likelihood terms and build blueice likelihood instances.

        Args:
            likelihood_config (dict): A dictionary defining the likelihood.

        Returns:
            LogLikelihoodSum: A blueice LogLikelihoodSum instance.

        """
        lls = []

        template_folder_list = get_template_folder_list(
            likelihood_config, extra_template_path=template_path
        )

        # Iterate through each likelihood term in the configuration
        for config in likelihood_config["likelihood_terms"]:
            blueice_config = self._process_blueice_config(config, template_folder_list)
            blueice_config["source_wise_interpolation"] = config.get(
                "source_wise_interpolation", True
            )
            print(blueice_config["source_wise_interpolation"])

            likelihood_class = cast(Callable, locate(config["likelihood_type"]))
            if likelihood_class is None:
                raise ValueError(f"Could not find {config['likelihood_type']}!")
            ll = likelihood_class(**blueice_config)

            for source in config["sources"]:
                # set rate parameters
                rate_parameters = [
                    p for p in source["parameters"] if self.parameters[p].ptype == "rate"
                ]
                if len(rate_parameters) != 1:
                    raise ValueError(
                        f"Source {source['name']} must have exactly one rate parameter."
                    )
                rate_parameter = rate_parameters[0]
                if rate_parameter.endswith("_rate_multiplier"):
                    rate_parameter = rate_parameter.replace("_rate_multiplier", "")
                    # The ancillary term is handled in CustomAncillaryLikelihood
                    ll.add_rate_parameter(rate_parameter, log_prior=None)
                else:
                    raise NotImplementedError(
                        "Only rate multipliers that end on _rate_multiplier"
                        " are currently supported."
                    )

                # set efficiency parameters
                if source.get("efficiency_name", None):
                    self._set_efficiency(source, ll)

                # set shape parameters
                shape_parameters = [
                    p
                    for p in source["parameters"]
                    if self.parameters[p].ptype in ["shape", "index"]
                ]
                for p in shape_parameters:
                    anchors = self.parameters[p].blueice_anchors
                    if anchors is None:
                        raise ValueError(f"Shape parameter {p} does not have any anchors.")
                    # The ancillary term is handled in CustomAncillaryLikelihood
                    ll.add_shape_parameter(p, anchors=anchors, log_prior=None)

            n_cores = config.get("n_cores", 1)
            if n_cores == 1:
                ll.prepare()
            else:
                ll.prepare(n_cores=n_cores)
            lls.append(ll)

        # ancillary likelihood
        ll = CustomAncillaryLikelihood(self.parameters.with_uncertainty)
        lls.append(ll)

        likelihood_weights = likelihood_config.get("likelihood_weights", None)
        return LogLikelihoodSum(lls, likelihood_weights=likelihood_weights)

    def _build_data_generators(self) -> list:
        """Build data generators for all likelihood terms.

        Returns:
            list: List of data generators for each likelihood term.

        Todo:
            Also implement data generator for ancillary ll term.

        """
        # last one is AncillaryLikelihood
        data_generators = []
        for ll_term in tqdm(self.likelihood_list[:-1], desc="building data generators"):
            methods = [s.config["pdf_interpolation_method"] for s in ll_term.base_model.sources]
            # make sure that all sources have the same pdf_interpolation_method
            if len(set(methods)) != 1:
                raise ValueError("All sources must have the same pdf_interpolation_method.")
            method = methods[0]
            if method == "piecewise":
                data_generators.append(BlueiceDataGenerator(ll_term))
            elif method == "linear":
                raise NotImplementedError(
                    "Linear interpolation is not yet supported."
                    " Choose piecewise as pdf_interpolation_method."
                )
            else:
                raise ValueError(f"Unknown pdf_interpolation_method {method}.")
        return data_generators

    def _ll(self, **generate_values) -> float:
        livetime_days = [generate_values.get(ln, None) for ln in self.livetime_parameter_names]
        return self._likelihood(livetime_days=livetime_days, **generate_values)

    def _generate_data(self, **generate_values) -> dict:
        """Generate data for all likelihood terms and ancillary likelihood.

        Keyword Args:
            generate_values (dict): A dictionary of parameter values.

        Returns:
            dict: A dict of data-sets,
            with key of the likelihood term name, "ancillary" and "generate_values".

        """
        # generate_values are already filtered and filled by the nominal values
        data = self._generate_science_data(**generate_values)
        ancillary_keys = self.parameters.with_uncertainty.names
        generate_values_anc = {k: v for k, v in generate_values.items() if k in ancillary_keys}
        data["ancillary"] = self._generate_ancillary(**generate_values_anc)
        data["generate_values"] = dict_to_structured_array(generate_values)
        return data

    def store_data(self, file_name, data_list, data_name_list=None, metadata=None):
        """Store data in a file.

        Append the generate_values to the data_name_list.

        """
        if data_name_list is None:
            data_name_list = self.likelihood_names + ["generate_values"]
        super().store_data(file_name, data_list, data_name_list, metadata)

    def _generate_science_data(self, **generate_values) -> dict:
        """Generate the science data for all likelihood terms except the ancillary likelihood."""
        livetime_days = [generate_values.get(ln, None) for ln in self.livetime_parameter_names]
        science_data = [
            gen.simulate(livetime_days=lt, **generate_values)
            for gen, lt in zip(self.data_generators, livetime_days)
        ]
        return dict(zip(self.likelihood_names[:-1], science_data))

    def _generate_ancillary(self, **generate_values) -> dict:
        """Generate data for the ancillary likelihood.

        Keyword Args:
            generate_values (dict): A dictionary of parameter values.

        Returns:
            numpy.array: A numpy structured array of ancillary measurements.

        """
        ancillary = {}
        anc_ll = self.likelihood_list[-1]
        ancillary_generators = anc_ll._get_constraint_functions(**generate_values)
        for name, gen in ancillary_generators.items():
            parameter_meas = gen.rvs()
            # correct parameter_meas if out of bounds
            param = self.parameters[name]
            if not param.value_in_fit_limits(parameter_meas):
                if param.fit_limits[0] is not None and parameter_meas < param.fit_limits[0]:
                    parameter_meas = param.fit_limits[0]
                elif param.fit_limits[1] is not None and parameter_meas > param.fit_limits[1]:
                    parameter_meas = param.fit_limits[1]
            ancillary[name] = parameter_meas

        return dict_to_structured_array(ancillary)

    def _set_efficiency(self, source: dict, ll):
        """Set the efficiency of a source in the blueice ll.

        Args:
            source (dict): A dictionary defining the source.
            ll (LogLikelihood): A blueice LogLikelihood instance.

        Raises:
            ValueError: If the efficiency_name is not specified in the source.

        """
        if "efficiency_name" not in source:
            raise ValueError(f"Unspecified efficiency_name for source {source['name']:s}")
        efficiency_name = source["efficiency_name"]

        if efficiency_name not in source["parameters"]:
            raise ValueError(
                f"The efficiency_name for source {source['name']:s} is not in its parameter list"
            )
        efficiency_parameter = self.parameters[efficiency_name]

        if efficiency_parameter.ptype != "efficiency":
            raise ValueError(f"The parameter {efficiency_name:s} must be an efficiency")
        limits = efficiency_parameter.fit_limits

        if limits[0] < 0:
            raise ValueError(
                f"Efficiency parameters including {efficiency_name:s}"
                " must be constrained to be nonnegative"
            )
        if ~np.isfinite(limits[1]):
            raise ValueError(
                f"Efficiency parameters including {efficiency_name:s}"
                " must be constrained to be finite"
            )
        ll.add_shape_parameter(efficiency_name, anchors=(limits[0], limits[1]))

    def _set_default_ptype(self):
        """Check if all parameters have a ptype that is in the list of allowed ptypes.

        If no ptype is specified, set the default ptype "needs_reinit".

        """
        allowed_ptypes = ["rate", "shape", "index", "efficiency", "livetime", "needs_reinit"]
        default_ptype = "needs_reinit"
        for p in self.parameters:
            if p.ptype is None:
                p.ptype = default_ptype
            elif p.ptype not in allowed_ptypes:
                raise ValueError(
                    f"Parameter {p.name} has ptype {p.ptype} which is not in the list of "
                    f"allowed ptypes: {allowed_ptypes}."
                )

    def store_real_data(self, file_name: str, real_data_list: list, metadata=None):
        """Store real data in a file with toydata format.

        Args:
            file_name (str): Name of the file.
            real_data_list (list): List of np.array of real data.

        """
        # check if real_data_list has the correct length
        if len(real_data_list) != len(self.likelihood_names) - 1:
            raise ValueError(
                f"real_data_list must have length {len(self.likelihood_names) - 1} "
                f"according to the number of likelihood terms in the model, "
                f"but has length {len(real_data_list)}."
            )
        # check if the dtypes of the real data match the dtypes of the data generators
        expected_dtypes = [np.dtype(gen.dtype) for gen in self.data_generators]
        if any([r_d.dtype != e_d for r_d, e_d in zip(real_data_list, expected_dtypes)]):
            raise ValueError(
                "The dtypes of the real data do not match the dtypes of the data generators."
            )
        # set ancillary_measurements to nominal values
        _ancillary = self.parameters.with_uncertainty.nominal_values
        if None in _ancillary.values():
            raise ValueError(
                "The nominal values of the ancillary measurements are not set. "
                "Please provide nominal values for all ancillary measurements."
            )
        ancillary = dict_to_structured_array(_ancillary)
        # combine all data
        data_name_list = self.likelihood_names
        data_list = real_data_list + [ancillary]
        real_data = [dict(zip(data_name_list, data_list))]
        self.store_data(file_name, real_data, self.likelihood_names, metadata=metadata)


class CustomAncillaryLikelihood(LogAncillaryLikelihood):
    """Custom ancillary likelihood that can be used to add constraint terms for parameters of the
    likelihood.

    Attributes:
        parameters (Parameters): Parameters object containing the parameters to be constrained.
        constraint_functions (dict): Dict of constraint functions for all ancillary parameters.

    """

    def __init__(self, parameters: Parameters):
        """Initialize the CustomAncillaryLikelihood."""
        self.parameters = parameters
        # check that there are no None values in the uncertainties dict
        if set(self.parameters.uncertainties.keys()) != set(self.parameters.names):
            raise ValueError("The uncertainties dict must contain all parameters as keys.")
        parameter_list = self.parameters.names

        self.constraint_functions = self._get_constraint_functions()
        super().__init__(
            func=self.ancillary_sum,
            parameter_list=parameter_list,
            config=self.parameters.nominal_values,
        )
        self.pdf_base_config["name"] = "ancillary"

    @property
    def constraint_terms(self) -> dict:
        """Dict of all constraint terms (logpdf of constraint functions) of the ancillary
        likelihood.

        Returns:
            dict: Dict of all constraint terms function.

        """
        return {name: func.logpdf for name, func in self.constraint_functions.items()}

    def set_data(self, d: np.array):
        """Set the data of the ancillary likelihood (ancillary measurements).

        Args:
            d (numpy.array): Data of ancillary measurements, stored as numpy array.

        """
        # This results in shifted constraint terms.
        d_dict = structured_array_to_dict(d)
        if set(d_dict.keys()) != set(self.parameters.names):
            raise ValueError(
                "The data dict must contain all parameters as keys in CustomAncillaryLikelihood. "
                f"But {set(d_dict.keys())} is provided and "
                f"{set(self.parameters.names)} is expected."
            )
        self.constraint_functions = self._get_constraint_functions(**d_dict)

    def ancillary_sum(self, evaluate_at: dict) -> float:
        """Return the sum of all constraint terms.

        Args:
            evaluate_at (dict): Values of the ancillary measurements.

        Returns:
            float: Sum of all constraint terms.

        """
        evaluated_constraint_terms = [
            term(evaluate_at[name]) for name, term in self.constraint_terms.items()
        ]
        return np.sum(evaluated_constraint_terms)

    def _get_constraint_functions(self, **generate_values) -> dict:
        """Get callable constraint functions for all ancillary parameters.

        Keyword Args:
            generate_values (dict): A dictionary of parameter values.

        Returns:
            dict: Dict of constraint functions for all ancillary parameters.

        Todo:
            Implement str-type uncertainties.

        """
        central_values = self.parameters(**generate_values)
        constraint_functions = {}
        for name, uncertainty in self.parameters.uncertainties.items():
            if isinstance(uncertainty, (float, int)):
                param = self.parameters[name]
                if param.relative_uncertainty:
                    if param.nominal_value is None:
                        raise ValueError(
                            f"Relative uncertainty of parameter {name} is set to {uncertainty} "
                            "but nominal value is None. "
                            "Please provide a nominal value."
                        )
                    if param.nominal_value == 0:
                        warnings.warn(
                            f"Relative uncertainty of parameter {name} is set to {uncertainty} "
                            "but nominal value is 0. "
                            "This will result in a relative uncertainty of 0."
                        )
                    uncertainty *= param.nominal_value
                func = stats.norm(central_values[name], uncertainty)
            elif hasattr(uncertainty, "logpdf") and hasattr(uncertainty, "rvs"):
                warnings.warn(
                    f"Uncertainty of {name} is a string-based uncertainty. "
                    "It is frozen and its argument(s) cannot be changed as ancillary measurement."
                )
                func = uncertainty
            else:
                raise NotImplementedError(
                    f"Uncertainty {uncertainty} is not understandable. "
                    "Only float, int, and scipy.stats distributions are supported."
                )
            constraint_functions[name] = func
        return constraint_functions
