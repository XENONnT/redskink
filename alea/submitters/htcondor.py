import os
import getpass
import tarfile
import shlex
import json
import tempfile
import time
import threading
import subprocess
from datetime import datetime
import logging
from pathlib import Path
from Pegasus.api import (
    Arch,
    Operation,
    Namespace,
    Workflow,
    File,
    Directory,
    FileServer,
    Job,
    Site,
    SiteCatalog,
    Transformation,
    TransformationCatalog,
    ReplicaCatalog,
)
from alea.submitter import Submitter
from alea.utils import load_yaml, dump_yaml


DEFAULT_IMAGE = "/cvmfs/singularity.opensciencegrid.org/xenonnt/base-environment:latest"
WORK_DIR = f"/scratch/{getpass.getuser()}/workflows"
TOP_DIR = Path(__file__).resolve().parents[2]


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


class SubmitterHTCondor(Submitter):
    """Submitter for htcondor cluster."""

    def __init__(self, *args, **kwargs):
        # General start
        self.htcondor_configurations = kwargs.get("htcondor_configurations", {})
        self.singularity_image = self.htcondor_configurations.pop(
            "singularity_image", DEFAULT_IMAGE
        )
        self.top_dir = TOP_DIR
        self.work_dir = WORK_DIR
        self.template_path = self.htcondor_configurations.pop("template_path", None)
        self.combine_n_outputs = self.htcondor_configurations.pop("combine_n_outputs", 100)

        # A flag to check if limit_threshold is added to the rc
        self.added_limit_threshold = False

        # Cluster size for toymc jobs
        self.cluster_size = self.htcondor_configurations.pop("cluster_size", 1)

        # Resources configurations
        self.request_cpus = self.htcondor_configurations.pop("request_cpus", 1)
        self.request_memory = self.htcondor_configurations.pop("request_memory", 2000)
        self.request_disk = self.htcondor_configurations.pop("request_disk", 2000000)
        self.combine_disk = self.htcondor_configurations.pop("combine_disk", 20000000)

        # Dagman configurations
        self.dagman_maxidle = self.htcondor_configurations.pop("dagman_maxidle", 100000)
        self.dagman_retry = self.htcondor_configurations.pop("dagman_retry", 2)
        self.dagman_maxjobs = self.htcondor_configurations.pop("dagman_maxjobs", 100000)

        super().__init__(*args, **kwargs)

        # Job input configurations
        self.config_file_path = os.path.abspath(self.config_file_path)

        # User can provide a name for the workflow, otherwise it will be the current time
        self._setup_workflow_id()
        # Pegasus workflow directory
        self.generated_dir = os.path.join(self.work_dir, "generated", self.workflow_id)
        self.runs_dir = os.path.join(self.work_dir, "runs", self.workflow_id)
        self.outputs_dir = os.path.join(self.work_dir, "outputs", self.workflow_id)

    @property
    def template_tarball(self):
        return os.path.join(self.generated_dir, "templates.tar.gz")

    @property
    def workflow(self):
        return os.path.join(self.generated_dir, "workflow.yml")

    @property
    def pegasus_config(self):
        """Pegasus configurations."""
        pconfig = {}
        pconfig["pegasus.metrics.app"] = "XENON"
        pconfig["pegasus.data.configuration"] = "nonsharedfs"
        pconfig["dagman.retry"] = self.dagman_retry
        pconfig["dagman.maxidle"] = self.dagman_maxidle
        pconfig["dagman.maxjobs"] = self.dagman_maxjobs
        pconfig["pegasus.transfer.threads"] = 4

        # Help Pegasus developers by sharing performance data (optional)
        pconfig["pegasus.monitord.encoding"] = "json"
        pconfig["pegasus.catalog.workflow.amqp.url"] = (
            "amqp://friend:donatedata@msgs.pegasus.isi.edu:5672/prod/workflows"
        )
        return pconfig

    @property
    def requirements(self):
        """Make the requirements for the job."""
        # Minimal requirements on singularity/cvmfs/ports/microarchitecture
        _requirements = (
            "HAS_SINGULARITY && HAS_CVMFS_xenon_opensciencegrid_org"
            + " && PORT_2880 && PORT_8000 && PORT_27017"
            + ' && (Microarch >= "x86_64-v3")'
        )

        # If in debug mode, use the MWT2 site because we own it
        if self.debug:
            _requirements += ' && GLIDEIN_ResourceName == "MWT2" '

        return _requirements

    def _get_file_name(self, file_path):
        """Get the filename from the file path."""
        return os.path.basename(file_path)

    def _validate_x509_proxy(self, min_valid_hours=20):
        """Ensure $X509_USER_PROXY exists and has enough time left.

        This is necessary only if you are going to use Rucio.

        """
        self.x509_user_proxy = os.getenv("X509_USER_PROXY")
        assert self.x509_user_proxy, "Please provide a valid X509_USER_PROXY environment variable."

        logger.debug("Verifying that the X509_USER_PROXY proxy has enough lifetime")
        shell = Shell("grid-proxy-info -timeleft -file %s" % (self.x509_user_proxy))
        shell.run()
        valid_hours = int(shell.get_outerr()) / 60 / 60
        if valid_hours < min_valid_hours:
            raise RuntimeError(
                "User proxy is only valid for %d hours. Minimum required is %d hours."
                % (valid_hours, min_valid_hours)
            )

    def _validate_template_path(self):
        """Validate the template path."""
        if self.template_path is None:
            raise ValueError("Please provide a template path.")
        # This path must exists locally, and it will be used to stage the input files
        if not os.path.exists(self.template_path):
            raise ValueError(f"Path {self.template_path} does not exist.")

        # Printout the template path file structure
        logger.info("Template path file structure:")
        for dirpath, dirnames, filenames in os.walk(self.template_path):
            for filename in filenames:
                logger.info(f"File: {filename} in {dirpath}")
        if self._contains_subdirectories(self.template_path):
            logger.warning(
                "The template path contains subdirectories. All templates files will be tarred."
            )

    def _tar_h5_files(self, directory, output_filename="templates.tar.gz"):
        """Tar all .h5 templates in the directory and its subdirectories into a tarball."""
        # Create a tar.gz archive
        with tarfile.open(output_filename, "w:gz") as tar:
            # Walk through the directory
            for dirpath, dirnames, filenames in os.walk(directory):
                for filename in filenames:
                    if filename.endswith(".h5"):
                        # Get the full path to the file
                        filepath = os.path.join(dirpath, filename)
                        # Add the file to the tar
                        # Specify the arcname to store relative path within the tar
                        tar.add(filepath, arcname=os.path.basename(filename))

    def _make_template_tarball(self):
        """Make tarball of the templates if not exists."""
        self._tar_h5_files(self.template_path, self.template_tarball)

    def _modify_yaml(self):
        """Modify the statistical model config file to correct the 'template_filename' fields.

        We will use the modified one to upload to OSG. This modification is necessary because the
        templates on the grid will have different path compared to the local ones, and the
        statistical model config file must reflect that.

        """
        # Output file will have the same name as input file but with '_modified' appended
        _output_file = self._get_file_name(self.statistical_model_config).replace(
            ".yaml", "_modified.yaml"
        )
        self.modified_statistical_model_config = os.path.join(self.generated_dir, _output_file)

        # Load the YAML data from the original file
        data = load_yaml(self.statistical_model_config)

        # Recursive function to update 'template_filename' fields
        def update_template_filenames(node):
            if isinstance(node, dict):
                for key, value in node.items():
                    if key == "template_filename":
                        filename = value.split("/")[-1]
                        node[key] = filename
                    else:
                        update_template_filenames(value)
            elif isinstance(node, list):
                for item in node:
                    update_template_filenames(item)

        # Update the data
        update_template_filenames(data)

        # Write the updated YAML data to the new file
        # Overwrite if the file already exists
        dump_yaml(self.modified_statistical_model_config, data)
        logger.info(
            "Modified statistical model config file "
            f"written to {self.modified_statistical_model_config}"
        )

    def _contains_subdirectories(self, directory):
        """Check if the specified directory contains any subdirectories.

        Args:
        directory (str): The path to the directory to check.

        Returns:
        bool: True if there are subdirectories inside the given directory, False otherwise.

        """
        # List all entries in the directory
        try:
            for entry in os.listdir(directory):
                # Check if the entry is a directory
                if os.path.isdir(os.path.join(directory, entry)):
                    return True
        except FileNotFoundError:
            print("The specified directory does not exist.")
            return False
        except PermissionError:
            print("Permission denied for accessing the directory.")
            return False

        # If no subdirectories are found
        return False

    def _setup_workflow_id(self):
        """Set up the workflow ID."""
        # If you have named the workflow, use that name. Otherwise, use the current time as name.
        _workflow_id = self.htcondor_configurations.pop("workflow_id", None)
        if _workflow_id:
            self.workflow_id = "-".join(
                (_workflow_id, self.computation, datetime.now().strftime("%Y%m%d%H%M"))
            )
        else:
            self.workflow_id = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    def _generate_sc(self):
        """Generates the SiteCatalog for the workflow."""
        sc = SiteCatalog()

        # Local site: this is the submit host
        logger.debug("Defining local site")
        local = Site("local")
        # Logs and pegasus output goes here. This place is called stash in OSG jargon.
        scratch_dir = Directory(
            Directory.SHARED_SCRATCH, path="{}/scratch/{}".format(self.work_dir, self.workflow_id)
        )
        scratch_dir.add_file_servers(
            FileServer(
                "file:///{}/scratch/{}".format(self.work_dir, self.workflow_id), Operation.ALL
            )
        )
        # Jobs outputs goes here, but note that it is in scratch so it only stays for short term
        # This place is called stash in OSG jargon.
        storage_dir = Directory(
            Directory.LOCAL_STORAGE, path="{}/outputs/{}".format(self.work_dir, self.workflow_id)
        )
        storage_dir.add_file_servers(
            FileServer(
                "file:///{}/outputs/{}".format(self.work_dir, self.workflow_id), Operation.ALL
            )
        )
        # Add scratch and storage directories to the local site
        local.add_directories(scratch_dir, storage_dir)
        # Add profiles to the local site
        local.add_profiles(Namespace.ENV, HOME=os.environ["HOME"])
        local.add_profiles(Namespace.ENV, GLOBUS_LOCATION="")
        local.add_profiles(
            Namespace.ENV,
            PATH=(
                "/cvmfs/xenon.opensciencegrid.org/releases/nT/development/anaconda/envs/XENONnT_development/bin:"  # noqa
                "/cvmfs/xenon.opensciencegrid.org/releases/nT/development/anaconda/condabin:/usr/bin:/bin"  # noqa
            ),
        )
        local.add_profiles(
            Namespace.ENV,
            LD_LIBRARY_PATH=(
                "/cvmfs/xenon.opensciencegrid.org/releases/nT/development/anaconda/envs/XENONnT_development/lib64:"  # noqa
                "/cvmfs/xenon.opensciencegrid.org/releases/nT/development/anaconda/envs/XENONnT_development/lib"  # noqa
            ),
        )
        local.add_profiles(Namespace.ENV, PEGASUS_SUBMITTING_USER=os.environ["USER"])
        local.add_profiles(Namespace.ENV, X509_USER_PROXY=os.environ["X509_USER_PROXY"])

        # Staging sites: for XENON it is physically at dCache in UChicago
        # You will be able to download results from there via gfal commands
        logger.debug("Defining stagging site")
        staging_davs = Site("staging-davs")
        scratch_dir = Directory(
            Directory.SHARED_SCRATCH, path="/xenon/scratch/{}".format(getpass.getuser())
        )
        scratch_dir.add_file_servers(
            FileServer(
                "gsidavs://xenon-gridftp.grid.uchicago.edu:2880/xenon/scratch/{}".format(
                    getpass.getuser()
                ),
                Operation.ALL,
            )
        )
        staging_davs.add_directories(scratch_dir)

        # Condorpool: These are the job nodes on grid
        logger.debug("Defining condorpool")
        condorpool = Site("condorpool")
        condorpool.add_profiles(Namespace.PEGASUS, style="condor")
        condorpool.add_profiles(Namespace.CONDOR, universe="vanilla")
        condorpool.add_profiles(
            Namespace.CONDOR, key="+SingularityImage", value=f'"{self.singularity_image}"'
        )
        # Ignore the site settings, since the container will set all this up inside
        condorpool.add_profiles(Namespace.ENV, OSG_LOCATION="")
        condorpool.add_profiles(Namespace.ENV, GLOBUS_LOCATION="")
        condorpool.add_profiles(Namespace.ENV, PYTHONPATH="")
        condorpool.add_profiles(Namespace.ENV, PERL5LIB="")
        condorpool.add_profiles(Namespace.ENV, LD_LIBRARY_PATH="")
        condorpool.add_profiles(Namespace.ENV, PEGASUS_SUBMITTING_USER=os.environ["USER"])
        condorpool.add_profiles(
            Namespace.CONDOR, key="x509userproxy", value=os.environ["X509_USER_PROXY"]
        )

        # Add the sites to the SiteCatalog
        sc.add_sites(local, staging_davs, condorpool)
        return sc

    def _generate_tc(self):
        """Generates the TransformationCatalog for the workflow.

        Every executable that is used in the workflow should be here.

        """
        # Wrappers that runs alea_run_toymc
        run_toymc_wrapper = Transformation(
            name="run_toymc_wrapper",
            site="local",
            pfn=self.top_dir / "alea/submitters/run_toymc_wrapper.sh",
            is_stageable=True,
            arch=Arch.X86_64,
        ).add_pegasus_profile(clusters_size=self.cluster_size)

        # Wrappers that collect outputs
        combine = Transformation(
            name="combine",
            site="local",
            pfn=self.top_dir / "alea/submitters/combine.sh",
            is_stageable=True,
            arch=Arch.X86_64,
        )

        tc = TransformationCatalog()
        tc.add_transformations(run_toymc_wrapper, combine)

        return tc

    def _generate_rc(self):
        """Generate the ReplicaCatalog for the workflow.

        1. The input files for the job, which are the templates in tarball,
            the yaml files and alea_run_toymc.
        2. The output files for the job, which are the toydata and the output files.
        Since the outputs are not known in advance, we will add them in the job definition.

        """
        rc = ReplicaCatalog()

        # Add the templates
        self.f_template_tarball = File(self._get_file_name(self.template_tarball))
        rc.add_replica(
            "local",
            self._get_file_name(self.template_tarball),
            "file://{}".format(self.template_tarball),
        )
        # Add the yaml files
        self.f_running_configuration = File(self._get_file_name(self.config_file_path))
        rc.add_replica(
            "local",
            self._get_file_name(self.config_file_path),
            "file://{}".format(self.config_file_path),
        )
        self.f_statistical_model_config = File(
            self._get_file_name(self.modified_statistical_model_config)
        )
        rc.add_replica(
            "local",
            self._get_file_name(self.modified_statistical_model_config),
            "file://{}".format(self.modified_statistical_model_config),
        )
        # Add run_toymc_wrapper
        self.f_run_toymc_wrapper = File("run_toymc_wrapper.sh")
        rc.add_replica(
            "local",
            "run_toymc_wrapper.sh",
            "file://{}".format(self.top_dir / "alea/submitters/run_toymc_wrapper.sh"),
        )
        # Add alea_run_toymc
        self.f_alea_run_toymc = File("alea_run_toymc")
        rc.add_replica(
            "local",
            "alea_run_toymc",
            "file://{}".format(self.top_dir / "bin/alea_run_toymc"),
        )
        # Add combine executable
        self.f_combine = File("combine.sh")
        rc.add_replica(
            "local",
            "combine.sh",
            "file://{}".format(self.top_dir / "alea/submitters/combine.sh"),
        )

        return rc

    def _initialize_job(
        self,
        name="run_toymc_wrapper",
        run_on_submit_node=False,
        cores=1,
        memory=1_700,
        disk=1_000_000,
    ):
        """Initilize a Pegasus job, also sets resource profiles.

        Memory in unit of MB, and disk in unit of MB.

        """
        job = Job(name)
        job.add_profiles(Namespace.CONDOR, "request_cpus", f"{cores}")

        if run_on_submit_node:
            job.add_selector_profile(execution_site="local")
            # no other attributes on a local job
            return job

        # Set memory and disk requirements
        # If the job fails, retry with more memory and disk
        memory_str = (
            "ifthenelse(isundefined(DAGNodeRetry) || "
            f"DAGNodeRetry == 0, {memory}, (DAGNodeRetry + 1) * {memory})"
        )
        disk_str = (
            "ifthenelse(isundefined(DAGNodeRetry) || "
            f"DAGNodeRetry == 0, {disk}, (DAGNodeRetry + 1) * {disk})"
        )
        job.add_profiles(Namespace.CONDOR, "request_disk", disk_str)
        job.add_profiles(Namespace.CONDOR, "request_memory", memory_str)

        return job

    def _add_combine_job(self, combine_i):
        """Add a combine job to the workflow."""
        logger.info(f"Adding combine job {combine_i} to the workflow")
        combine_name = "combine"
        combine_job = self._initialize_job(
            name=combine_name,
            cores=self.request_cpus,
            memory=self.request_memory * 2,
            disk=self.combine_disk,
        )
        combine_job.add_profiles(Namespace.CONDOR, "requirements", self.requirements)

        # Combine job configuration: all toymc results and files will be combined into one tarball
        combine_job.add_outputs(
            File("%s-%s-combined_output.tar.gz" % (self.workflow_id, combine_i)), stage_out=True
        )
        combine_job.add_args(self.workflow_id + f"-{combine_i}")
        self.wf.add_jobs(combine_job)

        return combine_job

    def _add_limit_threshold(self):
        """Add the Neyman thresholds limit_threshold to the replica catalog."""
        self.f_limit_threshold = File(self._get_file_name(self.limit_threshold))
        self.rc.add_replica(
            "local",
            self._get_file_name(self.limit_threshold),
            "file://{}".format(self.limit_threshold),
        )
        self.added_limit_threshold = True

    def _correct_paths_args_dict(self, args_dict):
        """Correct the paths in the arguments dictionary in a hardcoding way."""
        args_dict["statistical_model_args"]["template_path"] = "templates/"

        if "limit_threshold" in args_dict["statistical_model_args"].keys():
            limit_threshold = self._get_file_name(
                args_dict["statistical_model_args"]["limit_threshold"]
            )
            args_dict["statistical_model_args"]["limit_threshold"] = limit_threshold

        args_dict["toydata_filename"] = self._get_file_name(args_dict["toydata_filename"])
        args_dict["output_filename"] = self._get_file_name(args_dict["output_filename"])
        args_dict["statistical_model_config"] = self._get_file_name(
            self.modified_statistical_model_config
        )

        return args_dict

    def _reorganize_script(self, script):
        """Extract executable and arguments from the naked scripts.

        Correct the paths on the fly.

        """
        executable = self._get_file_name(script.split()[1])
        args_dict = Submitter.runner_kwargs_from_script(shlex.split(script)[2:])

        # Add the limit_threshold to the replica catalog if not added
        if (
            not self.added_limit_threshold
            and "limit_threshold" in args_dict["statistical_model_args"].keys()
        ):
            self.limit_threshold = args_dict["statistical_model_args"]["limit_threshold"]
            self._add_limit_threshold()

        # Correct the paths in the arguments
        args_dict = self._correct_paths_args_dict(args_dict)

        return executable, args_dict

    def _generate_workflow(self, name="run_toymc_wrapper"):
        """Generate the workflow.

        1. Define catalogs
        2. Generate jobs by iterating over the path-modified tickets
        3. Add jobs to the workflow

        """
        if self.combine_n_jobs != 1:
            raise ValueError(
                f"{self.__class__.__name__} can not combine jobs "
                f"but can only combine outputs so please set {self.combine_n_jobs} to 1."
            )

        # Initialize the workflow
        self.wf = Workflow("alea_workflow")
        self.sc = self._generate_sc()
        self.tc = self._generate_tc()
        self.rc = self._generate_rc()

        # Iterate over the tickets and generate jobs
        combine_i = 0
        new_to_combine = True

        # Generate jobstring and output names from tickets generator
        for job_id, (script, _) in enumerate(self.combined_tickets_generator()):
            # If the number of jobs to combine is reached, add a new combine job
            if new_to_combine:
                combine_job = self._add_combine_job(combine_i)

            # Reorganize the script to get the executable and arguments,
            # in which the paths are corrected
            executable, args_dict = self._reorganize_script(script)
            if not (args_dict["toydata_mode"] in ["generate_and_store", "generate"]):
                raise NotImplementedError(
                    "Only generate_and_store toydata mode is supported on OSG."
                )

            logger.info(f"Adding job {job_id} to the workflow")
            logger.debug(f"Naked Script: {script}")
            logger.debug(f"Output: {args_dict['output_filename']}")
            logger.debug(f"Executable: {executable}")
            logger.debug(f"Toydata: {args_dict['toydata_filename']}")
            logger.debug(f"Arguments: {args_dict}")

            # Create a job with base requirements
            job = self._initialize_job(
                name=name,
                cores=self.request_cpus,
                memory=self.request_memory,
                disk=self.request_disk,
            )
            job.add_profiles(Namespace.CONDOR, "requirements", self.requirements)

            # Add the inputs and outputs
            job.add_inputs(
                self.f_template_tarball,
                self.f_running_configuration,
                self.f_statistical_model_config,
                self.f_run_toymc_wrapper,
                self.f_alea_run_toymc,
                self.f_combine,
            )
            if self.added_limit_threshold:
                job.add_inputs(self.f_limit_threshold)

            job.add_outputs(File(args_dict["output_filename"]), stage_out=False)
            combine_job.add_inputs(File(args_dict["output_filename"]))

            # Only add the toydata file if instructed to do so
            if args_dict["toydata_mode"] == "generate_and_store":
                job.add_outputs(File(args_dict["toydata_filename"]), stage_out=False)
                combine_job.add_inputs(File(args_dict["toydata_filename"]))

            # Add the arguments into the job
            # Using escaped argument to avoid the shell syntax error
            def _extract_all_to_tuple(d):
                return tuple(
                    f"{json.dumps(str(d[key])).replace(' ', '')}".replace("'", '\\"')
                    for key in d.keys()
                )

            args_tuple = _extract_all_to_tuple(args_dict)
            job.add_args(*args_tuple)

            # Add the job to the workflow
            self.wf.add_jobs(job)

            # If the number of jobs to combine is reached, add a new combine job
            if (job_id + 1) % self.combine_n_outputs == 0:
                new_to_combine = True
                combine_i += 1
            else:
                new_to_combine = False

        # Finalize the workflow
        self.wf.add_replica_catalog(self.rc)
        self.wf.add_transformation_catalog(self.tc)
        self.wf.add_site_catalog(self.sc)
        self.wf.write(file=self.workflow)

    def _us_sites_only(self):
        raise NotImplementedError

    def _exclude_sites(self):
        raise NotImplementedError

    def _this_site_only(self):
        raise NotImplementedError

    def _plan_and_submit(self):
        """Plan and submit the workflow."""
        self.wf.plan(
            submit=not self.debug,
            cluster=["horizontal"],
            cleanup="none",
            sites=["condorpool"],
            verbose=3 if self.debug else 0,
            staging_sites={"condorpool": "staging-davs"},
            output_sites=["local"],
            dir=os.path.dirname(self.runs_dir),
            relative_dir=self.workflow_id,
            **self.pegasus_config,
        )

        print(f"Worfklow written to \n\n\t{self.runs_dir}\n\n")

    def _warn_outputfolder(self):
        """Warn users about the outputfolder in running config won't be really used."""
        logger.warning(
            "The outputfolder in the running configuration %s won't be used in this submission."
            % (self.outputfolder)
        )
        logger.warning("Instead, you should find your outputs at %s" % (self.outputs_dir))

    def _check_filename_unique(self):
        """Check if all the files in the template path are unique.

        We assume two levels of the template folder.

        """
        all_files = []
        for _, _, filenames in os.walk(self.template_path):
            for filename in filenames:
                all_files.append(filename)
        if len(all_files) != len(set(all_files)):
            raise RuntimeError("All files in the template path must have unique names.")

    def submit(self, **kwargs):
        """Serve as the main function to submit the workflow."""
        if os.path.exists(self.runs_dir):
            raise RuntimeError(f"Workflow already exists at {self.runs_dir}. Exiting.")
        self._validate_x509_proxy()

        # 0o755 means read/write/execute for owner, read/execute for everyone else
        os.makedirs(self.generated_dir, 0o755, exist_ok=True)
        os.makedirs(self.runs_dir, 0o755, exist_ok=True)
        os.makedirs(self.outputs_dir, 0o755, exist_ok=True)

        # Modify the statistical model config file to correct the 'template_filename' fields
        self._modify_yaml()

        # Handling templates as part of the inputs
        self._validate_template_path()
        self._check_filename_unique()
        self._make_template_tarball()

        self._generate_workflow()
        self._plan_and_submit()
        if self.debug:
            self.wf.graph(
                output=os.path.join(self.outputs_dir, "workflow_graph.dot"), label="xform-id"
            )
            self.wf.graph(
                output=os.path.join(self.outputs_dir, "workflow_graph.svg"), label="xform-id"
            )
        self._warn_outputfolder()


class Shell(object):
    """Provides a shell callout with buffered stdout/stderr, error handling and timeout."""

    def __init__(self, cmd, timeout_secs=1 * 60 * 60, log_cmd=False, log_outerr=False):
        self._cmd = cmd
        self._timeout_secs = timeout_secs
        self._log_cmd = log_cmd
        self._log_outerr = log_outerr
        self._process = None
        self._out_file = None
        self._outerr = ""
        self._duration = 0.0

    def run(self):
        def target():
            self._process = subprocess.Popen(
                self._cmd,
                shell=True,
                stdout=self._out_file,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setpgrp,
            )
            self._process.communicate()

        if self._log_cmd:
            print(self._cmd)

        # temp file for the stdout/stderr
        self._out_file = tempfile.TemporaryFile(prefix="outsource-", suffix=".out")

        ts_start = time.time()

        thread = threading.Thread(target=target)
        thread.start()

        thread.join(self._timeout_secs)
        if thread.is_alive():
            # do our best to kill the whole process group
            try:
                kill_cmd = "kill -TERM -%d" % (os.getpgid(self._process.pid))
                kp = subprocess.Popen(kill_cmd, shell=True)
                kp.communicate()
                self._process.terminate()
            except Exception:
                pass
            thread.join()
            # log the output
            self._out_file.seek(0)
            stdout = self._out_file.read().decode("utf-8").strip()
            if self._log_outerr and len(stdout) > 0:
                print(stdout)
            self._out_file.close()
            raise RuntimeError(
                "Command timed out after %d seconds: %s" % (self._timeout_secs, self._cmd)
            )

        self._duration = time.time() - ts_start

        # log the output
        self._out_file.seek(0)
        self._outerr = self._out_file.read().decode("utf-8").strip()
        if self._log_outerr and len(self._outerr) > 0:
            print(self._outerr)
        self._out_file.close()

        if self._process.returncode != 0:
            raise RuntimeError(
                "Command exited with non-zero exit code (%d): %s\n%s"
                % (self._process.returncode, self._cmd, self._outerr)
            )

    def get_outerr(self):
        """Returns the combined stdout and stderr from the command."""
        return self._outerr

    def get_exit_code(self):
        """Returns the exit code from the process."""
        return self._process.returncode

    def get_duration(self):
        """Returns the timing of the command (seconds)"""
        return self._duration
