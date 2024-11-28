import os
import time
import inspect
import socket
from utilix import batchq

from alea.submitter import Submitter


class SubmitterSlurm(Submitter):
    """Submitter for slurm cluster, using utilix.batchq.submit_job. The default batchq arguments are
    defined in BATCHQ_DEFAULT_ARGUMENTS. You can also overwrite them by passing them inside
    configuration file.

    Keyword Args:
        slurm_configurations (dict): The configurations for utilix.batchq.submit_job.
            There can be template_path inside it, indicating the path to the template.

    """

    max_jobs = 100

    def __init__(self, *args, **kwargs):
        """Initialize the SubmitterSlurm class."""
        self.name = self.__class__.__name__
        self.slurm_configurations = kwargs.get("slurm_configurations", {})
        self.template_path = self.slurm_configurations.pop("template_path", None)
        self.combine_n_jobs = self.slurm_configurations.pop("combine_n_jobs", 1)

        self._eval_partition()
        # suggested default arguments for utilix.batchq.submit_job
        # for XENONnT collaboration
        batchq_default_arguments = {
            "hours": 1,  # in the unit of hours
            "mem_per_cpu": 2000,  # in the unit of Mb
            "container": "xenonnt-development.simg",
            "partition": self.partition,
            "qos": self.qos,
            "cpus_per_task": 1,
            "exclude_nodes": self.exclude_nodes,
        }

        self.batchq_arguments = {
            **batchq_default_arguments,
            **self.slurm_configurations,
        }
        self._check_batchq_arguments()
        super().__init__(*args, **kwargs)

    def _eval_partition(self):
        # automatically set default partition based on hostname
        hostname = socket.gethostname()
        if "midway3" in hostname:
            self.partition = "lgrandi"
            self.qos = "lgrandi"
            self.exclude_nodes = "null"
        elif "midway2" in hostname or "dali" in hostname:
            self.partition = "xenon1t"
            self.qos = "xenon1t"
            self.exclude_nodes = "dali[028-030],midway2-0048"
        else:
            raise ValueError(
                f"Unknown hostname: {hostname},"
                f"please specify partition, qos, and exclude_nodes explicitly."
            )

    def _submit(self, job, **kwargs):
        """Submits job to batch queue which actually runs the analysis.

        Args:
            job (str): The job script to be submitted.

        Keyword Args:
            jobname (str): The name of the job.
            log (str): The path to the log file.

        """
        jobname = kwargs.pop("jobname", None)
        if jobname is None:
            jobname = self.name

        log = kwargs.pop("log", None)
        if log is None:
            log = os.path.join(self.outputfolder, f"{jobname.lower()}.log")

        kwargs_to_pop = []
        for key, val in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, val)
                kwargs_to_pop.append(key)
        for kw in kwargs_to_pop:
            kwargs.pop(kw)

        self.logging.debug(f"Submitting the following job: '{job}'")
        batchq.submit_job(job, jobname=jobname, log=log, **{**self.batchq_arguments, **kwargs})

    def _check_batchq_arguments(self):
        """Check if the self.batchq_arguments are valid."""
        args = set(inspect.signature(batchq.submit_job).parameters)
        if not set(self.batchq_arguments).issubset(args):
            raise ValueError(
                f"The following arguments are not supported of utilix.batchq.submit_job: "
                f"{set(self.batchq_arguments) - set(args)}."
            )

    def submit(self, **kwargs):
        """Submits job to batch queue which actually runs the analysis. Overwrite the
        BATCHQ_DEFAULT_ARGUMENTS by configuration file. If debug is True, only submit the first job.

        Keyword Args:
            jobname (str): The name of the job.

        """
        _jobname = kwargs.pop("jobname", self.name.lower())
        batchq_kwargs = {}
        for job, (script, last_output_filename) in enumerate(self.combined_tickets_generator()):
            if self.debug:
                print(script)
                if job > 0:
                    break
            while batchq.count_jobs(_jobname) > self.max_jobs:
                self.logging.info("Too many jobs. Sleeping for 30s.")
                time.sleep(30)
            batchq_kwargs["jobname"] = f"{_jobname}_{job:03d}"
            if last_output_filename is not None:
                batchq_kwargs["log"] = os.path.join(
                    self.outputfolder, f"{last_output_filename}.log"
                )
            self.logging.debug(f"Call '_submit' with job: {job} and kwargs: {batchq_kwargs}.")
            self._submit(script, **batchq_kwargs)
