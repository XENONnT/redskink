import os
import time
import inspect
import tempfile
import datetime


from typing import Any, Dict, List, Literal, Optional
from utilix import batchq

from alea.submitter import Submitter


BATCHQ_DEFAULT_ARGUMENTS = {
    "hours": 1,  # in the unit of hours
    "mem_per_cpu": 2000,  # in the unit of Mb
}


class SubmitterSlurm(Submitter):
    """Submitter for slurm cluster,

    using utilix.batchq.submit_job. The default batchq arguments are
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
        self.batchq_arguments = {**BATCHQ_DEFAULT_ARGUMENTS, **self.slurm_configurations}
        super().__init__(*args, **kwargs)

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
        self.submit_job(job, jobname=jobname, log=log, **{**self.batchq_arguments, **kwargs})

    def submit_job(
        self,
        jobstring: str,
        log: str = "job.log",
        qos: str = "xenon1t",
        account: Optional[str] = None,
        jobname: str = "somejob",
        sbatch_file: Optional[str] = None,
        dry_run: bool = False,
        mem_per_cpu: int = 1000,
        cpus_per_task: int = 1,
        hours: Optional[float] = None,
        node: Optional[str] = None,
        exclude_nodes: Optional[str] = None,
        dependency: Optional[str] = None,
        verbose: bool = False,
        partition=None,
        container=None,
        bind=None,
        bypass_validation: Optional[List[str]] = [],
        constraint: Optional[str] = None,
    ) -> None:
        """Submit a job to the SLURM queue.
        adapted from https://github.com/XENONnT/utilix/blob/master/utilix/batchq.py

        Args:
            jobstring (str): The command to execute.
            log (str): Where to store the log file of the job. Default is "job.log".
            qos (str): QOS to submit the job to. Default is "xenon1t".
            account (str): Account to submit the job to. Default is "pi-lgrandi".
            jobname (str): How to name this job. Default is "somejob".
            sbatch_file (Optional[str]): Deprecated. Default is None.
            dry_run (bool): Only print how the job looks like, without submitting. Default is False.
            mem_per_cpu (int): MB requested for job. Default is 1000.
            cpus_per_task (int): CPUs requested for job. Default is 1.
            hours (Optional[float]): Max hours of a job. Default is None.
            node (Optional[str]): Define a certain node to submit your job. Default is None.
            dependency (Optional[str]):
                Provide list of job ids to wait for before running this job. Default is None.
            verbose (bool): Print the sbatch command before submitting. Default is False.
            bypass_validation (List[str]): List of parameters to bypass validation for.
                Default is None.


        """
        if partition is not None or container is not None or bind is not None:
            print(partition, container, bind)
            raise NotImplementedError(
                "General SLURM submission does not implement partition, container, bind != None"
            )

        TMPDIR = os.environ["HOME"] + "/tmp/"
        os.makedirs(TMPDIR, exist_ok=True)

        slurm_params: Dict[str, Any] = {
            "job_name": jobname,
            "output": log,
            "qos": qos,
            "error": log,
            "mem_per_cpu": mem_per_cpu,
            "cpus_per_task": cpus_per_task,
        }

        # Conditionally add optional parameters if they are not None
        if hours is not None:
            slurm_params["time"] = datetime.timedelta(hours=hours)
        if account is not None:
            slurm_params["account"] = account
        if constraint is not None:
            slurm_params["constraint"] = constraint

        # Create the Slurm instance with the conditional arguments
        slurm = batchq.Slurm(**slurm_params)

        file_descriptor, exec_file = tempfile.mkstemp(suffix=".sh", dir=TMPDIR)
        batchq._make_executable(exec_file)
        os.write(file_descriptor, bytes("#!/bin/bash\n" + jobstring, "utf-8"))

        jobstring = f"source {exec_file}"
        slurm.add_cmd(jobstring)
        print("SLURM is ", slurm)

        # Handle dry run scenario
        if verbose or dry_run:
            print(f"Generated slurm script:\n{slurm.script()}")

        if dry_run:
            return
        # Submit the job

        try:
            job_id = slurm.sbatch(shell="/bin/bash")
            if job_id:
                print(f"Job submitted successfully. Job ID: {job_id}")
                print(f"Your log is located at: {log}")
            else:
                print("Job submission failed.")
        except Exception as e:
            print(f"An error occurred while submitting the job: {str(e)}")

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
