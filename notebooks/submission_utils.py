import os
import sys
import tempfile
import subprocess
import argparse
from zipfile import ZipFile
import pandas as pd
import traceback
import shutil
from datetime import datetime

from utils.zip_for_codalab import zip_for_codalab
from utils import input_data_check_dir, problem_dir, score_dir, ingestion_program_cmd, scoring_program_cmd

DEFAULT_MODEL_DIR = 'example_submission/submission'

INFO_ZIP_CREATE = """
INFO: Basic check and creation of the zip file for the folder {}
"""
INFO_UNZIP = """
INFO: Checking the zip file can be unzipped.
"""
INFO_CONTENT = """
INFO: Checking content is valid
"""

INFO_META = """
INFO: metadata found.
"""

INFO_RUNNING = """
INFO: This might take a while..
It will evaluate your agent on a whole lot of scenarios
(24 scenarios, with similar number of timesteps than the validation set)
"""

INFO_RUN_SUCCESS = """
INFO: Your agent could be run correctly. 
You can now check its performance
"""

INFO_RESULT = """
INFO: Check if the results can be read back
"""

INFO_SCORE = """
Your scores are :
(remember these score are not at all an indication of \
what will be used in codalab, as the data it is tested \
on are really different):"
"""

ERR_META = """
ERROR: Submission invalid
There is no file "metadata" in the zip file you provided:
{}
Did you zip it with using "zip_for_codalab" ?
"""

ERR_RUNNING = """
ERROR: Your agent could not be run. 
It will probably fail on codalab.
Here is the information we have:
"""
my_env = os.environ

EXTRA_INFO_GIF_HTML = """
------------------------------------
         Extra Informations         
------------------------------------
Don't hesitate to have a look at:
        {0}/results.html
        {0}/*.gif
To have high level information about your agent.
"""


def prepare_submission(agent_name):
    result_path = f'../results/{agent_name}'
    agent_path = f'../src/agents/{agent_name}'
    submission_path = f'../submissions/{agent_name}/sub_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}'

    if not os.path.exists(submission_path):
        os.makedirs(f"{submission_path}/{agent_name}")

    for f in os.listdir(result_path):
        shutil.copyfile(f"{result_path}/{f}", f"{submission_path}/{agent_name}/{f}")

    for f in os.listdir(agent_path):
        shutil.copyfile(f"{agent_path}/{f}", f"{submission_path}/{f}")
    
    return submission_path

def test_submission(model_dir):
    print(INFO_ZIP_CREATE.format(model_dir))
    archive_path = zip_for_codalab(os.path.join(os.path.abspath(model_dir)))

    print(INFO_UNZIP)
    tmp_dir = tempfile.TemporaryDirectory()
    sys.path.append(tmp_dir.name)
    with ZipFile(archive_path, 'r') as zipObj:
        # Extract all the contents of zip file in different directory
        zipObj.extractall(tmp_dir.name)

    print(INFO_CONTENT)
    if not os.path.exists(os.path.join(tmp_dir.name, "metadata")):
        raise RuntimeError(ERR_META.format(archive_path))
    else:
        print(INFO_META)

    print(INFO_RUNNING)

    if model_dir.endswith("/"):
        model_dir = model_dir[:-1]
    output_submission = f"{model_dir}_res"
    if not os.path.exists(output_submission):
        os.mkdir(output_submission)
    cmd_ = ingestion_program_cmd(submission_program=tmp_dir.name, output_submission=output_submission)
    li_cmd = cmd_.split()
    res_ing = subprocess.run(
        li_cmd,
        stdout=subprocess.PIPE
    )
    if res_ing.returncode != 0:
        print("--------")
        print(ERR_RUNNING)
        print(res_ing.stdout.decode('utf-8'))
        if res_ing.stderr is not None:
            print("----------")
            print("Error message:")
            print(res_ing.stderr.decode('utf-8'))
        print()
        print()
        print("You can run \n\"{}\"\n for more debug information".format(" ".join(li_cmd)))
        raise RuntimeError("INVALID SUBMISSION")
    else:
        print(INFO_RUN_SUCCESS)

    print(INFO_RESULT)
    input_scoring = output_submission
    output_scoring = output_submission
    cmd_ = scoring_program_cmd(input_scoring=input_scoring,
                               output_scoring=output_scoring)
    li_cmd = cmd_.split()
    res_sc = subprocess.run(
        li_cmd,
        stdout=subprocess.PIPE
    )

    if res_sc.returncode != 0:
        print(ERR_RUNNING)
        print(res_sc.stdout.decode('utf-8', 'ignore'))
        raise RuntimeError("INVALID SUBMISSION")

    with open(os.path.join(output_scoring, "scores.txt"), "r") as f:
        scores = f.readlines()
    scores = [el.rstrip().lstrip().split(":") for el in scores]
    print(INFO_SCORE)
    res = pd.DataFrame(scores)
    print(res)

    print(EXTRA_INFO_GIF_HTML.format(output_submission))
    return res


def cli():
    parser = argparse.ArgumentParser(description="Zip and check codalab.")
    parser.add_argument("--model_dir", default=DEFAULT_MODEL_DIR,
                        help="Path of the DeepQSimple you want to submit.")
    return parser.parse_args()


if __name__ == "__main__":
    args = cli()
    try:
        test_submission(args.model_dir)
    except Exception as exc_:

        print("------------------------------------")
        print("        Detailed error Logs         ")
        print("------------------------------------")
        # print standard python debug
        print("ERROR: ingestion program failed with error: \n{}".format(exc_))
        print("Traceback is:")
        traceback.print_exc(file=sys.stdout)
        print("------------------------------------")
        print("      End Detailed error Logs       ")
        print("------------------------------------")

        # attempt to help with debug
        res_debug = subprocess.run(
            [
                sys.executable,
                os.path.join(os.path.split(__file__)[0], "get_info_import.py"),
                "--model_dir", args.model_dir
            ],
            stdout=subprocess.PIPE)
        if res_debug.returncode == 0:
            print()
            print()
            print("------------------------------------------------")
            print("      Automatic Help for DEBUGGING IMPORTS      ")
            print("------------------------------------------------")
            print(res_debug.stdout.decode('utf-8', 'ignore'))
            print("------------------------------------------------")
            print("    END Automatic Help for DEBUGGING IMPORTS    ")
            print("------------------------------------------------")

        print()
        print()
        print("------------------------------------")
        print("             Need Help              ")
        print("------------------------------------")
        print("Still have trouble with your submission ?")
        print("You can come ask us on our discord at:")
        print("\t\t\t https://discord.gg/cYsYrPT   \t\t\t")
        print("Don't forget to include the complete \"Detailed error Logs\" :-)")
        print("------------------------------------")
        print("         End  Need Help             ")
        print("------------------------------------")
