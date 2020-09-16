import argparse

from beu_l2rpn.utilities.submission.submission_utils import prepare_submission, test_submission

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare submission")
    parser.add_argument("--chk_point", required=True)
    args = parser.parse_args()

    submission_path = prepare_submission(args.chk_point)
    test_submission(submission_path)
