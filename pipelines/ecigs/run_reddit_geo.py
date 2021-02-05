import json
import gzip
import glob
import os

from config import REDDIT_PROCESSED_DATA_DIR, SMGEO_DIR, SMGEO_OUT_FN, SMGEO_VIRTUAL_ENV
from utils import ensure_folder


def get_users(reddit_type):
    fn_list = glob.glob(os.path.join(REDDIT_PROCESSED_DATA_DIR, 'reddit_{}'.format(reddit_type),
                                     'output_reddit_keywords', '*.json.gz'))

    print("Number of files to process:", len(fn_list))
    users = set()
    for fn in fn_list:
        with gzip.open(fn, 'r') as inf:
            for line in inf:
                post = json.loads(line.strip().decode())
                assert len(post['annotations']['keywords'])
                users.add(post['author'])

    return users


def write_all_users():
    all_users = set()
    for reddit_type in ['submission', 'comment']:
        all_users.update(get_users(reddit_type))

    print("Number of users: {}".format(len(all_users)))

    users_fn = os.path.join(REDDIT_PROCESSED_DATA_DIR, 'reddit_authors.txt')
    with open(users_fn, 'w') as outf:
        for _user in all_users:
            outf.write("{}\n".format(_user))

    return users_fn


if __name__ == '__main__':
    ensure_folder(SMGEO_OUT_FN)
    users_fn = write_all_users()
    cmd = 'bash smgeo_infer.sh {input} {output} {smgeo_dir} {smgeo_venv}'
    os.system(cmd.format(input=users_fn, output=SMGEO_OUT_FN, smgeo_dir=SMGEO_DIR, smgeo_venv=SMGEO_VIRTUAL_ENV))
