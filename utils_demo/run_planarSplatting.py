import os
from utils.misc_util import fix_seeds, get_class

def run_planarSplatting(data, conf):
    # fix_seeds()

    exps_folder_name = conf.get_string('train.exps_folder_name')
    tag = conf.get_string('dataset.tag')
    runner = get_class(conf.get_string('train.train_runner_class'))(
                                    conf=conf,
                                    batch_size=1,
                                    exps_folder_name=exps_folder_name,
                                    is_continue=False,
                                    timestamp='latest',
                                    checkpoint='latest',
                                    do_vis=False,
                                    tag=tag,
                                    data=data,
                                    )
    runner.run()