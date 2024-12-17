import argparse
import os,sys
import numpy as np
from em_util.io import *

from task import *
sys.path.append('../')
from util import *

if __name__ == "__main__":
    parser = get_parser()    
    args = parser.parse_args()
    if args.neuron != '':
        args.neuron = [int(x) for x in args.neuron.split(',')] 
    conf = read_yml(f'param_{args.cluster}.yml')
    

    f_seg = os.path.join(conf['SEG']['ROOT_PATH'], conf['SEG']['SEG_PATH'])
    f_box = os.path.join(conf['SEG']['ROOT_PATH'], conf['SEG']['SEG_BBOX_PATH'])
    f_zran_p = os.path.join(conf['SEG']['ROOT_PATH'], conf['SEG']['SEG_ZRAN_PATH'], 'zran_%d_%d.h5')
    f_zran = os.path.join(conf['SEG']['ROOT_PATH'], conf['SEG']['SEG_ZRAN_PATH'], 'zran.h5')
    f_neuron_tile = os.path.join(conf['SEG']['ROOT_PATH'], conf['SEG']['NEURON_TILE_PATH'])
    f_neuron_box = os.path.join(conf['SEG']['ROOT_PATH'], conf['SEG']['NEURON_BBOX_PATH'])
    
    f_mito_tile = os.path.join(conf['MITO']['ROOT_PATH'], conf['MITO']['UNET_PATH'])
    f_mito_ts = os.path.join(conf['MITO']['ROOT_PATH'], conf['MITO']['TENSORSTORE_PATH'])
    f_mito_pred = lambda arr: f'{f_mito_tile}/{arr[4]:04d}/{arr[2]}-{arr[0]}.h5'
    f_mito_ws = lambda arr: f_mito_pred(arr)[:-3] + '_ws.h5'
    

    if args.task == 'slurm': 
        # run in parallel
        # python main.py -t slurm -e imu -s "-t bbox -jn 10"
        cmd = f'\ncd {conf["CLUSTER"]["REPO_PATH"]}'
        cmd += f'\n{conf["CLUSTER"]["CONDA"]}{args.env}/bin/python main.py "{args.cmd}" -ji %d -jn %d'  
        
        cmd_task = cmd.split(' ')
        output_file = os.path.join(conf['CLUSTER']['ROOT_PATH'], conf['CLUSTER']['SLURM_PATH'], cmd_task[1], 'slurm')
        mkdir(output_file, 'parent')
        write_slurm_all(cmd, output_file, args.job_num, args.partition, 1, args.num_gpu, args.memory, args.run_time)
    else:
        # execute each task
        if args.task == 'seg-bbox':
            # python run_slurm
            mkdir(f_box, 'parent')
            seg_bbox_p(f_seg, f_box, args.job_id, args.job_num) 
        elif args.task == 'seg-zran_p':
            mkdir(f_zran_p, 'parent')        
            if not os.path.exists(f_zran_p %(args.job_id, args.job_num)):
                out = seg_zran_p(f_box, args.job_id, args.job_num)
                write_h5(f_zran_p %(args.job_id, args.job_num), out)
        elif args.task == 'seg-zran':
            if not os.path.exists(f_zran):
                out_id, out_zran = seg_zran_merge(f_zran_p, args.job_num)
                write_h5(f_zran, [out_id, out_zran], ['id','zran'])
        elif args.task == 'neuron-tile': # id -> bbox and tiles
            mkdir(f_neuron_tile, 'parent')
            zid, zran = read_h5(f_zran,['id','zran'])         
            neurons = args.neuron.split(',')
            for neuron in neurons:
                out_bb, out_tile_bbox = neuron_to_tile(neuron, zid, zran, f_box, f_seg) 
                np.savetxt(f_neuron_tile % neuron, out_tile_bbox, '%d')
                np.savetxt(f_neuron_box % neuron, out_bb, '%d')
        
        elif args.task == 'mito-folder': # creat folder
            mkdir(f_mito_tile)
            for z in range(num_tile[0]): 
                mkdir(f'{f_mito_tile}/%04d/'%(z*mito_tile_size[0]))
        elif args.task == 'mito-ts': # pkl for ts
            mkdir(f_mito_ts, 'parent')
            ts_dict= {
                'driver': 'neuroglancer_precomputed',
                'kvstore': {'driver': 'gcs', 'bucket': 'h01-release'},
                'path': 'data/20210601/4nm_raw',
                'scale_metadata': {'resolution': [8, 8, 33]}}            
            write_pkl(f_mito_ts, ts_dict)        
        elif args.task == 'mito-check': # pkl for ts
            # python main.py -t mito-check -n 36750893213
            for neuron in args.neuron:
                tile = np.loadtxt(f_neuron_tile % neuron).astype(int)
                for arr in tile:
                    fn = f_mito_pred(arr)
                    if not os.path.exists(fn):
                        print(f'{neuron} missing: {arr}')
        elif args.task == 'mito-watershed': # decode prediction into instance
            # python main.py -t mito-watershed -n 36750893213
            for neuron in args.neuron:
                tile = np.loadtxt(f_neuron_tile % neuron).astype(int)
                for arr in tile[args.job_id::args.job_num]:
                    fn = f_mito_pred(arr)
                    fout = f_mito_ws(arr)
                    if not os.path.exists(fout) and os.path.exists(fn):
                        out = bc_watershed(read_h5(fn), thres1=0.85, thres2=0.6, thres3=0.8, thres_small=0, seed_thres=0)
                        write_h5(fout, out)