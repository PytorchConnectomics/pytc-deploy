import argparse
import os,sys
import numpy as np
from em_util.io import *
from em_util.seg import *

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
    f_tile = lambda arr: f'{arr[4]:04d}/{arr[2]}-{arr[0]}' 
    f_mito_pred = lambda arr: f'{f_mito_tile}/{f_tile(arr)}.h5'
    f_mito_ws = lambda arr: f'{f_mito_tile}/{f_tile(arr)}_ws.h5'
    f_mito_neuron = os.path.join(conf['MITO']['ROOT_PATH'], conf['MITO']['NEURON_MITO_PATH'])    
    f_mito_neuron_tile = lambda arr: f'{arr[4]:04d}_{arr[2]}_{arr[0]}' 
    f_mito_neuron_count = os.path.join(f_mito_neuron, 'count.h5')
    f_mito_neuron_match = os.path.join(f_mito_neuron, 'match.h5')
    f_mito_neuron_output = os.path.join(f_mito_neuron, 'mito.h5')
    

    if args.task == 'slurm': 
        # run in parallel
        # python main.py -t slurm -e imu -s "-t bbox -jn 10"
        # python main.py -t slurm -e lichtman -s "-t mito-watershed -n 36750893213 -jn 10"
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
        elif 'mito-neuron' in args.task:
            for nid, neuron in enumerate(args.neuron):
                tile = np.loadtxt(f_neuron_tile % neuron).astype(int)
                if args.task == 'mito-neuron-check': # pkl for ts
                    # python main.py -t mito-neuron-check -n 590612150
                    num = 0
                    for arr in tile:
                        #fn = f_mito_ws(arr)
                        #fn = f_mito_pred(arr)
                        fn = f'{f_mito_neuron%neuron}/{f_tile2(arr)}.h5'
                        if not os.path.exists(fn):
                            print(f'{neuron} missing: {fn}')
                            num += 1
                        else:
                            continue
                            sz = get_volume_size_h5(fn)
                            if np.abs(sz[1:] - mito_tile_size).max()!=0:
                                # print(f'wrong size: {fn}')
                                print(f'rm {fn[:-3]}*.h5')
                    print(f'{neuron} #missing={num}')
                elif args.task == 'mito-neuron-watershed': # decode prediction into instance
                    # python main.py -t mito-neuron-watershed -n 36750893213
                    for arr in tile[args.job_id::args.job_num]:
                        fn = f_mito_pred(arr)
                        fout = f_mito_ws(arr)
                        if not os.path.exists(fout) and os.path.exists(fn):
                            print(fn)
                            out = bc_watershed(read_h5(fn), thres1=0.85, thres2=0.6, thres3=0.8, thres_small=0, seed_thres=0)
                            write_h5(fout, out)
                elif args.task == 'mito-watershed-iou': # decode prediction into instance
                    # python main.py -t mito-watershed-iou -n 36750893213
                    # compare to the previous one
                    for arr in tile[args.job_id::args.job_num]:                    
                        mito_watershed_iou(f_mito_ws_func, arr_mito)
                                        
                elif args.task == 'mito-neuron-sid': # compute the seg ids within the mask
                    # python main.py -t mito-watershed-iou -n 36750893213
                    if nid == 0:
                        seg_fns = [f_seg%z for z in range(neuron_volume_size[0])]                                    
                    D0 = f_mito_neuron%neuron
                    mkdir(D0, 'all')
                    overlap_ratio = 0.6
                    for arr in tile[args.job_id::args.job_num]:                                        
                        fout = f'{D0}/{f_mito_neuron_tile(arr)}.h5'
                        if not os.path.exists(fout):
                            sid = mito_neuron_sid(f_mito_ws(arr), arr, overlap_ratio)
                            print(fout, len(sid))
                            write_h5(fout, np.array(sid))
                elif args.task == 'mito-neuron-sid-count': # compute the seg ids within the mask
                    # python main.py -t mito-neuron-sid-count -n 36750893213            
                    fout = f_mito_neuron_count%neuron
                    if not os.path.exists(fout):
                        count = np.zeros(tile.shape[0]+1, np.uint16)
                        for i,arr in enumerate(tile):
                            fn = f'{f_mito_neuron% neuron}/{f_mito_neuron_tile(arr)}.h5'
                            count[i+1] = len(read_h5(fn))
                        count = np.cumsum(count)
                        import pdb;pdb.set_trace()
                        write_h5(fout, count)
                elif args.task == 'mito-neuron-sid-iou': # connect seg id if needed
                    # python main.py -t mito-neuron-sid-iou -n 36750893213            
                    count = read_h5(f_mito_neuron_count%neuron)
                    mid = np.zeros([0,2], np.uint16)
                    iou_thres = 0.5
                    for i, arr in enumerate(tile):
                        # get mito id
                        sid = read_h5(f'{f_mito_neuron% neuron}/{f_mito_neuron_tile(arr)}.h5')
                        if len(sid) > 0:
                            sid_rl = dict(zip(sid, count[i]+np.arange(1,1+len(sid))))
                            # get iou
                            fn = f_mito_ws(arr)
                            for j,dd in enumerate('xyz'):
                                arr_nb = arr.copy()
                                arr_nb[j*2:(j+1)*2] -= mito_tile_size[2-j]
                                tile_exist = np.abs(tile-arr_nb).max(axis=1)==0
                                if tile_exist.any():                                    
                                    sid_nb = read_h5(f'{f_mito_neuron% neuron}/{f_mito_neuron_tile(arr_nb)}.h5')                                    
                                    if len(sid_nb) > 0:
                                        iou = read_h5(f'{fn[:-3]}_{dd}.h5')
                                        merge = np.in1d(iou[:,0], sid) * np.in1d(iou[:,1], sid_nb)
                                        if merge.any():
                                            iou_m = iou[merge]
                                            iou_score = iou_m[:,4]/ iou_m[:,2:4].min(axis=1)
                                            ind = iou_m[iou_score > iou_thres,:2]
                                            if len(ind) > 0:                                             
                                                tmp = ind.copy()                                                
                                                tmp[:,0] = [sid_rl[x] for x in ind[:,0]]
                                                sid_nb_rl = dict(zip(sid_nb, count[:-1][tile_exist]+np.arange(1,1+len(sid_nb))))
                                                tmp[:,1] = [sid_nb_rl[x] for x in ind[:,1]]
                                                # import pdb;pdb.set_trace()
                                                mid = np.vstack([mid, tmp])
                                                print(i,len(mid))
                    write_h5(f_mito_neuron_match%neuron, mid)
                elif args.task == 'mito-neuron-export': # output
                    # python main.py -t mito-neuron-export -n 36750893213
                    output_file = f_mito_neuron_output % neuron
                    if not os.path.exists(output_file):                        
                        # relabel seg
                        count = read_h5(f_mito_neuron_count%neuron)
                        relabel = UnionFind(np.arange(1,1+count[-1]))
                        mid = read_h5(f_mito_neuron_match%neuron)
                        relabel.union_arr(mid)
                        relabel_arr = np.arange(count[-1]+1).astype(np.uint16)
                        to_merge = [list(x) for x in relabel.components() if len(x)>1]
                        for component in to_merge:
                            cid = min(component)
                            relabel_arr[component] = cid
                    
                        bb = np.loadtxt(f_neuron_box%neuron).astype(int)
                        output_ratio = [1,4,4]

                        bb[:2] = bb[:2] * mito_volume_ratio[0] // output_ratio[0]
                        bb[2:4] = bb[2:4] * mito_volume_ratio[1]  // output_ratio[1]
                        bb[4:] = bb[4:] * mito_volume_ratio[2]  // output_ratio[2]   
                        zyx_sz = mito_tile_size // output_ratio
                        #import pdb;pdb.set_trace()
                        
                        def zyx_to_tile(arr):
                            coord = ((arr * neuron_tile_size) + neuron_volume_offset) * mito_volume_ratio
                            # xyz
                            out = np.tile(coord[::-1],[2,1]).T.reshape(-1)
                            out[1::2] += mito_tile_size[::-1]
                            return out
                         
                        
                        def h5Name(z,y,x):
                            arr = zyx_to_tile([z,y,x])
                            sn = f_mito_ws(arr)
                            if not (np.abs(tile-arr).max(axis=1)==0).any(): 
                                sn += 'no'
                            return sn
                                                
                        def h5_func(vol, z, y, x):            
                            arr = zyx_to_tile([z,y,x])
                            sid = read_h5(f'{f_mito_neuron% neuron}/{f_mito_neuron_tile(arr)}.h5')
                            # remove sid
                            if len(sid) == 0:
                                vol[:] =0
                            else:
                                vol = seg_remove_id(vol, sid, invert=True)
                                # local sid -> global sid
                                tile_id = np.abs(tile-arr).max(axis=1)==0 
                                rl = np.zeros(sid.max()+1,np.uint16)
                                rl[sid] = count[:-1][tile_id] + np.arange(1,1+len(sid))
                                vol = rl[vol]
                                # global sid -> merged global sid
                                vol = relabel_arr[vol] 
                            return vol

                        out = read_tile_h5_by_bbox(h5Name, bb[0], bb[1]+1, bb[2], bb[3]+1, bb[4], bb[5]+1, \
                                        zyx_sz, tile_type='seg', tile_step=output_ratio[1:], zstep=output_ratio[0], \
                                            output_file=output_file, h5_func=h5_func)