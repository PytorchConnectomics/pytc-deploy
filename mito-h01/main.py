import os,sys
import numpy as np
from em_util.io import *
from em_util.cluster import *
from task import *

if __name__ == "__main__":
    conf = read_yml(f'param.yml')
    parser = get_parser()    
    args = parser.parse_args()
    if args.neuron != '':
        if '.txt' in args.neuron:
            # read from file
            args.neuron = np.loadtxt(args.neuron).astype(int).ravel()
        else:
            # read from command line
            args.neuron = [int(x) for x in args.neuron.split(',')] 
    args.ratio = [int(x) for x in args.ratio.split(',')]    
    if args.partition != '':
        conf['CLUSTER']['PARTITION'] = args.partition
    
    conf_m, conf_p, conf_c = conf['MASK'], conf['PRED'], conf['CLUSTER']
    f_seg = os.path.join(conf_m['ROOT_PATH'], conf_m['SEG_PATH'])
    f_box = os.path.join(conf_m['ROOT_PATH'], conf_m['SEG_BBOX_PATH'])
    f_zran_p = os.path.join(conf_m['ROOT_PATH'], conf_m['SEG_ZRAN_PATH'], 'bb_zran_%d_%d.h5')
    f_zran = os.path.join(conf_m['ROOT_PATH'], conf_m['SEG_ZRAN_PATH'], 'bb_zran.h5')
    f_neuron_tile = os.path.join(conf_m['ROOT_PATH'], conf_m['NEURON_TILE_PATH'])
    f_neuron_box = os.path.join(conf_m['ROOT_PATH'], conf_m['NEURON_BBOX_PATH'])
    
    f_mito_tile = os.path.join(conf_p['ROOT_PATH'], conf_p['MODEL_OUTPUT'])
    f_mito_ts = os.path.join(conf_p['ROOT_PATH'], conf_p['TENSORSTORE_PATH'])    
    f_tile = lambda arr: f'{arr[4]:04d}/{arr[2]}-{arr[0]}' 
    f_mito_pred = lambda arr: f'{f_mito_tile}/{f_tile(arr)}.h5'
    f_mito_ws = lambda arr: f'{f_mito_tile}/{f_tile(arr)}_ws.h5'
    f_mito_neuron = os.path.join(conf_p['ROOT_PATH'], conf_p['NEURON_MITO_PATH'])    
    f_mito_neuron_tile = lambda arr: f'{arr[4]:04d}_{arr[2]}_{arr[0]}' 
    f_mito_neuron_count = os.path.join(f_mito_neuron, 'count.h5')
    f_mito_neuron_match = os.path.join(f_mito_neuron, 'match.h5')
    f_mito_neuron_output = os.path.join(f_mito_neuron, 'mito.h5')
    f_mito_neuron_output_ds = lambda arr: f'{f_mito_neuron_output[:-3]}_{arr[0]}_{arr[1]}_{arr[2]}.h5'
    f_mito_neuron_output_ds_box = lambda arr: f'{f_mito_neuron_output[:-3]}_{arr[0]}_{arr[1]}_{arr[2]}_bb.h5'
    f_mito_neuron_cls = os.path.join(f_mito_neuron, 'mito_cls_bbox.txt')
    f_mito_neuron_cls_pred = os.path.join(f_mito_neuron, 'mito_cls_bbox_pred.txt')
    

    if args.task == 'slurm': 
        # run in parallel        
        # python main.py -t slurm -s="-t mito-neuron-watershed-iou -n 590612150" -jn 20
        # python main.py -t slurm -s="-t mito-neuron-watershed -n 590612150" -jn 10
        # python main.py -t slurm -s="-t mito-neuron-sid -n 590612150" -jn 7
        cmd = f'\ncd {conf_c["REPO_PATH"]}'
        cmd += f'\n{conf_c["CONDA"]}{args.env}/bin/python main.py {args.cmd} -ji %d -jn %d'  
        
        cmd_task = args.cmd.split(' ')
        output_file = os.path.join(conf_c['SLURM_PATH'], cmd_task[1], 'job')
        mkdir(output_file, 'parent')
        write_slurm_all(cmd, output_file, args.job_num, args.partition, 1, args.num_gpu, args.memory, args.run_time)
    else:
        # execute each task
        if args.task == 'seg-bbox':
            # python main.py -t slurm -s="-t seg-bbox -jn 10"
            # python main.py -t seg-bbox -ji 0 -jn 1000
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
            # python main.py -t 
            # python main.py -t slurm -s="-t neuron-tile -n /n/boslfs02/LABS/lichtman_lab/donglai/H01/mito_mip1/pair_L2.txt" -jn 10 -ct 2-00:00
            mkdir(f_neuron_tile, 'parent')
            zid, zran = read_h5(f_zran,['id','zran'])         
            for neuron in args.neuron[args.job_id::args.job_num]:
                if not os.path.exists(f_neuron_box % neuron):
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
        elif args.task == 'mito-pred': # generate sbatch files for DL prediction
            # mmg 1 50000 1 gpu_test
            # python main.py -t mito-pred -n 4404218019,39879625402 -jn 5 -cp gpu-test
            for neuron in args.neuron:
                generate_jobs_dl(conf, neuron, job_num=args.job_num)
        elif 'mito-neuron' in args.task:
            for nid, neuron in enumerate(args.neuron):
                tile = np.loadtxt(f_neuron_tile % neuron).astype(int)
                if args.task == 'mito-neuron-check': # pkl for ts
                    # python main.py -t mito-neuron-check -n 590612150
                    num = 0
                    for arr in tile:
                        #fn = f_mito_ws(arr)
                        fn = f_mito_pred(arr)
                        #fn = f_mito_neuron_tile(arr)
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
                elif args.task == 'mito-neuron-watershed-iou': # decode prediction into instance
                    # python main.py -t mito-neuron-watershed-iou -n 36750893213
                    # compare to the previous one
                    for arr in tile[args.job_id::args.job_num]:                    
                        mito_watershed_iou(f_mito_ws, arr)
                                        
                elif args.task == 'mito-neuron-sid': # compute the seg ids within the mask
                    # python main.py -t mito-neuron-sid -n 36750893213
                    if nid == 0:
                        seg_fns = [f_seg%z for z in range(neuron_volume_size[0])]                                    
                    D0 = f_mito_neuron%neuron
                    mkdir(D0, 'all')
                    overlap_ratio = 0.6
                    for arr in tile[args.job_id::args.job_num]:                                        
                        fout = f'{D0}/{f_mito_neuron_tile(arr)}.h5'
                        if not os.path.exists(fout):
                            sid = mito_neuron_sid(f_mito_ws(arr), arr, seg_fns, neuron, overlap_ratio)
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
                    relabel = UnionFind(np.arange(1,1+count[-1]))
                    relabel.union_arr(mid)
                    relabel_arr = relabel.component_relabel_arr()                    
                    write_h5(f_mito_neuron_match%neuron, [relabel_arr, mid], ['relabel','mid'])
                elif args.task == 'mito-neuron-test': # output
                    relabel_arr = read_h5(f_mito_neuron_match%neuron, ['relabel'])
                    print_arr(np.unique(relabel_arr))

                elif args.task == 'mito-neuron-export': # output: 33x32x32nm
                    # python main.py -t mito-neuron-export -n 36750893213
                    output_file = f_mito_neuron_output % neuron
                    if not os.path.exists(output_file):                        
                        # relabel seg
                        count = read_h5(f_mito_neuron_count%neuron)
                        relabel_arr = read_h5(f_mito_neuron_match%neuron, ['relabel'])
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

                        out = read_tile_h5_volume(h5Name, bb[0], bb[1]+1, bb[2], bb[3]+1, bb[4], bb[5]+1, \
                                        zyx_sz, tile_type='seg', tile_step=output_ratio[1:], zstep=output_ratio[0], \
                                            output_file=output_file, h5_func=h5_func)
                elif args.task == 'mito-neuron-export-ds': # downsample: 132x128x128
                    # python main.py -t mito-neuron-export-ds -n 590612150 -r 4,4,4 -cn 20
                    # [mito[x*72:(x+1)*72].max() for x in range(10)]
                    ds_file = f_mito_neuron_output_ds(args.ratio) % neuron
                    if not os.path.exists(ds_file):
                        input_file = f_mito_neuron_output % neuron
                        vol_downsample_chunk(input_file, args.ratio, output_file=ds_file, chunk_num=args.chunk_num)
                elif args.task == 'mito-neuron-stats': # compute stats on low-res
                    # python main.py -t mito-neuron-stats -n 590612150 -r 4,4,4 -cn 20
                    bbox = compute_bbox_all_chunk(f_mito_neuron_output_ds(args.ratio) % neuron, True, chunk_num=args.chunk_num)                    
                    write_h5(f_mito_neuron_output_ds_box(args.ratio) % neuron, bbox)
                                
                elif args.task == 'mito-neuron-pf': # apply corrections                   
                    # python main.py -t mito-neuron-pf -n 590612150 -r 4,4,4
                    # bounding box offset: 128x128x132 -> 8x8x33
                    # import pdb;pdb.set_trace()
                    neuron_pos = np.loadtxt(f_neuron_box % neuron).astype(int)
                    oset = (neuron_pos[::2] + neuron_volume_offset) * mito_volume_ratio

                    # mito bounding box: 128x128x132 -> 8x8x33 
                    bbox = read_h5(f_mito_neuron_output_ds_box(args.ratio) % neuron)
                    bbox[:,1:3] = bbox[:,1:3] * 4 + oset[0]
                    bbox[:,3:5] = bbox[:,3:5] * 16 + oset[1]
                    bbox[:,5:7] = bbox[:,5:7] * 16 + oset[2]
                    np.savetxt(f_mito_neuron_cls % neuron, bbox, '%d') 
                    """
                    fp = [201, 66, 68, 70, 72, 73, 75, 77, 78, 117, 122, 123, 126, 128, 129, 130, 131, 711, 713, 715, 85, 440, 453, 454, 455, 463, 410, 411, 417, 416, 418, 419, 420, 434, 437, 441, 442, 443, 445, 446, 447, 448, 450, 456, 258, 260, 611, 614, 635, 144, 153, 145, 147, 148, 149, 150, 151, 154, 155, 156, 157, 158, 160, 89, 91, 94, 95, 96, 98, 100, 102, 81, 412, 439, 484, 487, 488, 490, 492, 494, 495, 498, 491, 505, 507, 503, 509, 481, 502, 510, 521, 532, 533, 534, 529, 527, 540, 541, 542, 544, 547, 307, 538, 622, 623, 632, 633, 637, 638, 642, 643, 636, 626, 627, 628, 629, 630, 634, 716, 717, 718, 734, 702, 703, 705, 708, 710, 816, 886, 887, 888, 15, 16, 13, 56, 58, 59, 26, 49, 54, 284, 285, 286, 292, 171, 181, 183, 182, 184, 187, 186, 191, 192, 193, 777, 774, 721, 725, 729, 722, 723, 724, 726, 732, 733, 234, 247, 245, 246, 252, 251, 248, 249, 253, 310, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 695, 696, 698, 699, 644, 244, 255, 1129, 1062, 132, 135, 324, 180, 185, 188, 189, 190, 195, 980, 981, 982, 983, 984, 985, 986, 987, 988, 989, 991, 993, 995, 996, 997, 999, 933, 937, 938, 939, 602, 603, 501, 767, 771, 770, 769, 768, 766, 752, 601, 964, 963, 970, 965, 969, 968, 966, 375, 376, 378, 380, 381, 382, 364, 363, 362, 361, 360, 365, 367, 908, 906, 904, 903, 898, 947, 948, 950, 951, 435, 875, 220, 397, 423, 424, 425, 426, 427, 428, 432, 465, 466, 473, 474, 569, 573, 574, 575, 578, 579, 580, 582, 584, 585, 586, 587, 588, 592, 593, 594, 595, 596, 597, 599, 1124, 1037, 1201, 1066, 1031, 833, 798, 800, 801, 802, 803, 805, 806, 807, 808, 809, 810, 742, 524, 525, 489, 477, 476, 470, 480, 475, 472, 478, 479, 471, 314, 315, 316, 317, 318, 326, 329, 300, 301, 302, 303, 305, 297, 298, 267, 268, 271, 172, 173, 18, 433, 396, 1114, 1115, 1116, 515, 516, 535, 338, 339, 340, 709, 712]
                    ind = np.in1d(bbox[:,0], fp, invert=True)
                    print('FP')
                    np.savetxt('tmp_fp.txt', bbox[ind, 1:-1], '%d')
                    print('TP')
                    np.savetxt('tmp_tp.txt', bbox[np.logical_not(ind), 1:-1], '%d')
                    """
                    pass

                elif args.task in ['mito-neuron-ng','mito-neuron-mesh']: # ng generation
                    # python main.py -t mito-neuron-ng -n 36750893213
                    from em_util.ng import NgDataset
                    Dw = 'file:///n/holylfs05/LABS/pfister_lab/Everyone/public/ng/'
                    wn = 'h01_mito'
                    input_file = f_mito_neuron_output % neuron
                    fid = h5py.File(input_file,'r')
                    vol = fid['main']

                    volume_size = vol.shape[::-1]
                    resolution = [32, 32, 33]
                    mip_ratio = [[1,1,1],[2,2,2],[4,4,4]]
                    output_seg = f'{Dw}{wn}/{neuron}/'
                    mkdir(output_seg[7:], 'all')
                    # 128x128x133
                    oo = 4 * (np.loadtxt(f_neuron_box%neuron).astype(int)[::2] + neuron_volume_offset)
                    dst = NgDataset(volume_size, resolution, mip_ratio, offset=oo[::-1])
                    
                    #mito_bid = None

                    mito_score = np.loadtxt(f_mito_neuron_cls_pred%neuron)
                    mito_bid = np.loadtxt(f_mito_neuron_cls%neuron)[mito_score<0.3,0].astype(int)

                    dst.create_info(output_seg, 'seg')
                    if args.task == 'mito-neuron-ng':
                        def get_seg(z0, z1, y0, y1, x0, x1, mip):
                            tmp = np.array(vol[z0: z1:mip[2], y0: y1:mip[1], x0: x1: mip[0]])
                            if mito_bid is not None:
                                tmp = seg_remove_id(tmp, mito_bid)
                            return tmp 
                        dst.create_tile(get_seg, output_seg, 'seg', range(len(mip_ratio)))
                    elif args.task == 'mito-neuron-mesh':
                        dst.create_mesh(output_seg, 1, (np.array(volume_size)+1)//2, do_subdir = True)
                        #dst.create_mesh(output_seg, 0, volume_size, do_subdir = True)
                    fid.close()
                elif args.task == 'mito-neuron-debug': # downsample
                    # python main.py -t mito-neuron-debug -n 36750893213
                    # downsample: all 0 for big z
                    """
                    count = read_h5(f_mito_neuron_count%neuron)
                    relabel = UnionFind(np.arange(1,1+count[-1]))
                    mid = read_h5(f_mito_neuron_match%neuron)
                    relabel.union_arr(mid)
                    relabel_arr = np.arange(count[-1]+1).astype(np.uint16)
                    to_merge = [list(x) for x in relabel.components() if len(x)>1]
                    for component in to_merge:
                        cid = min(component)
                        relabel_arr[component] = cid
                    print(len(np.unique(relabel_arr)))
                    # 1081 seg
                    import pdb; pdb.set_trace()
                    """
                    count = read_h5(f_mito_neuron_count%neuron)
                    arr = [4800,53248,328704]
                    sid = read_h5(f'/n/boslfs02/LABS/lichtman_lab/donglai/H01/mito_mip1/neuron/36750893213/{arr[0]}_{arr[1]}_{arr[2]}.h5')
                    fid = h5py.File(f'/n/boslfs02/LABS/lichtman_lab/donglai/H01/mito_mip1/neuron/36750893213/mito.h5','r')['main']
                    bb = [  509,  1227,   800,  2573, 15667, 17208]
                    oset = np.array([0,2560,3520])
                    bb2 = (oset+bb[::2])*mito_volume_ratio
                    import pdb; pdb.set_trace()
                    aa = np.array(fid[arr[0]-bb[0]*4:arr[0]+100-bb[0]*4])
                    import pdb; pdb.set_trace()
                    #mito = read_h5('/n/boslfs02/LABS/lichtman_lab/donglai/H01/mito_mip1/tile//4800/53248-328704_ws.h5')
                    print(len(np.unique(mito)))
