import os,sys
import numpy as np
from em_util.io import *
from em_util.cluster import *

if __name__ == "__main__":
    parser = get_parser()    
    args = parser.parse_args()    
    conf = read_yml(f'param.yml')
    conf_c = conf["CLUSTER"]
    conf_p = conf["PRED"]

    if args.task == 'slurm': 
        # run in parallel
        # python main.py -t slurm -s="-t bbox -jn 10"
        # python main.py -t slurm -s="-t mito-neuron-watershed-iou -n 590612150" -jn 20
        # python main.py -t slurm -s="-t mito-neuron-watershed -n 590612150" -jn 10
        # python main.py -t slurm -s="-t mito-neuron-sid -n 590612150" -jn 10
        cmd = f'\ncd {conf["CLUSTER"]["REPO_PATH"]}'
        cmd += f'\n{conf["CLUSTER"]["CONDA"]}{args.env}/bin/python main.py {args.cmd} -ji %d -jn %d'
        
        cmd_task = args.cmd.split(' ')
        output_file = os.path.join(conf['CLUSTER']['SLURM_PATH'], cmd_task[1], 'job')
        mkdir(output_file, 'parent')
        write_slurm_all(cmd, output_file, args.job_num, args.partition, 1, args.num_gpu, args.memory, args.run_time)
    else:
        if args.task == 'dl-ts': # prepare image source for inference
            ts_dict= {
                'driver': 'neuroglancer_precomputed',
                'kvstore': {'driver': 'gcs', 'bucket': 'vclem-xh'},
                'path': 'i14clem_vf/em/',
                'scale_metadata': {'resolution': [8, 8, 30]}}
            write_pkl(conf['IMAGE_FOLDER']+conf['IMAGE_TENSORSTORE'], ts_dict)
        elif args.task == 'dl-im': # prepare image coord for inference
            xx, yy, zz = conf['IMAGE_RANGE']                
            num = conf['IMAGE_TILE_NUM']
            out_tile_txt = get_tile_coord(xx,yy,zz,num)        
            write_txt(conf['IMAGE_FOLDER']+conf['IMAGE_COORD'], out_tile_txt)
        elif args.task == 'dl-pred': # write prepare image input for inference        
            # python main.py -t dl-pred -jn 10            
            cmd = f'\n {conf_c["CUDA"]}'
            cmd += f'\ncd {conf_c["MODEL_REPO"]}'
            cmd += f'\n{conf_c["CONDA"]}{conf_p["MODEL_ENV"]}/bin/python -u scripts/main.py --inference'    
            cmd += f'--config-base {conf_p["MODEL_FOLDER"]}/{conf_p["MODEL_CONFIG"][0]} '
            cmd += f'--config-file {conf_p["MODEL_FOLDER"]}/{conf_p["MODEL_CONFIG"][1]} '
            cmd += f'--checkpoint {conf_p["MODEL_FOLDER"]}/{conf_p["MODEL_CHECKPOINT"]} '
            cmd += 'INFERENCE.DO_SINGLY True INFERENCE.DO_CHUNK_TITLE 0 '
            cmd += f'INFERENCE.IMAGE_NAME {conf_p["IMAGE_FOLDER"]}/{conf_p["IMAGE_COORD"]} '
            cmd += f'INFERENCE.TENSORSTORE_PATH {conf_p["IMAGE_FOLDER"]}/{conf_p["IMAGE_TENSORSTORE"]} '
            cmd += f'INFERENCE.OUTPUT_PATH {conf_p["MODEL_OUTPUT"]} '
            cmd += """INFERENCE.OUTPUT_NAME "f'{arr[4]:04d}/{arr[2]}-{arr[0]}'" """        
            cmd += 'INFERENCE.DO_SINGLY_START_INDEX %d INFERENCE.DO_SINGLY_STEP %d '
                            
            output_file = f'{conf_c["SLURM_PATH"]}/job'
            mkdir(output_file, 'parent')
            write_slurm_all(cmd, output_file, args.job_num, conf_c['PARTITION'], \
                            conf_c['NUM_CPU'], conf_c['NUM_GPU'], conf_c['MEMORY'], \
                            conf_c['RUNTIME'])
                
        elif args.task == 'syn-chunk': # post-process synapse prediction
            # python main.py -e pytc -t slurm -s="-t syn-chunk" -jn 10
            from connectomics.utils.process import polarity2instance
            coords = np.loadtxt(conf_p['IMAGE_FOLDER']+conf_p['IMAGE_COORD']).astype(int)                        
            for coord in coords[args.job_id::args.job_num]:
                fn = f'{conf_p["MODEL_OUTPUT"]}/{coord[4]:04d}/{coord[2]}-{coord[0]}.h5'
                sn = fn[:-3] + '_ins.h5'
                if not os.path.exists(sn):
                    print(fn)                      
                    data = read_h5(fn)
                    out = polarity2instance(data, exclusive=True)
                    write_h5(sn, [out, out.max()], ['main', 'max'])
        elif args.task == 'syn-merge-count':
            # python main.py -t syn-merge-count  
            coords = np.loadtxt(conf_p['IMAGE_FOLDER']+conf_p['IMAGE_COORD']).astype(int)
            count = np.zeros([1+len(coords)], int)
            for i, coord in enumerate(coords):
                fn = f'{conf_p["MODEL_OUTPUT"]}/{coord[4]:04d}/{coord[2]}-{coord[0]}_ins.h5'
                count[1+i] = read_h5(fn, ['max'])
            acc = np.cumsum(count).astype(np.uint32)
            write_h5(f'{conf_p["MODEL_OUTPUT"]}/count.h5', acc)
        elif args.task == 'syn-merge-pair': # compute intersection matches            
            # python main.py -t slurm -s="-t syn-merge-pair" -jn 10
            import h5py        
            count = read_h5(f'{conf_p["MODEL_OUTPUT"]}/count.h5')
            coords = np.loadtxt(conf_p['IMAGE_FOLDER']+conf_p['IMAGE_COORD']).astype(int)
            xs, ys, zs = [(coords[1:,x*2]-coords[:-1,x*2]).max() for x in range(3)]

            fn_dict = {f'{coord[4]:04d}/{coord[2]}-{coord[0]}': i for i, coord in enumerate(coords)}
            sn = f'{conf_p["MODEL_OUTPUT"]}/merge_{args.job_id}_{args.job_num}.h5'
            x0, y0, z0 = [conf_p['IMAGE_RANGE'][x][0] for x in range(3)]
            if not os.path.exists(sn):
                mm = np.zeros([0,2], np.uint32)
                for coord in coords[args.job_id::args.job_num]:
                    print(coord, len(mm))
                    fn = f'{coord[4]:04d}/{coord[2]}-{coord[0]}'
                    #pred_max = read_h5(f'{conf_p["MODEL_OUTPUT"]}/{fn}_ins.h5', ['max'])
                    pred_max = 1
                    if pred_max != 0:                        
                        z, y, x = coord[4], coord[2], coord[0]
                        if z != z0:
                            fp = f'{z-zs:04d}/{y}-{x}'
                            seg0 = np.array(h5py.File(f'{conf_p["MODEL_OUTPUT"]}/{fp}_ins.h5', 'r')['main'][-1])
                            seg = np.array(h5py.File(f'{conf_p["MODEL_OUTPUT"]}/{fn}_ins.h5', 'r')['main'][0])
                            mm = np.vstack([mm, tile_merge_syn_ins(seg0, seg, count[fn_dict[fp]], count[fn_dict[fn]])])
                        
                        if y != y0:
                            fp = f'{z:04d}/{y-ys}-{x}'
                            seg0 = np.squeeze(np.array(h5py.File(f'{conf_p["MODEL_OUTPUT"]}/{fp}_ins.h5', 'r')['main'][:, -1]))
                            seg = np.squeeze(np.array(h5py.File(f'{conf_p["MODEL_OUTPUT"]}/{fn}_ins.h5', 'r')['main'][:, 0]))
                            mm = np.vstack([mm, tile_merge_syn_ins(seg0, seg, count[fn_dict[fp]], count[fn_dict[fn]])])

                        if x != x0:
                            fp = f'{z:04d}/{y}-{x-xs}'                            
                            seg0 = np.squeeze(np.array(h5py.File(f'{conf_p["MODEL_OUTPUT"]}/{fp}_ins.h5', 'r')['main'][:,:,-1]))
                            seg = np.squeeze(np.array(h5py.File(f'{conf_p["MODEL_OUTPUT"]}/{fn}_ins.h5', 'r')['main'][:,:,0]))
                            mm = np.vstack([mm, tile_merge_syn_ins(seg0, seg, count[fn_dict[fp]], count[fn_dict[fn]])])
                write_h5(sn, mm)
        elif args.task == 'syn-merge-relabel': # relabel prediction            
            # python main.py -t syn-merge-relabel -jn 10 
            rl = UnionFind()
            for i in range(args.job_num):
                sn = f'{conf_p["MODEL_OUTPUT"]}/merge_{i}_{args.job_num}.h5'            
                mm = read_h5(sn).astype(np.uint32)
                if len(mm) > 0:
                    rl.union_arr(mm)
            rl_arr = rl.component_relabel_arr()
            write_h5(f'{conf_p["MODEL_OUTPUT"]}/relabel.h5', rl_arr)            
                        
        elif args.task == 'upload-ts':# upload to tensorstore            
            # python main.py -t slurm -s="-t upload-ts" -jn 10
            # create bucket on the browser (google cloud storage, console, cloud storage, buckets)
            # srun --pty -p serial_requeue -t 1-00:00 --mem 200000 -n 4 -N 1 /bin/bash
            # ssh tunnel 8085
            # gcloud auth application-default login
            # screen
            # sa tensorstore
            # naive multiple-process: for i in {0..3};do sleep 5;python T_xiaomeng.py 5.63 ${i} 4 & done
            import tensorstore as ts
            out_dir = 'gs://wei_lab/vclem_xh/i14clem_vf/syn_20250528'
            vol_type = 'segmentation'
            dtype = 'uint32'
            
            resolution, thickness = 8, 30
            encoding = 'compressed_segmentation'
            ratios = [[1,1,1],[4,4,1],[8,8,2],[16,16,4]]
            out_ts = [None] * len(ratios)
            datasets = [None] * len(ratios)            
            coords = np.loadtxt(conf_p['IMAGE_FOLDER']+conf_p['IMAGE_COORD']).astype(int)            
            sz = [[(coords[1:,x*2]-coords[:-1,x*2]).max() for x in range(3)]] * len(ratios)            
            vol_sz = [conf_p["IMAGE_RANGE"][x][1]-conf_p["IMAGE_RANGE"][x][0] for x in range(3)]
            vol_oset = [conf_p["IMAGE_RANGE"][x][0] for x in range(3)]
            XMAX, YMAX, ZMAX = [conf_p["IMAGE_RANGE"][x][1] for x in range(3)]
            print(vol_sz,vol_oset, sz) 
            for i,ratio in enumerate(ratios):
                out_ts[i] = {
                "driver": "neuroglancer_precomputed",
                "kvstore": out_dir,
                "open": True,
                "create": True,
                "delete_existing": False,
                "multiscale_metadata": {
                    "type": vol_type,
                    "data_type": dtype,
                    "num_channels": 1
                    },
                "scale_metadata": {
                    "size": [(vol_sz[0]+ratio[0]-1)//ratio[0], (vol_sz[1]+ratio[1]-1)//ratio[1], (vol_sz[2]+ratio[2]-1)//ratio[2]],
                    "voxel_offset": [(vol_oset[0]+ratio[0]-1)//ratio[0], (vol_oset[1]+ratio[1]-1)//ratio[1], (vol_oset[2]+ratio[2]-1)//ratio[2]],
                    "encoding": encoding,
                    "chunk_size": [128, 128, 16],
                    "resolution": [resolution*ratio[0], resolution*ratio[1], thickness*ratio[2]],
                    'compressed_segmentation_block_size': [8, 8, 8]
                    }
                }
                datasets[i] = ts.open(out_ts[i]).result()
                sz[i] = np.array(sz[i]) // ratio

            
            coords = np.loadtxt(conf_p['IMAGE_FOLDER']+conf_p['IMAGE_COORD']).astype(int)
            acc = read_h5(f'{conf_p["MODEL_OUTPUT"]}/count.h5').astype(np.uint32)
            rl_arr = read_h5(f'{conf_p["MODEL_OUTPUT"]}/relabel.h5')
            rl_max = len(rl_arr)
            cc = 0
            TS_TIMEOUT=180
            z_chunk = 3 # for high resolution, need to split z axis into multiple chunks
            fn0 = '%04d/%d-%d'
            Do = conf_p['MODEL_OUTPUT']
            mkdir(Do)
            for cid, coord in enumerate(coords):
                if cid % args.job_num == args.job_id:
                    z, y, x = coord[4], coord[2], coord[0]
                    mkdir(Do+fn0%(z,y,x), 'parent')
                    done = True
                    for i in range(len(ratios)):
                        if i == 0:
                            for ii in range(z_chunk):
                                done = done and os.path.exists(Do+fn0%(z,y,x)+'_%d-%d.txt'%(i,ii))
                        else:
                            done = done and os.path.exists(Do+fn0%(z,y,x)+'_%d.txt'%(i))
                    if not done:
                        print(z,y,x)
                        #syn_max = read_h5(Do+fn0%(z,y,x)+'_ins.h5', ['max'])
                        syn_max = 1
                        if syn_max == 0:
                            for i in range(len(ratios)):
                                if i == 0:
                                    for ii in range(z_chunk):
                                        done = Do+fn0%(z,y,x)+'_%d-%d.txt'%(i,ii)
                                        np.savetxt(done, [0], '%d')
                                else:
                                    done = Do+fn0%(z,y,x)+'-%d.txt'%(i)
                                    np.savetxt(done, [0], '%d')
                        else:
                            syn = read_h5(Do+fn0%(z,y,x)+'_ins.h5', ['main']).astype(np.uint32)
                            syn[syn > 0] += acc[cid]
                            syn[syn < rl_max] = rl_arr[syn[syn < rl_max]]
                            znum = syn.shape[0] # the last chunk may be bigger than sz[i][2]
                            # zyx -> xyzc
                            syn = syn.transpose([2,1,0])[:,:,:,None]
                            
                            for i, ratio in enumerate(ratios):
                                x0, y0, z0 = (x+ratio[0]-1)//ratio[0], (y+ratio[1]-1)//ratio[1], (z+ratio[2]-1)//ratio[2]
                                z1 = min(z0 + max(sz[i][2], (znum+ratio[2]-1)//ratio[2]), (ZMAX+ratio[2]-1)//ratio[2])
                                zstep = z_chunk if i==0 else 1
                                zran = np.linspace(z0, z1, zstep+1).astype(int)
                                for ii in range(zstep):
                                    done = Do+fn0%(z,y,x)+'_%d.txt'%(i) if i!=0 else Do+fn0%(z,y,x)+'_%d-%d.txt'%(i,ii)
                                    if not os.path.exists(done):
                                        dataview = datasets[i][x0:x0+sz[i][0], \
                                                               y0:y0+sz[i][1], \
                                                               zran[ii]: zran[ii+1]]
                                        for j in range(3):
                                            print('trial %d-%d'%(ii,j))
                                            try:
                                                dataview.write(syn[::ratio[0],::ratio[1],::ratio[2]][:,:,zran[ii]-z0: zran[ii+1]-z0]).result(timeout=TS_TIMEOUT)
                                                np.savetxt(done, [0], '%d')
                                                break
                                            except:
                                                datasets[i] = ts.open(out_ts[i]).result()
                                                dataview = datasets[i][x0:x0+sz[i][0], \
                                                                       y0:y0+sz[i][1], \
                                                                       zran[ii]: zran[ii+1]]