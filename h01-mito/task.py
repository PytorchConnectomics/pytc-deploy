import os
import numpy as np
from em_util.io import *
from const import *
import cc3d, fastremap
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_fill_holes

def generate_jobs_dl(conf, neuron, job_num=1, mem='50GB', run_time='1-00:00', job_order=1):
    PARTITION, CUDA, PYTC, PYTHON, MODEL_ROOT = conf['CLUSTER']['PARTITION'], conf['CLUSTER']['CUDA'], conf['CLUSTER']['PYTC'], conf['CLUSTER']['PYTHON'], conf['CLUSTER']['MODEL_ROOT']
    image_name = os.path.join(conf['INPUT']['ROOT_PATH'], conf['INPUT']['NEURON_PATH']%neuron)
    unet_path = os.path.join(conf['OUTPUT']['ROOT_PATH'], conf['OUTPUT']['UNET_PATH'])
    ts_path = os.path.join(conf['OUTPUT']['ROOT_PATH'], conf['OUTPUT']['UNET_PATH'])
    slurm_path = os.path.join(conf['OUTPUT']['ROOT_PATH'], conf['OUTPUT']['CMD_PATH'])
    
    for i in range(job_num):
        tmp = f"""#!/bin/bash -e
#SBATCH --job-name=h01_mito # job name
#SBATCH -N 1 # how many nodes to use for this job
#SBATCH -n 1 # how many CPU-cores (per node) to use for this job
#SBATCH --gres=gpu:1
#SBATCH --mem={mem} # how much RAM (per node) to allocate
#SBATCH -t {run_time} # job execution time limit formatted hrs:min:sec
#SBATCH --partition={PARTITION} # see sinfo for available partitions
#SBATCH -e out.%N.%j.err # STDERR
#SBATCH -o out.%N.%j.out

{CUDA}
cd {PYTC}
{PYTHON} -u scripts/main.py --inference \
--config-base {MODEL_ROOT}/MitoEM-R-Base.yaml \
--config-file {MODEL_ROOT}/MitoEM-R-BC.yaml \
--checkpoint {MODEL_ROOT}/mito_u3d-bc_mitoem_300k.pth.tar \
INFERENCE.DO_SINGLY True INFERENCE.DO_CHUNK_TITLE 0 \
INFERENCE.DO_SINGLY_START_INDEX {i} INFERENCE.DO_SINGLY_STEP {job_num*job_order} \
INFERENCE.IMAGE_NAME {image_name} \
INFERENCE.OUTPUT_PATH {unet_path} \
INFERENCE.TENSORSTORE_PATH {ts_path} \
"""
        tmp += """INFERENCE.OUTPUT_NAME "f'{arr[4]:04d}/{arr[2]}-{arr[0]}'" """ 
        write_txt(f'{slurm_path}/{neuron}_{i}_{job_num}.sh', tmp)
    print(f'cd {slurm_path}')
    print('for i in {0..%d};do sbatch %d_${i}_%d.sh && sleep 1;done' % (job_num - 1, neuron, job_num))


def neuron_to_tile(neuron, zid, zran, f_box, f_seg):
    zz = zran[zid[:,0]==neuron][0]
    out_bb, out_tile = None, []
    for z in range(zz[0], zz[1]+1):
        print(z, len(out_tile))
        bbox = read_h5(f_box % z)
        if (bbox[:,0] == neuron).any(): 
            bb = [z,z] + list(bbox[bbox[:,0]==neuron, 1:-1][0])
            if z == zz[0]:
                out_bb = bb.copy()
            else:
                out_bb = merge_bbox(out_bb, bb) 
            
            st = bb[::2]//neuron_tile_size
            lt = (bb[1::2]+neuron_tile_size-1)//neuron_tile_size
            seg = read_h5(f_seg % z)                                
            for rr in range(st[1], lt[1]+1):
                for cc in range(st[2], lt[2]+1):
                    if (seg[rr*tsz[1]:(rr+1)*tsz[1], cc*tsz[2]:(cc+1)*tsz[2]]==neuron).any():
                        out_tile.append([st[0],rr,cc])
    # much of the box can be empty
    # tile = itertools.product(range(st[0],lt[0]),range(st[1],lt[1]),range(st[2],lt[2]))            
    out_tile = np.unique(np.vstack(out_tile), axis=0)
    out_tile_bbox = np.zeros([len(out_tile), 6], int)
    for j in range(len(out_tile)):
        xs = [(neuron_volume_offset[i]+out_tile[j][i]*tsz[i])*ratio[i] for i in range(3)]
        xl = [min(sz[i], (neuron_volume_offset[i]+(out_tile[j][i]+1)*tsz[i]))*ratio[i] for i in range(3)]
        #zyx -> xyz
        out_tile_bbox[j, ::2] = xs[::-1]
        out_tile_bbox[j, 1::2] = xl[::-1]        
    return out_bb, out_tile_bbox

def seg_zran_merge(f_zran_p, job_num):
    out = read_h5(f_zran_p % (0, job_num))
    out_id, out_zran = out[:,:1].copy(), out[:,1:].astype(np.uint16)
    del out
    for i in range(1, args.job_num):
        # sort seg id
        oid = np.argsort(out_id[:,0])
        out_id, out_zran = out_id[oid], out_zran[oid]

        bbox = read_h5(f_zran_p % (i, args.job_num))
        bbox_id, bbox_zran = bbox[:,:1].copy(), bbox[:,1:].astype(np.uint16)
        del bbox
        bid = np.argsort(bbox_id[:,0])
        bbox_id, bbox_zran = bbox_id[bid], bbox_zran[bid]

        out_in = np.in1d(out_id, bbox_id)
        bbox_in = np.in1d(bbox_id, out_id)
        out_id = np.vstack([out_id, bbox_id[np.logical_not(bbox_in)]])
        del bbox_id
        
        out_zran[out_in, 1] = bbox_zran[bbox_in, 1]        
        out_zran = np.vstack([out_zran, bbox_zran[np.logical_not(bbox_in)]])
        del bbox_zran
        print(i, len(out_id))
    return out_id, out_zran

def seg_zran_p(f_box, job_id, job_num):
    ind = np.linspace(0, sz[0], job_num+1).astype(int)
    bbox = read_h5(f_box % ind[job_id])
    out = np.hstack([bbox[:,:1], ind[job_id]*np.ones([len(bbox),2])]).astype(np.uint64)
    for z in range(ind[job_id]+1,ind[job_id+1]):
        print(z, flush=True)
        bbox = read_h5(f_box % z)
        if len(bbox) > 0:
            # update max
            ind_in = np.in1d(out[:,0], bbox[:,0])
            out[ind_in, 2] = z
            # add new
            ind_out = np.in1d(bbox[:,0], out[:,0], invert=True)
            out = np.vstack([out, np.hstack([bbox[ind_out,:1], z*np.ones([ind_out.sum(),2])])])
    return out

def seg_bbox_p(f_seg, f_box, job_id, job_num):                    
    for z in range(sz[0])[job_id::job_num]:
        if not os.path.exists(f_box % z):
            print(z)
            seg = read_h5(f_seg % z)
            bbox = compute_bbox_all(seg, True)
            write_h5(f_box % z, bbox)    


def remove_small_instances(segm: np.ndarray,
                           thres_small: int = 128,
                           mode: str = 'background'):
    """Remove small spurious instances.
    """
    assert mode in ['none',
                    'background',
                    'background_2d',
                    'neighbor',
                    'neighbor_2d']

    if mode == 'none':
        return segm

    # The function remove_small_objects expects ar to be an array with labeled objects, and 
    # removes objects smaller than min_size. If ar is bool, the image is first labeled. This 
    # leads to potentially different behavior for bool and 0-and-1 arrays. Reference:
    # https://scikit-image.org/docs/stable/api/skimage.morphology.html#remove-small-objects
    if mode == 'background':
        return remove_small_objects(segm, thres_small)
    elif mode == 'background_2d':
        temp = [remove_small_objects(segm[i], thres_small)
                for i in range(segm.shape[0])]
        return np.stack(temp, axis=0)

    if mode == 'neighbor':
        return merge_small_objects(segm, thres_small, do_3d=True)
    elif mode == 'neighbor_2d':
        temp = [merge_small_objects(segm[i], thres_small)
                for i in range(segm.shape[0])]
        return np.stack(temp, axis=0)

def bc_watershed(volume, thres1=0.9, thres2=0.8, thres3=0.85, thres_small=128, scale_factors=(1.0, 1.0, 1.0),
                 remove_small_mode='background', seed_thres=32, precomputed_seed=None):
    r"""Convert binary foreground probability maps and instance contours to
    instance masks via watershed segmentation algorithm.

    Note:
        This function uses the `skimage.segmentation.watershed <https://github.com/scikit-image/scikit-image/blob/master/skimage/segmentation/_watershed.py#L89>`_
        function that converts the input image into ``np.float64`` data type for processing. Therefore please make sure enough memory is allocated when handling large arrays.

    Args:
        volume (numpy.ndarray): foreground and contour probability of shape :math:`(C, Z, Y, X)`.
        thres1 (float): threshold of seeds. Default: 0.9
        thres2 (float): threshold of instance contours. Default: 0.8
        thres3 (float): threshold of foreground. Default: 0.85
        thres_small (int): size threshold of small objects to remove. Default: 128
        scale_factors (tuple): scale factors for resizing in :math:`(Z, Y, X)` order. Default: (1.0, 1.0, 1.0)
        remove_small_mode (str): ``'background'``, ``'neighbor'`` or ``'none'``. Default: ``'background'``
    """
    assert volume.shape[0] == 2
    semantic = volume[0]
    boundary = volume[1]
    foreground = (semantic > int(255*thres3))

    if precomputed_seed is not None:
        seed = precomputed_seed
    else: # compute the instance seeds
        seed_map = (semantic > int(255*thres1)) * (boundary < int(255*thres2))
        seed = cc3d.connected_components(seed_map, connectivity=6)
        if seed_thres > 0:
            seed = remove_small_objects(seed, seed_thres)

    segm = watershed(-semantic.astype(np.float64), seed, mask=foreground)
    
    if thres_small > 0:
        segm = remove_small_instances(segm, thres_small, remove_small_mode)

    segm, _ = fastremap.renumber(segm, in_place=True)
    return segm

    
def mito_watershed_iou(f_mito_ws_func, arr_mito):
    fn = f_mito_ws_func(arr_mito)
    if os.path.exists(fn):
        seg = None
        for i, dd in enumerate('xyz'):
            fout = f'{fn[:-3]}_{dd}.h5'
            if not os.path.exists(fout):
                arr_nb = arr_mito.copy()
                arr_nb[i*2:(i+1)*2] -= mito_tile_size[2-i]
                fn_nb = f_mito_ws_func(arr_nb)
                if os.path.exists(fn_nb):
                    if seg is None:
                        seg = read_h5(fn)
                    seg_nb = read_h5(fn_nb)                
                    if dd == 'x':
                        iou = seg_to_iou(seg[:,:,0], seg_nb[:,:,-1])
                    elif dd == 'y':
                        iou = seg_to_iou(seg[:,0], seg_nb[:,-1])
                    elif dd == 'z':
                        iou = seg_to_iou(seg[0], seg_nb[-1])
                    iou = iou[iou[:,1]!=0]
                    write_h5(fout, iou)              
    
    
def mito_neuron_sid(f_mito_ws, arr_mito, ratio=0.6):
    mito = read_h5(f_mito_ws)
    arr_neuron = arr_mito.copy()
    arr_neuron[::2] = arr_neuron[::2] // mito_volume_ratio[::-1] - neuron_volume_offset[::-1]
    arr_neuron[1::2] = arr_neuron[1::2] // mito_volume_ratio[::-1] - neuron_volume_offset[::-1]
    seg = read_slice_volume(seg_fns, arr_neuron[4], arr_neuron[5], arr_neuron[2], \
                            arr_neuron[3], arr_neuron[0], arr_neuron[1], np.uint64, \
                            mito_volume_ratio[1:], 0, neuron_volume_size)
    seg = binary_fill_holes(seg==neuron)
    assert seg.any()
    
    out = []
    for z in range(mito_volume_ratio[0]):                            
        ui, uc = np.unique(mito[z::mito_volume_ratio[0]] * seg, return_counts=True)
        uc = uc[ui>0]
        ui = ui[ui>0]
        if len(ui) >0:
            if len(out) == 0:
                out = dict(zip(ui, uc))
            else:
                for x,y in zip(ui,uc):
                    if x in out:
                        out[x] += y
                    else:
                        out[x] = y                        
    if len(out) == 0:
        sid = []
    else:
        # remove ones with small overlap                            
        ui, uc = np.unique(mito, return_counts=True) 
        ind = np.where(np.in1d(ui, [x for x in out.keys()]))[0]
        sid = [ui[x] for x in ind if out[ui[x]]/uc[x]>overlap_ratio]
        # score = [out[ui[x]]/uc[x] for x in ind]
    return sid
