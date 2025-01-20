import os,sys,shutil
sys.path.append('../')
import numpy as np
import scipy as sp
import h5py
from T_util import readh5, writeh5, readtxt, writetxt,get_bb,get_union,printArr
from T_util import vast2Seg, seg2Vast, U_mkdir, vastMetaRelabel
from em_util.io import *
import shutil


Dv = '/n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/vcg_connectomics/'
Dl = '/n/boslfs/LABS/lichtman_lab/Donglai/'
Dl2 = '/n/boslfs02/LABS/lichtman_lab/donglai/'
Dw = '/n/holylfs05/LABS/pfister_lab/Everyone/public/'
opt = sys.argv[1]
job_id=0;job_num=1
if len(sys.argv) > 3:
    job_id = int(sys.argv[2])
    job_num = int(sys.argv[3])

D0i = Dl2 + 'xiaomeng/'
if opt[0] == '0':
    # [512,512,20]
    nn = 'pos0'
    nn = 'neg0'
    Din = D0i + nn+ '/'
    ran = [[238,619], [26470, 29253], [31084, 33779]] 
    sz = [x[1]-x[0]+1 for x in ran]
    print(sz)
    nns=['im','seg']
    if opt == '0':# find unique id
        bid = []
        tmp = imread(Din+'im/_s%03d.png'%(ran[0][0]))[::8,::8]
        for z in range(ran[0][0]+1, ran[0][1]+1):
            tmp2 = imread(Din+'im/_s%03d.png'%(z))[::8,::8]
            if np.all(tmp.astype(float)-tmp2==0):
                bid.append(z-ran[0][0])
            tmp = tmp2
        np.savetxt(Din+'bid.txt',bid,'%d')

    elif opt == '0.01':
        for nn in nns:
            sn = Din +'%s.h5'%nn
            if True:#not os.path.exists(sn):
                out = np.zeros(sz, np.uint8)
                for z in range(ran[0][0], ran[0][1]+1):
                    tmp = imread(Din+'%s/_s%03d.png'%(nn,z))
                    if nn =='seg':
                        tmp = vast2Seg(tmp)
                    out[z-ran[0][0]] = tmp
                writeh5(sn, out)
    elif opt == '0.1':
        seg = readh5('/n/pfister_lab2/Lab/zudilin/data/Xiaomeng/mito_oct15_22/segm_xy.h5')
        #seg = readh5('/n/pfister_lab2/Lab/zudilin/data/Xiaomeng/mito_oct14_22/segm_xy.h5')
        writeh5('xm/mito_32nm_v2.h5', seg[:,::4,::4])
        #im = readh5(Do +'im.h5')
        import pdb; pdb.set_trace()
    elif opt == '0.11':
        seg = h5py.File('/n/pfister_lab2/Lab/zudilin/data/Xiaomeng/mito_oct15_22/segm_xy.h5','r')['main']
        for z in range(seg.shape[0]):
            imwrite(D0i+'terminal/mito_pred/%04d.png'%z, np.array(seg[z]))
    elif opt == '0.2':# read in pf
        #bid = np.loadtxt(Din+'bid.txt').astype(int)
        #Din = D0i + 'neg0/'
        Din = D0i + 'pos0/'
        bb = np.loadtxt(Din+'im_bb.txt').astype(int)
        sz = bb[1::2] - bb[::2] + 1
        rr = 4
        rr = 1
        out = np.zeros([sz[0],sz[1]//rr,sz[2]//rr],np.uint16)
        rl = vastMetaRelabel(Din+'mito_pf/meta.txt') 
        for z in range(sz[0]):
            out[z] = rl[vast2Seg(imread(Din+'mito_pf/_s%03d.png'%z))][::rr,::rr]
            """
            if z in bid:
                out[z] = out[z-1]
            else:
                out[z] = rl[vast2Seg(imread(Din+'mito_pf/_s%03d.png'%z))][::rr,::rr]
                #out[z] = rl[vast2Seg(imread(Din+'mito_pf/_s%03d.png'%z))]
            """
        writeh5(Din+'mito_%dnm_pf.h5'%(8*rr), out)
    elif opt == '0.21':# read in pf
        for nn in ['pos0','neg0']:
            Din = D0i + nn + '/'
            seg = readh5(Din+'mito_8nm_pf.h5')
            writeh5(Din+'mito_32nm_pf.h5', seg[:,::4,::4])

if opt[0]=='9':
    nns = ['pos','neg']
    #nns = ['neg']
    for nn in nns:
        for nid in range(1):
        #for nid in range(10):
            mm = '%s%d'%(nn,nid)
            if mm =='pos0':
                pass
                #continue
            Din = D0i + mm + '/'
            Dmito= '/n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/zudilin/data/Xiaomeng_Mito/segm_022823/'
            if opt == '9':# Vast export -> h5 [im, seg]
                tt = 'bb'
                tt = 'crop'
                tt = 'im'
                tt = 'seg'
                tt = 'mito'
                tt = 'ves'
                from glob import glob
                from T_util import get_union
                sn_b = Din+'im_bb.txt'
                fn_s = Din + 'seg/_s%03d.png'
                fn_i = Din + 'im/_s%03d.png'
                if tt == 'bb':# get bounding box
                    if not os.path.exists(sn_b):
                        fns = sorted(glob(Din+'seg/*.png')) 
                        z0 = int(fns[0][fns[0].rfind('s')+1:-4])
                        zn = int(fns[-1][fns[-1].rfind('s')+1:-4])
                        bb = [1e6,0,1e6,0]
                        for fn in fns:
                            tmp = get_bb(imread(fn))
                            if tmp[0]>-1:
                                bb = get_union(bb, tmp)
                            else:
                                import pdb; pdb.set_trace()
                            print(fn,bb)
                        np.savetxt(sn_b, bb, '%d')
                    else:
                        bb = np.loadtxt(sn_b).astype(int)
                        print(bb)
                else:
                    bb = np.loadtxt(sn_b).astype(int)
                    sz1 = bb[1::2]-bb[::2]+1
                    #sz2 = h5py.File(Din+'im.h5','r')['main'].shape
                    print(mm,sz1)
                    if tt == 'im':
                        sn = Din+'im.h5'
                        #if np.abs(sz1-sz2).max()==0:
                        #    pass
                        #else:
                        if not os.path.exists(sn):
                            out = np.zeros(bb[1::2]-bb[::2]+1,np.uint8)
                            for i in range(bb[0],bb[1]+1):
                                out[i-bb[0]] = imread(fn_i%i)[bb[2]:bb[3]+1,\
                                                            bb[4]:bb[5]+1]
                            writeh5(sn, out)
                    elif tt == 'crop':
                        fn_i2 = Din + 'crop/im_%03d.png'
                        fn_s2 = Din + 'crop/seg_%03d.png'
                        U_mkdir(fn_s2, 'parent')
                        for i in range(bb[0],bb[1]+1):
                            imwrite(fn_i2%i, imread(fn_i%i)[bb[2]:bb[3]+1,\
                                                        bb[4]:bb[5]+1])
                            imwrite(fn_s2%i, imread(fn_s%i)[bb[2]:bb[3]+1,\
                                                        bb[4]:bb[5]+1])
                    elif tt == 'seg':
                        sn = Din+'seg_32nm.h5'
                        if not os.path.exists(sn):
                            sz = bb[1::2]-bb[::2]+1
                            sz = np.ceil(sz/[1,4.0,4.0]).astype(int)
                            out = np.zeros(sz,np.uint8)
                            for i in range(bb[0],bb[1]+1):
                                out[i-bb[0]] = imread(fn_s%i)[bb[2]:bb[3]+1:4,\
                                                     bb[4]:bb[5]+1:4]
                            writeh5(sn, out)
                            print(out.shape)
                    elif tt == 'mito': # export
                        sz = bb[1::2]-bb[::2]+1
                        out = np.zeros(sz,np.uint8)
                        # pos 1-9, neg0-9
                        seg = h5py.File(Dmito+'segm_%02d_xy.h5'%((nn=='neg')*10+nid-1),'r')['main']
                        sn = Din + 'mito/%03d.png' 
                        U_mkdir(sn, 'parent')
                        for i in range(bb[0],bb[1]+1):
                            mito = np.array(seg[i-bb[0]])
                            cell = imread(fn_s%i)[bb[2]:bb[3]+1,\
                                                 bb[4]:bb[5]+1]
                            imwrite(sn%(i-bb[0]), seg2Vast(mito*(cell>0)))

            elif opt == '9.1':
                print(mm)
                seg = readh5(Dmito+'segm_%02d_xy.h5'%((nn=='neg')*10+nid-1))
                cell = readh5(D0i + mm + '/seg_32nm.h5')
                writeh5(D0i + mm + '/mito_32nm.h5', (cell>0)*seg[:,::4,::4])
            elif opt == '9.11': # error detection ids
                seg = readh5(D0i + mm + '/mito_32nm.h5')
                ui,uc=np.unique(seg, return_counts=True)
                uc[ui==0] = 0
                """
                # seg id to proofread
                print(mm, '%.2f'%(uc[uc>30].sum()*(0.032*0.032*0.03)))
                """
                # seg id to proofread
                print('-----')
                print(mm)
                printArr(ui[np.argsort(-uc)][:(uc>30).sum()],10)
                print('-----')

elif opt[0] == '2':# vesicles
    if opt == '2': #image for intern
        D1 = D0i + 'terminal/'
        bid = np.loadtxt(D1+'bid.txt').astype(int)
        for z in range(238,620):
            do_txt = True
            if z not in bid:
                seg = imread(D1+'seg/_s%03d.png'%z)
                if seg.max()>0:
                    im = imread(D1+'im/_s%03d.png'%z)
                    im[seg==0] = 0
                    seg = imread(D1+'mito_pf/_s%03d.png'%(z-238))
                    im[seg>0] = 0
                    imwrite(D1+'im_vesicle/%03d.png'%z, im)
                    do_txt = False
            if do_txt:
                np.savetxt(D1+'im_vesicle/%03d.txt'%z, [0],'%d')
    elif opt == '2.1': #jason 2022/12/06 result
        from imageio import volread
        D1 = D0i + 'terminal/jason_vesicle/'
        opt = 0
        for i,z in enumerate(range(288,620,50)):
            # seg
            if opt == 0:
                seg = imread(D1+'%d_final.tif'%z)
                import pdb; pdb.set_trace()
                mito = imread(D1+'../mito_pf/_s%03d.png'%(z-238))
                seg[mito>0] = 0
                print(seg.max())
                imwrite(D1+'%d_seg.png'%i, seg2Vast(seg))
            elif opt == 1:
                shutil.copy(D1+'../im/_s%03d.png'%z, D1+'%d.png'%i)
            elif opt == 2:
                cell = imread(D1+'../seg/_s%03d.png'%(z))
                imwrite(D1+'%d_cell.png'%i, seg2Vast(cell==17))
    elif opt == '2.2': # xiaomeng pf
        import cc3d
        D1 = D0i + 'terminal/vesicle_pf/'
        zz = range(288,620,50)
        for i,z in enumerate(zz):
            seg = vast2Seg(imread(D1+'_s%d.png'%i))
            mito = imread(D1+'../mito_pf/_s%03d.png'%(z-238))
            seg[mito>0] = 0
            out = cc3d.connected_components(seg)
            print(len(np.unique(seg)), out.max())
            imwrite(D1+'v%03d.png'%zz[i], seg2Vast(out))
    elif opt == '2.3': #jason 2022/12/23 result
        from imageio import volread
        D1 = D0i + 'terminal/jason_vesicle_v2/'
        opt = 0
        for i,z in enumerate(range(288,620)):
            # seg
            z = 488;i=z-238
            #seg = imread(D1+'%d.tif'%z)
            seg = imread(D1+'%d_final.tif'%z)
            mito = imread(D1+'../mito_pf/_s%03d.png'%(z-238))
            seg[mito>0] = 0
            gt = vast2Seg(imread(D1+'../vesicle_pf/v%03d.png'%z))
            print(seg.max(),gt.max())
            import pdb; pdb.set_trace()
            imwrite(D1+'%d_seg.png'%i, seg2Vast(seg))
            import pdb; pdb.set_trace()

elif opt[0] == '3': # export
    if opt == '3': # check range
        from T_util import readVastSeg
        #for name in ['pos']:
        for name in ['neg']:
            dd,nn = readVastSeg(D0i+'meta/meta_%s.txt'%name)
            ii = [x for x in range(len(nn)) if ('finish' not in nn[x] and 'Terminal' in nn[x])]
            # big range
            # coord = np.hstack([dd[ii,-6:-3].min(axis=0),dd[ii,-3:].max(axis=0)])
            # export each terminal
            print(dd[ii,-6:])
            # [nn[x] for x in ii]
    elif opt == '3.1': # check export
        from glob import glob
        for name in ['pos', 'neg']:
            for i in range(10): 
                mm = '/%s%d/'%(name, i) 
                fns = glob(D0i + mm + 'seg/*.png') 
                # check shape
                #print(name,i,imread(fns[0]).shape)
                # check number
                gns = glob(D0i + mm + 'im/*.png') 
                if len(fns) != len(gns):
                    # not saving empty files
                    print(mm, len(fns), len(gns))
    elif opt == '3.11': # neg6 too big
        # wrong meta file: [y,x] 1200,1000
        # orig: 18178 27202 308 36466 34993 511
        # new: 18178 27202 308 19177 28401 511
        from glob import glob
        Dd = D0i + '/%s%d/seg/'%('neg', 5)
        fns = sorted(glob(Dd+'*.png')) 
        bb = [1e6,0,1e6,0]
        for fn in fns:
            im = imread(fn)[::10,::10]
            bb = get_union(bb,get_bb(im)) 
            print(fn,bb)
    elif opt == '3.9': # folder reorder
        nn = 'neg';ii = [2,4,5,7,8,9,10,11,12,13]
        for i,j in enumerate(ii):
            U_mkdir(D0i+'%s%d'%(nn,i))
            shutil.move(D0i+'im_%s%d'%(nn,j), D0i+'%s%d/im'%(nn,i))
            shutil.move(D0i+'seg_%s%d'%(nn,j), D0i+'%s%d/seg'%(nn,i))
elif opt[0] == '4': # vesicle classes
    nn = range(288,589,50)
    DD = D0i + 'pos0/'
    if opt == '4': # take one example
        from T_util import get_bb_label2d_v2 
        DD = D0i + 'pos0/'
        Do = Dw + 'xiaomeng/vesicle_patch/%d_%d.png'
        ww_h = 5
        ww = ww_h * 2 + 1
        for i in range(7):
            im = imread(DD+'im/_s%03d.png'%nn[i])
            pp_fn = DD+'vesicle_pf/v%03d_pp.h5'%nn[i]
            if not os.path.exists(pp_fn):
                bb_fn = DD+'vesicle_pf/v%03d_bb.txt'%nn[i]
                if not os.path.exists(bb_fn):
                    seg = vast2Seg(imread(DD+'vesicle_pf/v%03d.png'%nn[i]))
                    bb = get_bb_label2d_v2(seg)
                    np.savetxt(bb_fn, bb, '%d')
                else:
                    bb = np.loadtxt(bb_fn).astype(int)
                out = np.zeros([bb.shape[0], ww, ww], np.uint8)
                for j in range(bb.shape[0]):
                    bc = (bb[j,1::2]+bb[j,2::2])//2
                    im_p = im[bc[0]-ww_h:bc[0]+ww_h+1,\
                              bc[1]-ww_h:bc[1]+ww_h+1]
                    out[j] = im_p
                    #imwrite(Do%(i,j), im_p)
                writeh5(pp_fn, out)
    elif opt == '4.00': # take 100 examples for refinement
        from T_util import get_bb_label2d_v2 
        DD = D0i + 'pos0/'
        ww_h = 11
        Do = Dw + 'xiaomeng/vesicle_patch_'+str(ww_h)+'/%d_%d'
        ww = ww_h * 2 + 1
        for i in range(7):
            im = imread(DD+'im/_s%03d.png'%nn[i])
            bb_fn = DD+'vesicle_pf/v%03d_bb.txt'%nn[i]
            seg = vast2Seg(imread(DD+'vesicle_pf/v%03d.png'%nn[i]))
            if not os.path.exists(bb_fn):
                bb = get_bb_label2d_v2(seg)
                np.savetxt(bb_fn, bb, '%d')
            else:
                bb = np.loadtxt(bb_fn).astype(int)
            for j in np.linspace(0,bb.shape[0]-1,14).astype(int):
                bc = (bb[j,1::2]+bb[j,2::2])//2
                im_p = im[bc[0]-ww_h:bc[0]+ww_h+1,\
                          bc[1]-ww_h:bc[1]+ww_h+1]
                imwrite(Do%(i,j)+'.png', im_p)
                seg_p = seg[bc[0]-ww_h:bc[0]+ww_h+1,\
                          bc[1]-ww_h:bc[1]+ww_h+1] == bb[j,0]
                imwrite(Do%(i,j)+'_m.png', seg_p.astype(np.uint8))
    elif opt == '4.01': # take one example
        Do = Dw + 'xiaomeng/vesicle_patch/%d_%d.png'
        for i in range(7):
            out = readh5(DD+'vesicle_pf/v%03d_pp.h5'%nn[i])
            for j in range(out.shape[0]):
                if not os.path.exists(Do%(i,j)):
                    imwrite(Do%(i,j), out[j])
    elif opt == '4.1': # sorting by laplacian value
        import cv2
        Do = Dw + 'xiaomeng/vesicle_patch/%d_%d.png'
        ww = 3 
        for i in range(7):
            out = readh5(DD+'vesicle_pf/v%03d_pp.h5'%nn[i]).astype(float)/255
            out_p = np.zeros(out.shape[0])
            for j in range(out.shape[0]):
                if out[j].mean() >0.5:# overall bright
                #if (out[j]<0.3).sum() < 16:# need enough black
                    out_p[j] = -10
                else:
                    th = np.percentile(out[j], [10,90])
                    out[j][out[j]<th[0]] = th[0]
                    out[j][out[j]>th[1]] = th[1]
                    out[j] = (out[j]-out[j].min())/(out[j].max()-out[j].min())
                    #LoG
                    blur = cv2.GaussianBlur(out[j],(7,7),0)
                    laplacian = cv2.Laplacian(blur,cv2.CV_64F)
                    out_p[j] = laplacian[ww:-ww,ww:-ww].mean()
                    #out_p[j] = out[j][ww:-ww,ww:-ww].mean() - (out[j].sum()-out[j][ww:-ww,ww:-ww].sum())/(11**2-(11-2*ww)**2)
            writeh5(DD+'vesicle_pf/v%03d_ppl.h5'%nn[i], out_p)
    elif opt == '4.11': # sorting by laplacian value
        gg, val = [], []
        for i in range(7):
            out_p = list(readh5(DD+'vesicle_pf/v%03d_ppl.h5'%nn[i]))
            val += out_p 
            gg += ['"%d_%d"'%(i,x) for x in range(len(out_p))]
        val = np.array(val)
        ind = np.argsort(val)
        print(','.join([gg[x] for x in ind[val[ind]>-10]]))
        print(','.join([gg[x] for x in ind[val[ind]==-10]]))
        #print(','.join([str(x) for x in ind[out_p[ind]>-10]]))
        #print(','.join([str(x) for x in ind[out_p[ind]==-10]]))
elif opt[0] == '1': # ng
    from em_util.ng import * 
    # create image and seg info
    # xyz
    resolution = [8,8,30]
    Do0 = 'file:///n/holylfs05/LABS/pfister_lab/Everyone/public/ng/xm/'
    Di0 = Dl2 + 'xiaomeng/'
    for mm in ['pos','neg']:
        for mid in range(1):
            nn = mm + str(mid)
            print(nn)
            if opt[:3] == '1.0': 
                # im/cell/mito/vesicle
                # 1.00/1.01[0,1]/1.02[0,1]/1.03
                Di = Di0 + nn+'/'
                Do = Do0 + nn+'_'

                volume_size = np.array(h5py.File(Di+'im.h5','r')['main'].shape)[::-1]
                seg_size = np.ceil(volume_size/[4.0,4.0,1]).astype(int)
                seg_res = [32,32,30]

            chunk_size = [64,64,64]
            mip_ratio_im = [[1,1,1],[2,2,1],[4,4,1],[8,8,2],[16,16,4]]
            mip_ratio_seg = [[1,1,1],[2,2,2],[4,4,4]]
            mip_ratio_ves = [[1,1,1]]
            num_thread = 1

            output_im = Do + 'im'
            output_seg = Do + 'seg'
            output_mito = Do + 'mito'
            output_ves = Do + 'ves'
            U_mkdir(output_im[7:])
            U_mkdir(output_seg[7:])
            U_mkdir(output_mito[7:])
            U_mkdir(output_ves[7:])
            if opt[3] in ['0','3']: # im/vesicle
                dst = ngDataset(volume_size = volume_size, resolution = resolution,\
                         chunk_size=chunk_size, mip_ratio = mip_ratio_im)
                if opt[3] == '0':
                    dst.createInfo(output_im, 'im')
                    im = readh5(Di+'im.h5')
                    def get_im(z0, z1, y0, y1, x0, x1):
                        return im[z0: z1, y0: y1, x0: x1]
                    dst.createTile(get_im, output_im, 'im', range(len(mip_ratio_im)), do_subdir = True, num_thread = num_thread)
                elif opt[3] == '3':
                    # python T_xiaomeng.py 1.03
                    pts = readh5(Di0 + 'ves/' + nn+'_pts.h5').T[:,::-1]
                    print(pts.shape)
                    # zyx -> xyz
                    # physical resolution
                    dst.createSkeleton(pts, output_ves, volume_size, np.array(resolution)*1e-9)
            elif opt[3] in ['1','2']:
                if opt[3] == '1': # cell seg
                    sn = output_seg
                elif opt[3] == '2': # mito seg
                    sn = output_mito
                dst = ngDataset(volume_size = seg_size, resolution = seg_res,\
                    chunk_size=chunk_size, mip_ratio = mip_ratio_seg)
                dst.createInfo(sn, 'seg')
                if opt[4] == '0':
                    if opt[3] == '1':
                        seg = readh5(Di + 'seg_32nm.h5')
                    elif opt[3] == '2':
                        seg = readh5(Di + 'mito_32nm.h5')
                        #seg = readh5(Di + 'mito_32nm_pf.h5')
                    def get_seg(z0, z1, y0, y1, x0, x1):
                        return seg[z0: z1, y0: y1, x0: x1]
                    dst.createTile(get_seg, sn, 'seg', range(len(mip_ratio_seg)), do_subdir = True, num_thread = num_thread)
                elif opt[4] == '1': # igneous env
                    dst.createMesh(sn, 0, seg_size, num_thread, dust_threshold = 0, do_subdir = True)

elif opt[0] == '5': # alzheimer project
    from glob import glob
    if opt == '5.0':
        # 05/18: not good
        D0 = Dl2+ '../Xiaomeng Han/Alzheimer segmentation/30 um box export/'
        fns = sorted(glob(D0+'*.tif'))
        vol = read_vol(fns)
        write_h5('db/xm/alzheimer_30um.h5', vol)
    elif opt == '5.1':
        # 05/22
        D0 = Dl2+ '../Xiaomeng Han/Alzheimer segmentation/30um_box1_8nm/'
        for nn in ['30 um box 1_8nm', 'syn_0-49']:
            fns = sorted(glob(D0+nn+'/*.png'))
            vol = read_vol(fns)
            print(vol.shape)
            if '3' in nn:
                write_h5('db/xm/vol_0522_im.h5', vol)
                break
                write_h5('db/xm/vol_0522_im_0-49.h5', vol[:50])
                write_h5('db/xm/vol_0522_im_50-199.h5', vol[50:])
            else:
                write_h5('db/xm/vol_0522_syn_0-49.h5', vol)
    elif opt == '5.2': # 06/02
        vol = read_vol('db/xm/vol_0522_0-199_pred.h5')
        fn = Dl2 + 'Xiaomeng Han/Alzheimer segmentation/vol_0522_syn_pred/%04d.png'
        for z in range(vol.shape[0]):
            imwrite(fn%z, seg2Vast(vol[z]))
    elif opt == '5.3': # sneha test
        # test data
        D0 = '/n/boslfs02/LABS/lichtman_lab/SHWang/Mona/multitiled/'
        fn = D0 + 'mip1/slice%d-%d/r%d/tile_r%d-c%d.jpg'
        sn = Dl2 + 'xiaomeng/alzheimer/test/%04d/%d_%d.h5'
        num_tile = [6,6,7]
        out = np.zeros([96,6144,7168], np.uint8)
        for z in range(1, 848, 96)[job_id::job_num]:
            mkdir(sn%(z-1, 0, 0), 'parent')
            for r in range(9): 
                for c in range(8): 
                    sno = sn %(z-1, r, c)
                    if not os.path.exists(sno):
                        print(sno)
                        out[:] = 0
                        for zz in range(num_tile[0]): 
                            z0 = z + zz * 16
                            for rr in range(num_tile[1]):
                                for cc in range(num_tile[2]):
                                    out[zz*16: (zz+1)*16,\
                                        rr*1024: (rr+1)*1024,\
                                        cc*1024: (cc+1)*1024\
                                       ] = read_image(fn %(z0, z0+16-1, r*num_tile[1]+rr+1, r*num_tile[1]+rr+1, c*num_tile[2]+cc+1)).reshape([16,1024,1024])
                        if z==769:# last tile
                            write_h5(sno, out[:81])
                        else:
                            write_h5(sno, out)
    elif opt == '5.4': # export results
        from T_util_seg import polarity2instance
        fn = Dl2 + 'xiaomeng/alzheimer/debug/dl_%d_%d.h5'
        sn = Dl2 + 'xiaomeng/alzheimer/debug/result_ins_Ex/'
        sn = Dl2 + 'xiaomeng/alzheimer/debug/result_sem_Ex/'
        z,r,c = 0, 5, 4
        # too much m
        data = read_h5(fn% (r,c))
        print(5*6144,4*7168)
        out = polarity2instance(data, exclusive=True, semantic=True)
        for z in range(out.shape[0]):
            write_image(sn + '%04d.png'%z, seg2Vast(out[z]))
    elif opt == '5.41': # check image
        fn = Dl2 + 'xiaomeng/alzheimer/test/%04d/%d_%d.h5'
        z,r,c = 0, 5, 4
        aa = np.array(h5py.File(fn% (z,r,c),'r')['main'][0])
        write_image('test.png', aa)
    elif opt == '5.42': # decoding
        from T_util_seg import polarity2instance
        Df = 'xm/alzheimer_synapse/' 
        Dim = Dl2 + 'xiaomeng/alzheimer/test/'
        Din = Dl2 + 'xiaomeng/alzheimer/result/'
        Do = Dl2 + 'xiaomeng/alzheimer/result_ins/'
        fns = [x[:-1] for x in readtxt(f'{Df}/test.txt')]
        for fn in fns[job_id::job_num]:
            if os.path.exists(Din+fn) and not os.path.exists(Do+fn):
                im = read_h5(Dim+fn)<=250
                data = read_h5(Din+fn)
                # mask out border regions
                for cid in range(3):
                    data[cid] = data[cid] * im
                del im
                # need to make it unique ...
                out = polarity2instance(data, exclusive=True)
                out_m = np.array([out.max()])
                write_h5(Do+fn, [out, out_m], ['main','max'])
        """
        for z in range(9):
            mkdir(Do+'%04d/'%(z*96))
        """
    elif opt in ['5.5', '5.51', '5.52']: # deep learning 
        Din = Dl2 + 'xiaomeng/alzheimer/test/'
        Do = Dl2 + 'xiaomeng/alzheimer/result/'
        Df = 'xm/alzheimer_synapse/' 
        fin = '%04d/%d_%d.h5'
        job_num = 13
        if opt == '5.5':
            # regular
            fns = []
            for z in range(9):
                #mkdir(fin%(z*96,0,0),'parent')
                for r in range(9): 
                    for c in range(8): 
                        fns.append(fin %(z*96,r,c))
            writetxt(f'{Df}/test.txt', fns)
            #writetxt(Dl2 + 'xiaomeng/alzheimer/test.txt', fns)
        if opt == '5.51': # rename files
            import glob
            for z in range(9):
                gn0 = Do+'/%04d/'%(z*96)
                gns = glob.glob(gn0+'result_*.h5') 
                for gn in gns:
                    try:
                        sz = get_volume_size_h5(gn)
                        idx = int(gn[gn.rfind('_')+1:-3])
                        shutil.move(gn, gn0+fns[idx])
                    except:
                        print('file error:', gn)
        if opt == '5.52': # write slurm jobs
            #  mmg 1 100000 1 gpu_requeue
            for i in range(job_num):
                tmp = f"""#!/bin/bash -e
#SBATCH --job-name=xm_syn # job name
#SBATCH -N 1 # how many nodes to use for this job
#SBATCH -n 1 # how many CPU-cores (per node) to use for this job
#SBATCH --gres=gpu:1
#SBATCH --mem=150GB # how much RAM (per node) to allocate
#SBATCH -t 3-00:00 # job execution time limit formatted hrs:min:sec
#SBATCH --partition=gpu_requeue # see sinfo for available partitions
#SBATCH -e out.%N.%j.err # STDERR
#SBATCH -o out.%N.%j.out

module load cuda/12.0.1-fasrc01 cudnn/8.8.0.121_cuda12-fasrc01
cd /n/home04/donglai/lib/seg/pytorch_connectomics
/n/home04/donglai/miniconda3/envs/pytc/bin/python -u scripts/main.py --inference \
--config-base /n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/donglai/eng/xm/alzheimer_synapse/JWR15-Synapse-Base.yaml \
--config-file /n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/donglai/eng/xm/alzheimer_synapse/JWR15-Synapse-BCE.yaml \
--checkpoint /n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/donglai/eng/xm/alzheimer_synapse/checkpoint_225000.pth.tar \
        INFERENCE.DO_SINGLY True INFERENCE.DO_CHUNK_TITLE 0 INFERENCE.IMAGE_NAME /n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/donglai/eng/xm/alzheimer_synapse/test.txt  INFERENCE.DO_SINGLY_START_INDEX {i+432} INFERENCE.DO_SINGLY_STEP {job_num}
                """
                writetxt(f'{Df}/slurm_{i}_{job_num}.sh', tmp)
    elif opt[:3] == '5.6': # bbox decoding
        Din = Dl2 + 'xiaomeng/alzheimer/result/'
        Do = Dl2 + 'xiaomeng/alzheimer/result_ins/'
        Dom = Dl2 + 'xiaomeng/alzheimer/result_ins_merge/'
        Df = 'xm/alzheimer_synapse/' 
        fns = [x[:-1] for x in readtxt(f'{Df}/test.txt')]
        fn_dict = dict(zip(fns, range(len(fns))))
 
        if opt == '5.6': # count system
            count = np.zeros([1+72*9])
            for i,fn in enumerate(fns):
                count[1+i] = read_h5(Do+fn, ['max'])
                """
                if os.path.exists(Do+fn) and len(h5py.File(Do+fn,'r').keys()) == 1:
                    print(fn)
                    #os.remove(Do+fn)
                """
            acc = np.cumsum(count).astype(np.uint32)
            write_h5(Do+'count.h5', acc)
        elif opt == '5.61': # compute intersection
            count = read_h5(Do+'count.h5')
            fn0 = '%04d/%d_%d.h5'
            """
            for z in range(9):
                mkdir(Dom + '%04d/'%(z*96))
            """
            def merge_syn_ins(seg0, seg, count0, count):
                # convert syn pre/post label to the same label 
                if seg0.max() != 0 and seg.max() != 0:
                    ind1, ind2 = seg0%2==0, seg0%2==1
                    seg0[ind1], seg0[ind2] = seg0[ind1]//2, (seg0[ind2]+1)//2
                    ind1, ind2 = seg%2==0, seg%2==1
                    seg[ind1], seg[ind2] = seg[ind1]//2, (seg[ind2]+1)//2
                    # find unique non-zero pair
                    mm0 = np.unique(np.hstack([seg0.reshape(-1,1), seg.reshape(-1,1)]), axis=0)
                    mm0 = mm0[mm0.min(axis=1)!=0] # remove non-zero pairs
                    if len(mm0) != 0:
                        # convert back to pre-pre, post-post matching
                        mm0 = np.vstack([mm0*2, mm0*2-1]).astype(count.dtype)
                        mm0[:, 0] += count0
                        mm0[:, 1] += count
                    return mm0
                else:
                    return np.zeros([0,2])
 
            for fn in fns[1:][job_id::job_num]:
                if not os.path.exists(Dom+fn):
                    print(fn)
                    mm = np.zeros([0,2], np.uint32)
                    if read_h5(Do+fn, ['max']) != 0:
                        zz, rc = fn.split('/')
                        rr, cc = rc[:-3].split('_')
                        zz, rr, cc = int(zz), int(rr), int(cc)
                        if zz != 0:
                            sn = fn0 % (zz-96,rr,cc)
                            seg0 = np.array(h5py.File(Do+sn, 'r')['main'][-1])
                            seg = np.array(h5py.File(Do+fn, 'r')['main'][0])
                            mm = np.vstack([mm, merge_syn_ins(seg0, seg, count[fn_dict[sn]], count[fn_dict[fn]])])
                           
                        if rr != 0:
                            sn = fn0 % (zz,rr-1,cc)
                            seg0 = np.squeeze(np.array(h5py.File(Do+sn, 'r')['main'][:, -1]))
                            seg = np.squeeze(np.array(h5py.File(Do+fn, 'r')['main'][:, 0]))
                            mm = np.vstack([mm, merge_syn_ins(seg0, seg, count[fn_dict[sn]], count[fn_dict[fn]])])
     
                        if cc != 0:
                            sn = fn0 % (zz,rr,cc-1)
                            seg0 = np.squeeze(np.array(h5py.File(Do+sn, 'r')['main'][:,:,-1]))
                            seg = np.squeeze(np.array(h5py.File(Do+fn, 'r')['main'][:,:,0]))
                            mm = np.vstack([mm, merge_syn_ins(seg0, seg, count[fn_dict[sn]], count[fn_dict[fn]])])
                    write_h5(Dom+fn, mm)
        elif opt == '5.62': # relabel
            import scipy.sparse
            rl = UnionFind()
            mid = 0
            for fn in fns[1:]:
                mm = read_h5(Dom+fn).astype(np.uint32)
                if len(mm) > 0:
                    rl.union_arr(mm)
                    mid = max([mid, mm.max()])
            rl_arr = np.arange(mid+1, dtype=np.uint32)
            for x in rl.components():
                rl_arr[list(x)] = min(x)
            write_h5(Do+'relabel.h5', rl_arr)
        elif opt == '5.621': # count unique
            for fn in fns[job_id::job_num]:
                sn = Do+fn[:-3]+'_uid.h5'
                if not os.path.exists(sn):
                    print(sn)
                    seg0 = read_h5(Do+fn, ['main'])
                    uid = np.unique(seg0)
                    write_h5(sn, uid[uid>0])
        elif opt == '5.622': # count unique
            rl_arr = read_h5(Do+'relabel.h5')
            rl_max = len(rl_arr)
            count = read_h5(Do+'count.h5')
            out = [None] * len(fns)
            for i,fn in enumerate(fns):
                seg0 = read_h5(Do+fn[:-3]+'_uid.h5')
                ind1, ind2 = seg0%2==0, seg0%2==1
                seg0[ind1], seg0[ind2] = seg0[ind1]//2, (seg0[ind2]+1)//2
                out[i] = 2*np.unique(seg0) + count[fn_dict[fn]]
            out = np.hstack(out)
            out[out < rl_max] = rl_arr[out[out < rl_max]]
            out = np.unique(out)
            print(len(out))
            write_h5(Do+'uid.h5', out)




        elif opt == '5.63':# upload to tensorstore
            # create bucket on the browser
            # srun --pty -p serial_requeue -t 1-00:00 --mem 200000 -n 4 -N 1 /bin/bash
            # ssh tunnel 8085
            # gcloud auth application-default login
            # screen
            # sa tensorstore
            # naive multiple-process: for i in {0..3};do sleep 5;python T_xiaomeng.py 5.63 ${i} 4 & done
            import tensorstore as ts
            out_dir = 'gs://wei_lab/vclem_xh/alzheimers/syn_20241121'
            vol_type = 'segmentation'
            dtype = 'uint32'
            XMAX, YMAX, ZMAX = 57344, 55296, 849
            resolution, thickness = 8, 30
            encoding = 'compressed_segmentation'
            ratios = [[1,1,1],[4,4,1],[8,8,2]]
            out_ts = [None] * len(ratios)
            datasets = [None] * len(ratios)
            sz = [[7168,6144,96]] * len(ratios)
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
                    "size": [(XMAX+ratio[0]-1)//ratio[0], (YMAX+ratio[1]-1)//ratio[1], (ZMAX+ratio[2]-1)//ratio[2]],
                    "encoding": encoding,
                    "chunk_size": [128, 128, 16],
                    "resolution": [resolution*ratio[0], resolution*ratio[1], thickness*ratio[2]],
                    'compressed_segmentation_block_size': [8, 8, 8]
                    }
                }
                datasets[i] = ts.open(out_ts[i]).result()
                sz[i] = np.array(sz[i]) // ratio

            fn0 = '%04d/%d_%d'
            acc = read_h5(Do+'count.h5').astype(np.uint32)
            rl_arr = read_h5(Do+'relabel.h5')
            rl_max = len(rl_arr)
            cc = 0
            TS_TIMEOUT=180
            z_num = 3
            for z in range(9):
                for r in range(9): 
                    for c in range(8): 
                        cc = z*9*8 + r*8 + c
                        if cc % job_num == job_id:
                            done = True
                            for i in range(len(ratios)):
                                if i == 0:
                                    for ii in range(z_num):
                                        done = done and os.path.exists(Do+fn0%(z*96,r,c)+'_%d-%d.txt'%(i,ii))
                                else:
                                    done = done and os.path.exists(Do+fn0%(z*96,r,c)+'_%d.txt'%(i))
                            if not done:
                                print(z,r,c)
                                syn_max = read_h5(Do+fn0%(z*96,r,c)+'.h5', ['max'])
                                if syn_max==0:
                                    for i in range(len(ratios)):
                                        if i == 0:
                                            for ii in range(z_num):
                                                done = Do+fn0%(z*96,r,c)+'_%d-%d.txt'%(i,ii)
                                                np.savetxt(done, [0], '%d')
                                        else:
                                            done = Do+fn0%(z*96,r,c)+'-%d.txt'%(i)
                                            np.savetxt(done, [0], '%d')
                                    continue
                                syn = read_h5(Do+fn0%(z*96,r,c)+'.h5', ['main']).astype(np.uint32)
                                syn[syn > 0] += acc[cc]
                                syn[syn < rl_max] = rl_arr[syn[syn < rl_max]]
                                syn = syn.transpose([2,1,0])[:,:,:,None]
                                for i, ratio in enumerate(ratios):
                                    z0, z1 = sz[i][2]*z, min(sz[i][2]*(z+1), (ZMAX+ratio[2]-1)//ratio[2])
                                    zstep = z_num if i==0 else 1
                                    zran = np.linspace(z0, z1, zstep+1).astype(int)
                                    for ii in range(zstep):
                                        done = Do+fn0%(z*96,r,c)+'_%d.txt'%(i) if i!=0 else Do+fn0%(z*96,r,c)+'_%d-%d.txt'%(i,ii)
                                        if not os.path.exists(done):
                                            dataview = datasets[i][sz[i][0]*c:sz[i][0]*(c+1), \
                                                                   sz[i][1]*r:sz[i][1]*(r+1), \
                                                                   zran[ii]: zran[ii+1]]
                                            for j in range(3):
                                                print('trial %d-%d'%(ii,j))
                                                try:
                                                    dataview.write(syn[::ratio[0],::ratio[1],::ratio[2]][:,:,zran[ii]-z0: zran[ii+1]-z0]).result(timeout=TS_TIMEOUT)
                                                    np.savetxt(done, [0], '%d')
                                                    break
                                                except:
                                                    datasets[i] = ts.open(out_ts[i]).result()
                                                    dataview = datasets[i][sz[i][0]*c:sz[i][0]*(c+1), \
                                                                       sz[i][1]*r:sz[i][1]*(r+1), \
                                                                       zran[ii]: zran[ii+1]]

        elif opt == '5.631':# change mip0
            fn0 = '%04d/%d_%d'
            for z in range(9):
                for r in range(9): 
                    for c in range(8): 
                        fn = Do+fn0%(z*96,r,c)+'_%d.txt'%(0)
                        if os.path.exists(fn):
                            for ii in range(3):
                                np.savetxt(Do+fn0%(z*96,r,c)+'_%d-%d.txt'%(0,ii), [0], '%d')
                            os.remove(fn)
elif opt[0] == '6': # 2020407 project
    D1 = f'{D0i}/202407/' 
    if opt == '6':
        import glob
        # visualize seg
        fns = sorted(glob.glob(f'{D1}cell_red/*.png'))
        seg = read_image_folder(fns[::2], image_type='seg', ratio=[0.125,0.125])
        write_h5(f'{D1}cell_red_64nm.h5', seg)
    elif opt[:3] == '6.0':
        from em_util.ng import NgDataset
        Dw = 'file:///n/holylfs05/LABS/pfister_lab/Everyone/public/ng/'
        # xyz
        wn='xm_202407'
        volume_size = [1394, 1412, 551]
        resolution = [64, 64, 60]
        mip_ratio = [[1,1,1],[2,2,2],[4,4,4]]
        output_seg = f'{Dw}{wn}/cell0/'
        dst = NgDataset(volume_size, resolution, mip_ratio,offset=[6391,4840,0])
        if opt == '6.01': # cloudvolume env: make segmentation tiles
            vol = read_h5(f'{D0i}/202407/cell_red_64nm.h5')
            dst.create_info(output_seg, 'seg')
            def get_seg(z0, z1, y0, y1, x0, x1, mip):
                return vol[z0: z1:mip[2], y0: y1:mip[1], x0: x1: mip[0]]
            dst.create_tile(get_seg, output_seg, 'seg', range(len(mip_ratio)))
        elif opt == '6.02': # cloudvolume env: make segmentation tiles
            dst.create_mesh(output_seg, 0, volume_size, do_subdir = True)
