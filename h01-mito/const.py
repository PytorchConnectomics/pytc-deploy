import numpy as np

# H01 dataset parameter    
# neuron mask: low-res mip5 (128x128x132)
# mito: high-res mip1 (8x8x33)
# tile_size: [25,128,128] -> [100,2048,2048] (similar #voxel of one slice)

neuron_volume_size = [1324,15552,27072]
neuron_volume_offset = [0,2560,3520]
neuron_tile_size = np.array([25,128,128])
mito_volume_ratio = [4,16,16]
mito_tile_size = neuron_tile_size * mito_volume_ratio
num_tile = (neuron_volume_size+neuron_tile_size-1) // neuron_tile_size

neuron_id = [590612150, 36750893213]