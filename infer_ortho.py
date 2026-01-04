from deadwood.deadwood_inference import DeadwoodInference
import rasterio
from common.common import *

#filename = "/scratch/cmosig/test_image_seg/20211001_FVA_Walddrohnen_Totholz_3_ortho.tif"
filename = "/mnt/gsdata/projects/deadtrees/data_bayrischer_wald_heurich/RGB Mosaik 20110822.tif"

deadwodinference = DeadwoodInference(config_path="deadwood_inference_config.json")
polygons = deadwodinference.inference_deadwood(filename)
save_poly("deadwood_test.gpkg", polygons, crs=rasterio.open(filename).crs)
