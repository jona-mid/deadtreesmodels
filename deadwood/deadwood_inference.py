import json
from os.path import join
from pathlib import Path

import numpy as np
import safetensors.torch
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import crop
from tqdm import tqdm
from shapely.ops import unary_union

from common import *

from .InferenceDataset import InferenceDataset


class DeadwoodInference:
    def __init__(self, config_path):
        # set float32 matmul precision for higher performance
        torch.set_float32_matmul_precision("high")

        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.load_model()

    def load_model(self):
        if "segformer_b5" in self.config["model_name"]:
            model = smp.Unet(
                encoder_name="mit_b5",
                encoder_weights="imagenet",
                in_channels=3,
                classes=1,
            ).to(memory_format=torch.channels_last)

            model = torch.compile(model)
            safetensors.torch.load_model(
                model,
                join(
                    str(Path(__file__).parent.parent),
                    "data",
                    self.config["model_name"] + ".safetensors",
                ),
            )
            # model = nn.DataParallel(model)
            model = model.to(memory_format=torch.channels_last, device=self.device)

            model.eval()

            self.model = model

        else:
            print("Invalid model name: ", self.config["model_name"], "Exiting...")
            exit()

    def _write_to_tile_buffers(self, output_tile, nodata_mask_tile, cropped_window, 
                              dataset, tile_buffers, idx):
        """
        Write a single processed tile to all relevant tile buffers.
        """
        # Crop tensor by dataset padding
        output_tile = crop(
            output_tile,
            top=dataset.padding,
            left=dataset.padding,
            height=dataset.tile_size - (2 * dataset.padding),
            width=dataset.tile_size - (2 * dataset.padding),
        )

        nodata_mask_tile = crop(
            nodata_mask_tile,
            top=dataset.padding,
            left=dataset.padding,
            height=dataset.tile_size - (2 * dataset.padding),
            width=dataset.tile_size - (2 * dataset.padding),
        )

        # Get tile position in full image coordinates
        minx = cropped_window["col_off"][idx]
        maxx = minx + cropped_window["width"][idx]
        miny = cropped_window["row_off"][idx]
        maxy = miny + cropped_window["height"][idx]

        # Determine which tiles this inference tile intersects
        for (tile_x, tile_y), tile_info in tile_buffers.items():
            # Check if inference tile intersects this output tile
            if (maxx <= tile_info['start_x'] or minx >= tile_info['end_x'] or
                maxy <= tile_info['start_y'] or miny >= tile_info['end_y']):
                continue
                
            tile_mask = tile_info['mask']

            # Convert to tile-relative coordinates
            tile_minx = minx - tile_info['start_x']
            tile_maxx = maxx - tile_info['start_x']
            tile_miny = miny - tile_info['start_y']
            tile_maxy = maxy - tile_info['start_y']

            # Clip to tile boundaries
            diff_minx = max(0, -tile_minx)
            diff_miny = max(0, -tile_miny)
            diff_maxx = max(0, tile_maxx - tile_mask.shape[1])
            diff_maxy = max(0, tile_maxy - tile_mask.shape[0])

            clip_minx = max(0, tile_minx)
            clip_miny = max(0, tile_miny)
            clip_maxx = min(tile_mask.shape[1], tile_maxx)
            clip_maxy = min(tile_mask.shape[0], tile_maxy)

            # Skip if no actual overlap after clipping
            if clip_minx >= tile_mask.shape[1] or clip_maxx <= 0 or \
               clip_miny >= tile_mask.shape[0] or clip_maxy <= 0:
                continue

            # Crop output tile to the correct size for this output tile
            cropped_output = output_tile[
                :,
                diff_miny : output_tile.shape[1] - diff_maxy,
                diff_minx : output_tile.shape[2] - diff_maxx,
            ]

            cropped_nodata = nodata_mask_tile[
                diff_miny : nodata_mask_tile.shape[0] - diff_maxy,
                diff_minx : nodata_mask_tile.shape[1] - diff_maxx,
            ]

            tile_data = cropped_output[0].numpy()

            # Threshold the output
            tile_data = (tile_data > self.config["probabilty_threshold"]).astype(bool)

            # Apply nodata mask
            tile_data[~cropped_nodata] = 0

            # Write to tile mask
            tile_mask[clip_miny:clip_maxy, clip_minx:clip_maxx] = tile_data

    def inference_deadwood(self, input_tif):
        """
        Gets path to tif file and returns polygons of deadwood in the CRS of the tif.
        Uses optimized tile processing with overlap to handle large datasets efficiently.
        """
        # Will always return a vrt, even when not reprojecting
        vrt_src = image_reprojector(
            input_tif, min_res=self.config["deadwood_minimum_inference_resolution"]
        )

        dataset = InferenceDataset(
            image_src=vrt_src,
            tile_size=1024,
            padding=256,
            brightness_factor=self.config.get("brightness_factor", 1.5),
            contrast_factor=self.config.get("contrast_factor", 1.3),
        )

        # Calculate tile size based on sqrt of mask_tiling_threshold
        tile_size_inference = int(np.sqrt(self.config["mask_tiling_threshold"]))
        # Use small overlap for tile boundaries (independent of inference tile padding)
        tile_overlap = 32
        
        num_tiles_x = (dataset.width + tile_size_inference - tile_overlap - 1) // (tile_size_inference - tile_overlap) + 1
        num_tiles_y = (dataset.height + tile_size_inference - tile_overlap - 1) // (tile_size_inference - tile_overlap) + 1
        
        print(f"Processing with {tile_size_inference}×{tile_size_inference} pixel tiles, overlap: {tile_overlap} pixels ({num_tiles_x}×{num_tiles_y} = {num_tiles_x * num_tiles_y} tiles)")

        loader_args = {
            "batch_size": self.config["batch_size"],
            "num_workers": self.config["num_dataloader_workers"],
            "pin_memory": True,
            "shuffle": False,
        }
        inference_loader = DataLoader(dataset, **loader_args)

        # Initialize tile buffers with overlap info
        tile_buffers = {}
        all_polygons = []
        processed_tiles = set()
        
        # Progress bar for inference
        pbar = tqdm(total=len(inference_loader), desc="Inference and vectorization", delay=1.0)

        for nodata_mask, images, cropped_windows in inference_loader:
            
            # Skip if there is no data
            if not nodata_mask.any():
                pbar.update(1)
                continue

            images = images.to(device=self.device, memory_format=torch.channels_last)

            # Run inference ONCE per batch
            with torch.no_grad():
                if images.shape[0] < self.config["batch_size"]:
                    pad = torch.zeros(
                        (self.config["batch_size"], 3, 1024, 1024), dtype=torch.float32
                    )
                    pad[: images.shape[0]] = images
                    pad = pad.to(device=self.device, memory_format=torch.channels_last)
                    output = self.model(pad)
                    output = output[: images.shape[0]]
                else:
                    output = self.model(images)

                output = torch.sigmoid(output)

            # Determine which tiles this batch affects
            batch_min_x = dataset.width
            batch_max_x = 0
            batch_min_y = dataset.height
            batch_max_y = 0
            
            for i in range(output.shape[0]):
                minx = cropped_windows["col_off"][i]
                maxx = minx + cropped_windows["width"][i]
                miny = cropped_windows["row_off"][i]
                maxy = miny + cropped_windows["height"][i]
                
                batch_min_x = min(batch_min_x, minx)
                batch_max_x = max(batch_max_x, maxx)
                batch_min_y = min(batch_min_y, miny)
                batch_max_y = max(batch_max_y, maxy)

            # Initialize any new tile buffers needed
            for tile_y in range(num_tiles_y):
                for tile_x in range(num_tiles_x):
                    start_x = tile_x * (tile_size_inference - tile_overlap)
                    end_x = min(start_x + tile_size_inference, dataset.width)
                    start_y = tile_y * (tile_size_inference - tile_overlap)
                    end_y = min(start_y + tile_size_inference, dataset.height)
                    
                    # Skip if tile is completely outside image bounds
                    if start_x >= dataset.width or start_y >= dataset.height:
                        continue
                    
                    # Skip if tile has zero or negative dimensions
                    current_tile_width = end_x - start_x
                    current_tile_height = end_y - start_y
                    if current_tile_width <= 0 or current_tile_height <= 0:
                        continue
                    
                    # Check if this tile intersects with the current batch
                    if end_x <= batch_min_x or start_x >= batch_max_x or \
                    end_y <= batch_min_y or start_y >= batch_max_y:
                        continue
                        
                    if (tile_x, tile_y) not in tile_buffers:
                        tile_buffers[(tile_x, tile_y)] = {
                            'mask': np.zeros((current_tile_height, current_tile_width), dtype=bool),
                            'start_x': start_x,
                            'end_x': end_x,
                            'start_y': start_y,
                            'end_y': end_y
                        }

            # Write each inference tile to relevant output tiles (process each ONCE)
            for i in range(output.shape[0]):
                output_tile = output[i].cpu()
                nodata_mask_tile = nodata_mask[i]
                
                self._write_to_tile_buffers(
                    output_tile, nodata_mask_tile, cropped_windows,
                    dataset, tile_buffers, i
                )

            # Vectorize and clear completed tiles
            # A tile is complete when the current batch is beyond its bounds
            tiles_to_process = []
            for (tile_x, tile_y) in list(tile_buffers.keys()):
                if (tile_x, tile_y) in processed_tiles:
                    continue
                    
                tile_info = tile_buffers[(tile_x, tile_y)]
                # If current batch is completely past this tile in both dimensions, it's safe to process
                if batch_min_y >= tile_info['end_y'] and batch_min_x >= tile_info['end_x']:
                    tiles_to_process.append((tile_x, tile_y))
            
            for tile_coords in sorted(tiles_to_process):
                tile_info = tile_buffers[tile_coords]
                
                # Vectorize the tile
                tile_polygons = mask_to_polygons(
                    tile_info['mask'],
                    dataset.image_src,
                    offset_x=tile_info['start_x'],
                    offset_y=tile_info['start_y'],
                )
                all_polygons.extend(tile_polygons)
                
                # Clear the tile from memory
                del tile_buffers[tile_coords]
                processed_tiles.add(tile_coords)

            pbar.update(1)

        # Process any remaining tiles
        for tile_coords in sorted(tile_buffers.keys()):
            if tile_coords in processed_tiles:
                continue
                
            tile_info = tile_buffers[tile_coords]
            
            tile_polygons = mask_to_polygons(
                tile_info['mask'],
                dataset.image_src,
                offset_x=tile_info['start_x'],
                offset_y=tile_info['start_y'],
            )
            all_polygons.extend(tile_polygons)

        pbar.close()

        # Close the vrt
        vrt_src.close()

        print("Dissolving overlapping polygons...")
        # Fix invalid geometries and dissolve overlapping polygons from tile boundaries
        if len(all_polygons) > 0:
            # Clean invalid geometries using buffer(0)
            valid_polygons = []
            for poly in tqdm(all_polygons, desc="validating geometries"):
                if poly.is_valid:
                    valid_polygons.append(poly)
                else:
                    # Try to fix invalid geometries
                    fixed = poly.buffer(0)
                    if fixed.is_valid and not fixed.is_empty:
                        if fixed.geom_type == 'Polygon':
                            valid_polygons.append(fixed)
                        elif fixed.geom_type == 'MultiPolygon':
                            valid_polygons.extend(list(fixed.geoms))
                        
            # Dissolve overlapping polygons
            try:
                dissolved_geom = unary_union(valid_polygons)
                
                # Convert back to list of polygons
                if dissolved_geom.geom_type == 'Polygon':
                    polygons = [dissolved_geom]
                elif dissolved_geom.geom_type == 'MultiPolygon':
                    polygons = list(dissolved_geom.geoms)
                else:
                    polygons = valid_polygons  # Fall back to original if something went wrong
            except Exception as e:
                print(f"Warning: Dissolve failed with error: {e}")
                print("Continuing without dissolving polygons...")
                polygons = valid_polygons
        else:
            polygons = []

        polygons = filter_polygons_by_area(
            polygons, self.config["minimum_polygon_area"]
        )

        # Reproject the polygons back into the crs of the input tif
        polygons = reproject_polygons(
            polygons, dataset.image_src.crs, rasterio.open(input_tif).crs
        )

        return polygons
