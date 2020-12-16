These codes are used to automatically annotate images and generate JSON files for other steps such as generating COCO dataset.

Description of each code:
  img2json.py: convert images to serialized data and store as JSON files.
  findcont.py: run this file to look for outline points of every image.
  imitate_json.py: generate JSON files that can be parsed by LabelMe interface, which also will be used by code/dataset balance/labelme2coco_datasetBalance.py
