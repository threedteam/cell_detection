This part is the code of the whole project. BUT what you need to do before running the project is:
  --Configure the environment of MMDetection.
  --Put the COCO dataset in the "data" subdirectory under the root directory and delete the empty file. The structure of "data" directory should be:
	|--data
	|   |--coco
	|       |--annotations
	|           |--instances_train2017.json
	|           |--instances_val2017.json
	|           |--instances_test2017.json
	|       |--images
	|           |--train2017
	|               |--carcinoma_in_situ_0.BMP
	|               | ……
	|           |--val2017
	|               |--carcinoma_in_situ_57.BMP
	|               | ……
	|           |--test2017
	|               |--carcinoma_in_situ_8.BMP
	|               | ……
