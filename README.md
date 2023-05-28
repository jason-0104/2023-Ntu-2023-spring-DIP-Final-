# DIP 2023 Spring Final project
Applied Digital Image Prcessing at National Taiwan University (NTU) CSIE
# DIP 2023 Spring Final project

## Environment
Since the need of packages of this project isn't a lot and isn't complex, so we list all the package we use and you can install youselves by pip install.
- numpy
- argparse
- cv2 
- matplotlib
- pathlib
- PIL
- math

## Parameters
```shell
usage: noise_removal.py [-h] --input INPUT [--impulse IMPULSE]
                        [--salt_pepper SALT_PEPPER] [--gaussian GAUSSIAN]
                        [--ss SS] [--sr SR] [--si SI] [--sj SJ]
                        [--output_dir OUTPUT_DIR]
```

## Recommand setting of parameters
* **impulse**: impulse noise rate [0,1)
* **salt_pepper**: salt and pepper noise rate [0,1)
* **gaussian**: gaussian nosie recommand: [10,50]
* **ss**: sigma in ws, Recommand: 0.5 for salt and pepper or impulse noise, 5 for gaussian noise amd mix noise
* **sr**: sigma in wr, Recommand: 2 times for your gaussian sigma (amplitude)
* **sj**: sigma in wi, Recommand: [25,55]