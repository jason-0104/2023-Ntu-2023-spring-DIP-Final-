import numpy as np
from argparse import ArgumentParser, Namespace
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
def add_impulse_noise(im, p):
    im2 = im.copy()
    n = np.random.randint(0, 256, im.shape)
    sel = np.random.rand(*im.shape) < p # uniform distribution , [0,1)
    im2[sel] = n[sel]
    return im2

def add_salt_and_pepper_noise(img_in, prob):
    distribution_map = np.random.uniform(0, 1, img_in.shape)
    res = np.copy(img_in)
    row, col = img_in.shape
    adjprob = prob/4 #paper probability
    # two-tail test-alike rejection interval
    for i in range(row):
        for j in range(col):
            if distribution_map[i, j] < adjprob:
                res[i, j] = 0
            elif distribution_map[i, j] > 1 - adjprob: 
                res[i, j] = 255
    return res
def gen_gaussian_noise(img_in, mu, sigma, amp):
    return img_in + amp * np.random.normal(mu, sigma, img_in.shape)
def Road_value(im_gray, radius=3):
    h, w = im_gray.shape
    if radius == 1:
        m = 5
    else:
        m = radius * radius // 3
    road = np.zeros(im_gray.shape, np.int32)
    for i in np.arange(radius, h-radius):
        for j in np.arange(radius, w-radius):
            copy_img1 = im_gray[i, j]
            ro = []
            for x in np.arange(-radius, radius+1):
                for y in np.arange(-radius, radius+1):
                    nei_x = i + x
                    nej_y = j + y
                    copy_img2 = im_gray[nei_x, nej_y]
                    ro.append(abs(int(copy_img2)-copy_img1))
            ro = np.sort(np.array(ro))
            road[i, j] = int(np.sum(ro[:m]))
    return road

def padding(img, filter):
    w,h = img.shape[0],img.shape[1]
    pad = np.zeros((w + filter // 2 * 2, h + filter // 2 * 2), np.int)
    for i in np.arange(filter // 2, w + filter // 2):
        for j in np.arange(filter // 2, h + filter // 2):
            pad[i, j] = img[i - filter // 2,j - filter // 2]
    return pad

def median(img, filter):
    result = np.zeros(img.shape)
    pad = padding(img, filter)
    w,h = result.shape[0],result.shape[1]
    for i in range(w):
        for j in range(h):
            ifil = i + filter
            jfil = j + filter
            result[i, j] = np.median(pad[i: ifil, j: jfil])
    return result


def trilateral_filter(im_gray, road, radius, ss, sr, si, sj):
    im2 = im_gray.copy()
    h, w = im2.shape

    for i in np.arange(radius, h-radius):
        for j in np.arange(radius, w-radius):
            copy_img1 = im2[i, j]
            sum = 0
            wall = 0
            for x in np.arange(-radius, radius+1):
                for y in np.arange(-radius, radius+1):
                    nei_x = i + x
                    nej_y = j + y
                    copy_img2 = im2[nei_x, nej_y]
                    a = (nei_x-i)*(nei_x-i) + (nej_y-j)*(nej_y-j)
                    b = (int(copy_img2) - copy_img1)*(int(copy_img2)-copy_img1)
                    ws = np.exp(-a / 2 / (ss * ss))
                    wr = np.exp(-b / 2 / (sr * sr)) 
                    c = road[nei_x, nej_y] * road[nei_x, nej_y]
                    d = (road[nei_x, nej_y] + road[i, j])**2
                    wi = np.exp(-c / 2 / (si * si))
                    J = 1 - np.exp(-d/8/(sj * sj))
                    weight = ws * (wr**(1-J)) * (wi**J)
                    sum += weight * copy_img2
                    wall += weight
            im2[i, j] = np.round(sum / wall).astype(np.int32)
            # if i < 10 and j < 10:
            #     print(i, j, sum_v, w_v, im2[i, j], im_gray[i, j])
    im2 = np.clip(im2, 0, 255).astype(np.uint8)
    return im2

def bilateral_filter(im_gray, radius, ss, sr):
    im2 = im_gray.copy()
    h, w = im2.shape
    for i in np.arange(radius, h-radius):
        for j in np.arange(radius, w-radius):
            copy_img1 = im2[i, j]
            sum = 0
            wall = 0
            for x in np.arange(-radius, radius+1):
                for y in np.arange(-radius, radius+1):
                    nei_x = i + x
                    nej_y = j + y
                    copy_img2 = im2[nei_x, nej_y]
                    a = (nei_x-i)*(nei_x-i) + (nej_y-j)*(nej_y-j)
                    b = (int(copy_img2) - copy_img1)*(int(copy_img2)-copy_img1)
                    ws = np.exp(-a / 2 / (ss * ss))
                    wr = np.exp(-b / 2 / (sr * sr)) 
                    weight = ws * wr
                    sum += weight * copy_img2
                    wall += weight
            im2[i, j] = np.round(sum / wall).astype(np.int32)
    im2 = np.clip(im2, 0, 255).astype(np.uint8)
    return im2



from math import log10, sqrt
def PSNR(img, clean_result):
    MSE = np.mean((img - clean_result) ** 2)
    if(MSE == 0): 
        return 100
    psnr = 20 * log10(255.0 / sqrt(MSE))
    return psnr

def main(args):
    file = args.input
    im = cv2.imread(file,0)
    img_res = im
    bifilter_file_name = 'bi_'
    trifilter_file_name = 'tri_'
    medfilter_file_name = 'med_'
    nosie_img = 'noise_'
    if args.impulse != 0:
        print("Adding impulse noise...")
        img_res = add_impulse_noise(img_res, args.impulse)
        bifilter_file_name += "imp_" + str(args.impulse) + "_"
        trifilter_file_name +=  "imp_" + str(args.impulse) + "_"
        medfilter_file_name +=  "imp_" + str(args.impulse) + "_"
        nosie_img +=  "imp_" + str(args.impulse) + "_"
        
    if args.salt_pepper != 0.0:
        print("Adding salt & pepper noise...")
        img_res = add_salt_and_pepper_noise(img_res, args.salt_pepper)
        bifilter_file_name += "pepper_" + str(args.salt_pepper) + "_"
        trifilter_file_name +=  "pepper_" + str(args.salt_pepper) + "_"
        medfilter_file_name +=  "pepper_" + str(args.salt_pepper) + "_"
        nosie_img +=  "pepper_" + str(args.salt_pepper) + "_"
    #show_img4(im, im_salt_and_pepper, im_salt_and_pepper2, im_salt_and_pepper3)
    if args.gaussian != 0:
        print("Adding gaussian noise...")
        img_res = gen_gaussian_noise(img_res, 0 , 1, args.gaussian)
        bifilter_file_name +=  "gu_" + str(args.gaussian) + "_"
        trifilter_file_name +=  "gu_" + str(args.gaussian) + "_"
        medfilter_file_name +=  "gu_" + str(args.gaussian) + "_"
        nosie_img +=  "gu_" + str(args.gaussian) + "_"
    if args.salt_pepper == 0.0 and args.gaussian == 0 and args.impulse == 0:
        print("Error, Image without noise")
        exit(0)


    cv2.imwrite(str(args.output_dir) +'/'+ nosie_img + ".png", img_res) 
    noise1 = Image.open(str(args.output_dir) +'/'+ nosie_img + ".png")
    lena_n = np.array(noise1).astype(int)
    print("Median Filter noise...")
    result13 = median(lena_n, 3)
    Image.fromarray(result13.astype('uint8')).save(str(args.output_dir) +'/'+ medfilter_file_name + ".png")
    
    img_res = img_res.astype("float32")
    print("Bilateral Filter noise...")
    im_bi = bilateral_filter(img_res, 1,ss=args.ss, sr=args.sr)
    cv2.imwrite(str(args.output_dir) +'/'+ bifilter_file_name + ".png", im_bi) 
    road = Road_value(img_res, 1)
    print("Trilateral Filter noise...")
    im_tri = trilateral_filter(img_res, road, 1,ss=args.ss, sr=args.sr, si=args.si, sj=args.sj)
    cv2.imwrite(str(args.output_dir) +'/'+ trifilter_file_name +  ".png", im_tri) 
    print("Noise cleaning finish")
    
    
    ori = Image.open(file)
    f1 = Image.open(str(args.output_dir) +'/'+ medfilter_file_name + ".png")
    f2 = Image.open(str(args.output_dir) +'/'+ bifilter_file_name + ".png")
    f3 = Image.open(str(args.output_dir) +'/'+ trifilter_file_name +  ".png")
    
    lena_ori = np.array(ori).astype(int)

    print("The PSNR  after median filter is : " + str(PSNR(lena_ori,np.array(f1).astype(int))))
    print("The PSNR  after bilateral filter is : " + str(PSNR(lena_ori,np.array(f2).astype(int))))
    print("The PSNR  after trilateral filter is : " + str(PSNR(lena_ori,np.array(f3).astype(int))))
    


    
def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        help="input",
        required = True
    )
    parser.add_argument(
        "--impulse",
        type=float,
        help="impulse noise rate [0,1)",
        default=0.0,
    )
    parser.add_argument(
        "--salt_pepper",
        type=float,
        help="salt and pepper noise rate [0,1)",
        default=0.0,
    )
    parser.add_argument(
        "--gaussian",
        type=int,
        help="gaussian nosie recommand: [10,50]",
        default=0,
    )
    parser.add_argument(
        "--ss",
        type=float,
        help="sigma in ws, Recommand: 0.5 for salt and pepper or impulse noise, 5 for gaussian noise amd mix noise ",
        default=5,
    )
    parser.add_argument(
        "--sr",
        type=int,
        help="sigma in wr, Recommand: 2 times for your gaussian sigma (amplitude)",
        default=20,
    )
    parser.add_argument(
        "--si",
        type=int,
        help="sigma in wi, Recommand: [25,55]",
        default=40,
    )
    parser.add_argument(
        "--sj",
        type=int,
        help="sigma in wj, Recommand: [30,80]",
        default=50,
    )
  
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Directory to save the processed file.",
        default="output",
    )
 
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    main(args)

