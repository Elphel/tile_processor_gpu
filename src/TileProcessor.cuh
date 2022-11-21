/**
 **
 ** TileProcessor.cuh
 **
 ** Copyright (C) 2018 Elphel, Inc.
 **
 ** -----------------------------------------------------------------------------**
 **
 **  TileProcessor.cuh is free software: you can redistribute it and/or modify
 **  it under the terms of the GNU General Public License as published by
 **  the Free Software Foundation, either version 3 of the License, or
 **  (at your option) any later version.
 **
 **  This program is distributed in the hope that it will be useful,
 **  but WITHOUT ANY WARRANTY; without even the implied warranty of
 **  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 **  GNU General Public License for more details.
 **
 **  You should have received a copy of the GNU General Public License
 **  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 **
 **  Additional permission under GNU GPL version 3 section 7
 **
 **  If you modify this Program, or any covered work, by linking or
 **  combining it with NVIDIA Corporation's CUDA libraries from the
 **  NVIDIA CUDA Toolkit (or a modified version of those libraries),
 **  containing parts covered by the terms of NVIDIA CUDA Toolkit
 **  EULA, the licensors of this Program grant you additional
 **  permission to convey the resulting work.
 ** -----------------------------------------------------------------------------**
 */

/**
**************************************************************************
* \file TileProcessor.cuh
* \brief Top level of the Tile Processor for frequency domain
*/
// Avoiding includes in jcuda, all source files will be merged
#pragma once
#ifndef JCUDA
#include "tp_defines.h"
#include "dtt8x8.h"
#include "geometry_correction.h"
#include "TileProcessor.h"
#endif // #ifndef JCUDA

// CUDA fast math is slower!
//#define FASTMATH 1
/*
 fast
GPU run time =620.210698ms, (direct conversion: 24.077195999999997ms, imclt: 17.218263ms), corr2D: 85.503204ms), textures: 237.225665ms, RGBA: 256.185703ms
nofast
GPU run time =523.451927ms, (direct conversion: 24.080189999999998ms, imclt: 17.090526999999998ms), corr2D: 30.623282999999997ms), textures: 231.154339ms, RGBA: 220.503017ms
 */

//#define TASK_TEXTURE_BITS ((1 << TASK_TEXTURE_N_BIT) | (1 << TASK_TEXTURE_E_BIT) | (1 << TASK_TEXTURE_S_BIT) | (1 << TASK_TEXTURE_W_BIT))

#define TASK_TEXTURE_BITS ((1 << TASK_TEXT_N_BIT) | (1 << TASK_TEXT_NE_BIT) | (1 << TASK_TEXT_E_BIT) | (1 << TASK_TEXT_SE_BIT)\
		| (1 << TASK_TEXT_S_BIT) | (1 << TASK_TEXT_SW_BIT) | (1 << TASK_TEXT_W_BIT) | (1 << TASK_TEXT_NW_BIT))


//#define IMCLT14
//#define NOICLT 1
//#define TEST_IMCLT
//#define SAVE_CLT
// Not enough shared memory to have more threads per block,even just for the result clt tiles
// What to do:
// 1) make single image aberration correction: 1/4 of the result tiles
// With 4 cameras = calculate correlations (9x9), reusing kernel or just clt ones after color reducing, then output them to device memory
//Average run time =1308.124146 ms
//#define TILES_PER_BLOCK    2
//Average run time =12502.638672 - with 2 tiles/block it is longer!
///12129.268555 ms
//Average run time =4704.506348 ms (syncwarp)
//Average run time =4705.612305 ms (syncthreads)
//Average run time =1051.411255 ms
//Average run time =861.866577 ms
//Average run time =850.871277 ms had bugs
//Average run time =857.947632 ms fixed bugs


// Something broke, even w/o LPF: Average run time =1093.115112 ms
// without clt copying to device memory - Average run time =965.342407 ms - still worse
//Average run time =965.880554 ms
// combined tx and ty into a single int : Average run time =871.017944 ms
//Average run time =873.386597 ms (reduced number of registers)
//__umul24 : Average run time =879.125122 ms
// without __umul24 - back to Average run time =871.315552 ms
// Added copying clt to device memory - Average run time =942.071960 ms
// Removed rest of NOICLT : Average run time =943.456177 ms
// Added lpf: Average run time =1046.101318 ms (0.1 sec, 10%) - can be combined with the PSF kernel
//#define USE_UMUL24
////#define TILES_PER_BLOCK    4
//Average run time =5155.922852 ms
//Average run time =1166.388306 ms
//Average run time =988.750977 ms
//#define TILES_PER_BLOCK    8
//Average run time =9656.743164 ms
// Average run time =9422.057617 ms (reducing divergence)
//#define TILES_PER_BLOCK    1

//#define THREADS_PER_TILE   8
//#define IMCLT_THREADS_PER_TILE 16
//#define IMCLT_TILES_PER_BLOCK   4


/// #define KERNELS_STEP  (1 << KERNELS_LSTEP)
//#define TILES-X        (IMG-WIDTH / DTT_SIZE)
//#define TILES-Y        (IMG-HEIGHT / DTT_SIZE)
#define CONVERT_DIRECT_INDEXING_THREADS_LOG2 5
#define CONVERT_DIRECT_INDEXING_THREADS (1 << CONVERT_DIRECT_INDEXING_THREADS_LOG2) // 32

// Make TILES-YA >= TILES-X and a multiple of 4
//#define TILES-YA       ((TILES-Y +3) & (~3))

// increase row length by 1 so vertical passes will use different ports
#define MCLT_UNION_LEN   (DTT_SIZE2 * (DTT_SIZE2 + 2))

struct CltExtra{
	float data_x;   // kernel data is relative to this displacement X (0.5 pixel increments)
	float data_y;   // kernel data is relative to this displacement Y (0.5 pixel increments)
	float center_x; // actual center X (use to find derivatives)
	float center_y; // actual center X (use to find derivatives)
	float dxc_dx;   // add this to data_x per each pixel X-shift relative to the kernel center location
	float dxc_dy;   // same per each Y-shift pixel
	float dyc_dx;
	float dyc_dy;
};

/*
 Python code to generate constant coefficients:
def setup_hwindow(n=8, l=4):
    hwindow = [math.sin(math.pi*((1.0+2*i)/(4*n))) for i in range(2*n)]
    print("__constant__ float HWINDOW[] = {", end="") #
    for i in range (n):
        print("%ff"%(hwindow[i]), end ="")
        if i == (n-1):
            print("};")
        elif ((i + 1) % l) == 0:
            print(",")
            print("                                ", end ="")
        else:
            print(", ",end="")

def setup_hwindow2(n=8, l=4):
    hwindow = [0.5*math.sin(math.pi*((1.0+2*i)/(4*n))) for i in range(2*n)]
    print("__constant__ float HWINDOW2[] = {", end="") #
    for i in range (n):
        print("%ff"%(hwindow[i]), end ="")
        if i == (n-1):
            print("};")
        elif ((i + 1) % l) == 0:
            print(",")
            print("                                 ", end ="")
        else:
            print(", ",end="")

def setup_hwindow_sq(n=8, l=4):
    hwindow = [(math.sin(math.pi*((1.0+2*i)/(4*n)))) ** 2 for i in range(2*n)]
    print("__constant__ float HWINDOW_SQ[] = {", end="") #
    for i in range (n):
        print("%ff"%(hwindow[i]), end ="")
        if i == (n-1):
            print("};")
        elif ((i + 1) % l) == 0:
            print(",")
            print("                                 ", end ="")
        else:
            print(", ",end="")


def setup_hwindow_sqi(n=8, l=4):
    hwindow = [1.0/(math.sin(math.pi*((1.0+2*i)/(4*n)))) ** 2 for i in range(2*n)]
    print("__constant__ float HWINDOW_SQi[] = {", end="") #
    for i in range (n):
        print("%ff"%(hwindow[i]), end ="")
        if i == (n-1):
            print("};")
        elif ((i + 1) % l) == 0:
            print(",")
            print("                                 ", end ="")
        else:
            print(", ",end="")



def get_fold_rindices(n=8):
    n1 = n>>1;
    rind = [0] * (2 * n) # reverse indices
    rcs =  [0] * (2 * n) # reverse signs for cosine term
    rss =  [0] * (2 * n) # reverse signs for sine term
    for x in range (n1):
        ri0 = n + n1 - x - 1
        ri1 = n + n1 + x
        ri2 =          x
        ri3 = n      - x - 1
        rind[ri0] = x
        rind[ri1] = x
        rind[ri2] = x + n1
        rind[ri3] = x + n1
        rcs[ri0] = -1
        rss[ri0] =  1
        rcs[ri1] = -1
        rss[ri1] = -1
        rcs[ri2] =  1
        rss[ri2] =  1
        rcs[ri3] = -1
        rss[ri3] =  1
    rind0 = []
    rind1 = []
    # generate start indices for the first 2 bayer rows
    for a in rind:
        rind0.append(a+rind[0]*n)
        rind1.append(a+rind[1]*n)
    #column increments for odd/even bayer rows
    inc_even = []
    inc_odd = []
    for i in range (n-1):
        inc_even.append(rind[2*i+2]-rind[2*i])
        inc_odd.append (rind[2*i+3]-rind[2*i+1])
    inc_even.reverse()
    inc_odd.reverse()
    # combine increments into int data
    inc_e = 0
    inc_o = 0
    for d in inc_even:
        inc_e = ((inc_e) << 4) | (d & 0xf)
    for d in inc_odd:
        inc_o = ((inc_o) << 4) | (d & 0xf)
    print("__constant__ int fold_indx2[2][%d] = {{"%(2*n), end="") #
    for d in rind0[:-1]:
        print('0x%2x,'%(d), end="")
    print('0x%2x},'%(rind0[-1]))
    print("                                      {", end="") #
    for d in rind1[:-1]:
        print('0x%2x,'%(d), end="")
    print('0x%2x}};'%(rind1[-1]))
    print("__constant__ int fold_inc[]=          {0x%08x, 0x%08x};"%(inc_e, inc_o))

def set_imclt_sa(stride=9):
    sa8 =[0x24,0x2c,0x34,0x3c,0x3c,0x34,0x2c,0x24,0x1c,0x14,0x0c,0x04,0x04,0x0c,0x14,0x1c]
    sa8s = [d // 8 + (d % 8) * stride for d in sa8]
    print("__constant__ int imclt_indx9[16] = {", end="") #
    for d in sa8s[:-1]:
        print('0x%02x,'%(d), end="")
    print('0x%2x};'%(sa8s[-1]))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np # math
def printTextureBlend(transform_size):
    ts2 = 2 * transform_size
    ts2m1 = ts2-1
    alphabBlend = np.zeros(shape=(8,ts2*ts2), dtype=float) #
    blend1d =   np.zeros(shape=(ts2,), dtype=float)
    dirBlend = ((0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1),(1,0),(1,1))
    for i in range (transform_size):
        blend1d[i] = 0.5 * (1.0 + np.cos(np.pi * (i +0.5) /transform_size))
    for i in range (ts2):
        for j in range (ts2):
            indx = i * ts2 + j
            for m, dir in enumerate(dirBlend):
                a = 1.0
                if dir[0] > 0:
                    a *= blend1d[j]
                elif dir[0]<0:
                    a *= blend1d[ts2m1 - j]
                else:
                    if (j >= transform_size):
                        a*= blend1d[j - transform_size] # 8->0
                    else:
                        a*= blend1d[transform_size - 1 -j] # 7->0
                if dir[1] > 0:
                    a *= blend1d[i]
                elif dir[1]<0:
                    a *= blend1d[ts2m1 - i]
                else:
                    if (i >= transform_size):
                        a*= blend1d[i - transform_size] # 8->0
                    else:
                        a*= blend1d[transform_size - 1 -i] # 7->0
                alphabBlend[m][indx] = a;

    floats_in_line0=8 # 16 #8
    segment_len = transform_size*transform_size//2
    print("__constant__ float textureBlend[8][%d] = {"%(segment_len)) #32
    # only for transform_size == 8
    for m, blend in enumerate (alphabBlend):

        for i in range (segment_len):
            if m in   (0,1):
                x = 4 + (i % 8)
                y = 4 + (i // 8)
            elif m in (2,3):
                x = 8 + (i % 4)
                y = 4 + (i // 4)
            elif m in (4,5):
                x = 4 + (i % 8)
                y = 8 + (i // 8)
            elif m in (6,7):
                x = 4 + (i % 4)
                y = 4 + (i // 4)
            indx = x + 16 * y
            floats_in_line = floats_in_line0
            if ((m >>1) & 1) !=0:
                floats_in_line = floats_in_line0 // 2
            if ((i % floats_in_line) == 0):
                print("    ",end="")
                if (i == 0) :
                    print("{",end="")
                else:
                    print(" ",end="")
            print("%ff"%(blend[indx]), end ="")
            if (((i + 1) % floats_in_line) == 0):
                if (i == (segment_len -1)):
                    print("}",end="")
                else:
                    print(",")
            else:
                print(", ",end="")
        if (m == len(alphabBlend)-1):
            print("};")
        else:
            print(",")

printTextureBlend(8)


"""
Set up correlation pairs - run:
setup_pairs(0,16)

def get_pairs_list(num_cams):
    pairs = []
    for i in range(1,num_cams // 2 + 1):
        numj = num_cams
        if ((2 * i) == num_cams):
            numj = num_cams//2
        #print ("i=",i," num_j=",numj)
        for j in range (numj):
            #print ("    j=",j)
            pairs.append([j, (i+j) % num_cams, i])
    #print("length=",len(pairs))
    if (num_cams == 4): # quad cameras are numbered 0,1/2,3 instead of 3,0/2,1
        reorder4 = [1,3,2,0]
        pairs4 = []
        for pair in pairs:
            pairs4.append([reorder4[pair[0]],reorder4[pair[1]], pair[2]])
        return pairs4
    return pairs

def setup_pairs(min_n,max_n, pairs_per_line = 8):
    indices =   [0]
    all_pairs = []
    for num_cams in range(min_n,max_n+1):
        all_pairs += get_pairs_list(num_cams)
        indices.append (len(all_pairs))
#    print ("indices=",indices)
#    print ("all_pairs=",all_pairs)
    print("__constant__ int pairs_offsets[]=           {", end="")
    for i, offs in enumerate(indices):
        print(offs,end="")
        if i < len(indices)-1:
            print(", ",end="")
    print("};");
    print("// {pair_start, pair_end, pair_length}");
    print("__constant__ unsigned char all_pairs[][3] = {", end="")
    for i, pair in enumerate(all_pairs):
        print ("{%2d, %2d, %2d}"%(pair[0],pair[1],pair[2]),end="")
        if i < len(all_pairs)-1:
            print(", ",end="")
            if (((i+1) % pairs_per_line) == 0):
                print("\n                                             ",end="")
    print("};");

*/


__constant__ float HWINDOW[] =   {0.098017f, 0.290285f, 0.471397f, 0.634393f,
                                  0.773010f, 0.881921f, 0.956940f, 0.995185f};


__constant__ float HWINDOW_SQ[] = {0.009607f, 0.084265f, 0.222215f, 0.402455f,
                                   0.597545f, 0.777785f, 0.915735f, 0.990393f};
__constant__ float HWINDOW_SQi[] = {104.086869f, 11.867296f, 4.500149f, 2.484751f,
                                      1.673514f, 1.285702f, 1.092019f, 1.009701f};

// Offsets in 8x8 DCT_CC/DST_SC tile for the first 2 lines of the 16x16 bayer image
__constant__ int fold_indx2[2][16] = {{0x24,0x25,0x26,0x27,0x27,0x26,0x25,0x24,0x23,0x22,0x21,0x20,0x20,0x21,0x22,0x23},
                                      {0x2c,0x2d,0x2e,0x2f,0x2f,0x2e,0x2d,0x2c,0x2b,0x2a,0x29,0x28,0x28,0x29,0x2a,0x2b}};

// increments of the offsets in 8x8 tile when going down, jumping two lines (same Bayer). Each 4 bits have to be <<3,
// addd to the current index and result should be AND-ed with 0x3f. inc_e is for even rows (0,2, ...) while inc_o - for odd ones (1,3,)
__constant__ int fold_inc[]=          {0x02feee12, 0x021eeef2};

// LPF for sigma 0.9 each color (modify through cudaMemcpyToSymbol() or similar in Driver API
__constant__ float lpf_data[4][64]={
		{ // red
				1.00000000f, 0.87041007f, 0.65943687f, 0.43487258f, 0.24970076f, 0.12518080f, 0.05616371f, 0.02728573f,
				0.87041007f, 0.75761368f, 0.57398049f, 0.37851747f, 0.21734206f, 0.10895863f, 0.04888546f, 0.02374977f,
				0.65943687f, 0.57398049f, 0.43485698f, 0.28677101f, 0.16466189f, 0.08254883f, 0.03703642f, 0.01799322f,
				0.43487258f, 0.37851747f, 0.28677101f, 0.18911416f, 0.10858801f, 0.05443770f, 0.02442406f, 0.01186582f,
				0.24970076f, 0.21734206f, 0.16466189f, 0.10858801f, 0.06235047f, 0.03125774f, 0.01402412f, 0.00681327f,
				0.12518080f, 0.10895863f, 0.08254883f, 0.05443770f, 0.03125774f, 0.01567023f, 0.00703062f, 0.00341565f,
				0.05616371f, 0.04888546f, 0.03703642f, 0.02442406f, 0.01402412f, 0.00703062f, 0.00315436f, 0.00153247f,
				0.02728573f, 0.02374977f, 0.01799322f, 0.01186582f, 0.00681327f, 0.00341565f, 0.00153247f, 0.00074451f
		},{ // blue
				1.00000000f, 0.87041007f, 0.65943687f, 0.43487258f, 0.24970076f, 0.12518080f, 0.05616371f, 0.02728573f,
				0.87041007f, 0.75761368f, 0.57398049f, 0.37851747f, 0.21734206f, 0.10895863f, 0.04888546f, 0.02374977f,
				0.65943687f, 0.57398049f, 0.43485698f, 0.28677101f, 0.16466189f, 0.08254883f, 0.03703642f, 0.01799322f,
				0.43487258f, 0.37851747f, 0.28677101f, 0.18911416f, 0.10858801f, 0.05443770f, 0.02442406f, 0.01186582f,
				0.24970076f, 0.21734206f, 0.16466189f, 0.10858801f, 0.06235047f, 0.03125774f, 0.01402412f, 0.00681327f,
				0.12518080f, 0.10895863f, 0.08254883f, 0.05443770f, 0.03125774f, 0.01567023f, 0.00703062f, 0.00341565f,
				0.05616371f, 0.04888546f, 0.03703642f, 0.02442406f, 0.01402412f, 0.00703062f, 0.00315436f, 0.00153247f,
				0.02728573f, 0.02374977f, 0.01799322f, 0.01186582f, 0.00681327f, 0.00341565f, 0.00153247f, 0.00074451f
		},{ // green
				1.00000000f, 0.91166831f, 0.75781950f, 0.57470069f, 0.39864249f, 0.25575500f, 0.15880862f, 0.11071780f,
				0.91166831f, 0.83113910f, 0.69088002f, 0.52393641f, 0.36342972f, 0.23316373f, 0.14478079f, 0.10093791f,
				0.75781950f, 0.69088002f, 0.57429040f, 0.43551939f, 0.30209905f, 0.19381613f, 0.12034827f, 0.08390411f,
				0.57470069f, 0.52393641f, 0.43551939f, 0.33028089f, 0.22910011f, 0.14698258f, 0.09126743f, 0.06362960f,
				0.39864249f, 0.36342972f, 0.30209905f, 0.22910011f, 0.15891583f, 0.10195481f, 0.06330787f, 0.04413682f,
				0.25575500f, 0.23316373f, 0.19381613f, 0.14698258f, 0.10195481f, 0.06541062f, 0.04061610f, 0.02831663f,
				0.15880862f, 0.14478079f, 0.12034827f, 0.09126743f, 0.06330787f, 0.04061610f, 0.02522018f, 0.01758294f,
				0.11071780f, 0.10093791f, 0.08390411f, 0.06362960f, 0.04413682f, 0.02831663f, 0.01758294f, 0.01225843f
		},{ // mono
				1.00000000f, 0.94100932f, 0.83403534f, 0.69821800f, 0.55623487f, 0.42968171f, 0.33580928f, 0.28608280f,
				0.94100932f, 0.88549854f, 0.78483503f, 0.65702965f, 0.52342219f, 0.40433449f, 0.31599966f, 0.26920658f,
				0.83403534f, 0.78483503f, 0.69561495f, 0.58233849f, 0.46391954f, 0.35836973f, 0.28007681f, 0.23860316f,
				0.69821800f, 0.65702965f, 0.58233849f, 0.48750838f, 0.38837320f, 0.30001150f, 0.23446808f, 0.19974816f,
				0.55623487f, 0.52342219f, 0.46391954f, 0.38837320f, 0.30939723f, 0.23900395f, 0.18678883f, 0.15912923f,
				0.42968171f, 0.40433449f, 0.35836973f, 0.30001150f, 0.23900395f, 0.18462637f, 0.14429110f, 0.12292455f,
				0.33580928f, 0.31599966f, 0.28007681f, 0.23446808f, 0.18678883f, 0.14429110f, 0.11276787f, 0.09606926f,
				0.28608280f, 0.26920658f, 0.23860316f, 0.19974816f, 0.15912923f, 0.12292455f, 0.09606926f, 0.08184337f
		}};

__constant__ float lpf_rb_corr[64]={ // modify if needed
				1.00000000f, 0.92598908f, 0.79428680f, 0.63198650f, 0.46862740f, 0.32891038f, 0.22914618f, 0.17771927f,
				0.92598908f, 0.85745578f, 0.73550091f, 0.58521260f, 0.43394386f, 0.30456742f, 0.21218686f, 0.16456610f,
				0.79428680f, 0.73550091f, 0.63089153f, 0.50197854f, 0.37222456f, 0.26124917f, 0.18200779f, 0.14116007f,
				0.63198650f, 0.58521260f, 0.50197854f, 0.39940694f, 0.29616619f, 0.20786692f, 0.14481729f, 0.11231618f,
				0.46862740f, 0.43394386f, 0.37222456f, 0.29616619f, 0.21961164f, 0.15413642f, 0.10738418f, 0.08328412f,
				0.32891038f, 0.30456742f, 0.26124917f, 0.20786692f, 0.15413642f, 0.10818204f, 0.07536856f, 0.05845371f,
				0.22914618f, 0.21218686f, 0.18200779f, 0.14481729f, 0.10738418f, 0.07536856f, 0.05250797f, 0.04072369f,
				0.17771927f, 0.16456610f, 0.14116007f, 0.11231618f, 0.08328412f, 0.05845371f, 0.04072369f, 0.03158414f

		};
__constant__ float lpf_corr[64]={ // modify if needed
				1.00000000f, 0.87041007f, 0.65943687f, 0.43487258f, 0.24970076f, 0.12518080f, 0.05616371f, 0.02728573f,
				0.87041007f, 0.75761368f, 0.57398049f, 0.37851747f, 0.21734206f, 0.10895863f, 0.04888546f, 0.02374977f,
				0.65943687f, 0.57398049f, 0.43485698f, 0.28677101f, 0.16466189f, 0.08254883f, 0.03703642f, 0.01799322f,
				0.43487258f, 0.37851747f, 0.28677101f, 0.18911416f, 0.10858801f, 0.05443770f, 0.02442406f, 0.01186582f,
				0.24970076f, 0.21734206f, 0.16466189f, 0.10858801f, 0.06235047f, 0.03125774f, 0.01402412f, 0.00681327f,
				0.12518080f, 0.10895863f, 0.08254883f, 0.05443770f, 0.03125774f, 0.01567023f, 0.00703062f, 0.00341565f,
				0.05616371f, 0.04888546f, 0.03703642f, 0.02442406f, 0.01402412f, 0.00703062f, 0.00315436f, 0.00153247f,
				0.02728573f, 0.02374977f, 0.01799322f, 0.01186582f, 0.00681327f, 0.00341565f, 0.00153247f, 0.00074451f

		};

__constant__ float LoG_corr[64]={ // modify if needed high-pass filter before correlation to fit into float range
				1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f,
				1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f,
				1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f,
				1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f,
				1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f,
				1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f,
                                1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f,
                                1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f
                };
__constant__ float textureBlend[8][32] = {
    {0.240485f, 0.313023f, 0.368542f, 0.398588f, 0.398588f, 0.368542f, 0.313023f, 0.240485f,
     0.132783f, 0.172835f, 0.203490f, 0.220080f, 0.220080f, 0.203490f, 0.172835f, 0.132783f,
     0.050352f, 0.065540f, 0.077165f, 0.083456f, 0.083456f, 0.077165f, 0.065540f, 0.050352f,
     0.005741f, 0.007472f, 0.008798f, 0.009515f, 0.009515f, 0.008798f, 0.007472f, 0.005741f},
    {0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.003867f, 0.033913f, 0.089431f, 0.161970f,
     0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.002135f, 0.018725f, 0.049379f, 0.089431f,
     0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000810f, 0.007101f, 0.018725f, 0.033913f,
     0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000092f, 0.000810f, 0.002135f, 0.003867f},
    {0.005741f, 0.050352f, 0.132783f, 0.240485f,
     0.007472f, 0.065540f, 0.172835f, 0.313023f,
     0.008798f, 0.077165f, 0.203490f, 0.368542f,
     0.009515f, 0.083456f, 0.220080f, 0.398588f,
     0.009515f, 0.083456f, 0.220080f, 0.398588f,
     0.008798f, 0.077165f, 0.203490f, 0.368542f,
     0.007472f, 0.065540f, 0.172835f, 0.313023f,
     0.005741f, 0.050352f, 0.132783f, 0.240485f},
    {0.000000f, 0.000000f, 0.000000f, 0.000000f,
     0.000000f, 0.000000f, 0.000000f, 0.000000f,
     0.000000f, 0.000000f, 0.000000f, 0.000000f,
     0.000000f, 0.000000f, 0.000000f, 0.000000f,
     0.000092f, 0.000810f, 0.002135f, 0.003867f,
     0.000810f, 0.007101f, 0.018725f, 0.033913f,
     0.002135f, 0.018725f, 0.049379f, 0.089431f,
     0.003867f, 0.033913f, 0.089431f, 0.161970f},
    {0.005741f, 0.007472f, 0.008798f, 0.009515f, 0.009515f, 0.008798f, 0.007472f, 0.005741f,
     0.050352f, 0.065540f, 0.077165f, 0.083456f, 0.083456f, 0.077165f, 0.065540f, 0.050352f,
     0.132783f, 0.172835f, 0.203490f, 0.220080f, 0.220080f, 0.203490f, 0.172835f, 0.132783f,
     0.240485f, 0.313023f, 0.368542f, 0.398588f, 0.398588f, 0.368542f, 0.313023f, 0.240485f},
    {0.003867f, 0.002135f, 0.000810f, 0.000092f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,
     0.033913f, 0.018725f, 0.007101f, 0.000810f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,
     0.089431f, 0.049379f, 0.018725f, 0.002135f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,
     0.161970f, 0.089431f, 0.033913f, 0.003867f, 0.000000f, 0.000000f, 0.000000f, 0.000000f},
    {0.240485f, 0.132783f, 0.050352f, 0.005741f,
     0.313023f, 0.172835f, 0.065540f, 0.007472f,
     0.368542f, 0.203490f, 0.077165f, 0.008798f,
     0.398588f, 0.220080f, 0.083456f, 0.009515f,
     0.398588f, 0.220080f, 0.083456f, 0.009515f,
     0.368542f, 0.203490f, 0.077165f, 0.008798f,
     0.313023f, 0.172835f, 0.065540f, 0.007472f,
     0.240485f, 0.132783f, 0.050352f, 0.005741f},
    {0.161970f, 0.089431f, 0.033913f, 0.003867f,
     0.089431f, 0.049379f, 0.018725f, 0.002135f,
     0.033913f, 0.018725f, 0.007101f, 0.000810f,
     0.003867f, 0.002135f, 0.000810f, 0.000092f,
     0.000000f, 0.000000f, 0.000000f, 0.000000f,
     0.000000f, 0.000000f, 0.000000f, 0.000000f,
     0.000000f, 0.000000f, 0.000000f, 0.000000f,
     0.000000f, 0.000000f, 0.000000f, 0.000000f}};

__constant__ int pairs_offsets[]=           {0, 0, 0, 1, 4, 10, 20, 35, 56, 84, 120, 165, 220, 286, 364, 455, 560, 680};
// {pair_start, pair_end, pair_length}
__constant__ unsigned char all_pairs[][3] = {{ 0,  1,  1}, { 0,  1,  1}, { 1,  2,  1}, { 2,  0,  1}, { 1,  3,  1}, { 3,  2,  1}, { 2,  0,  1}, { 0,  1,  1},
                                             { 1,  2,  2}, { 3,  0,  2}, { 0,  1,  1}, { 1,  2,  1}, { 2,  3,  1}, { 3,  4,  1}, { 4,  0,  1}, { 0,  2,  2},
                                             { 1,  3,  2}, { 2,  4,  2}, { 3,  0,  2}, { 4,  1,  2}, { 0,  1,  1}, { 1,  2,  1}, { 2,  3,  1}, { 3,  4,  1},
                                             { 4,  5,  1}, { 5,  0,  1}, { 0,  2,  2}, { 1,  3,  2}, { 2,  4,  2}, { 3,  5,  2}, { 4,  0,  2}, { 5,  1,  2},
                                             { 0,  3,  3}, { 1,  4,  3}, { 2,  5,  3}, { 0,  1,  1}, { 1,  2,  1}, { 2,  3,  1}, { 3,  4,  1}, { 4,  5,  1},
                                             { 5,  6,  1}, { 6,  0,  1}, { 0,  2,  2}, { 1,  3,  2}, { 2,  4,  2}, { 3,  5,  2}, { 4,  6,  2}, { 5,  0,  2},
                                             { 6,  1,  2}, { 0,  3,  3}, { 1,  4,  3}, { 2,  5,  3}, { 3,  6,  3}, { 4,  0,  3}, { 5,  1,  3}, { 6,  2,  3},
                                             { 0,  1,  1}, { 1,  2,  1}, { 2,  3,  1}, { 3,  4,  1}, { 4,  5,  1}, { 5,  6,  1}, { 6,  7,  1}, { 7,  0,  1},
                                             { 0,  2,  2}, { 1,  3,  2}, { 2,  4,  2}, { 3,  5,  2}, { 4,  6,  2}, { 5,  7,  2}, { 6,  0,  2}, { 7,  1,  2},
                                             { 0,  3,  3}, { 1,  4,  3}, { 2,  5,  3}, { 3,  6,  3}, { 4,  7,  3}, { 5,  0,  3}, { 6,  1,  3}, { 7,  2,  3},
                                             { 0,  4,  4}, { 1,  5,  4}, { 2,  6,  4}, { 3,  7,  4}, { 0,  1,  1}, { 1,  2,  1}, { 2,  3,  1}, { 3,  4,  1},
                                             { 4,  5,  1}, { 5,  6,  1}, { 6,  7,  1}, { 7,  8,  1}, { 8,  0,  1}, { 0,  2,  2}, { 1,  3,  2}, { 2,  4,  2},
                                             { 3,  5,  2}, { 4,  6,  2}, { 5,  7,  2}, { 6,  8,  2}, { 7,  0,  2}, { 8,  1,  2}, { 0,  3,  3}, { 1,  4,  3},
                                             { 2,  5,  3}, { 3,  6,  3}, { 4,  7,  3}, { 5,  8,  3}, { 6,  0,  3}, { 7,  1,  3}, { 8,  2,  3}, { 0,  4,  4},
                                             { 1,  5,  4}, { 2,  6,  4}, { 3,  7,  4}, { 4,  8,  4}, { 5,  0,  4}, { 6,  1,  4}, { 7,  2,  4}, { 8,  3,  4},
                                             { 0,  1,  1}, { 1,  2,  1}, { 2,  3,  1}, { 3,  4,  1}, { 4,  5,  1}, { 5,  6,  1}, { 6,  7,  1}, { 7,  8,  1},
                                             { 8,  9,  1}, { 9,  0,  1}, { 0,  2,  2}, { 1,  3,  2}, { 2,  4,  2}, { 3,  5,  2}, { 4,  6,  2}, { 5,  7,  2},
                                             { 6,  8,  2}, { 7,  9,  2}, { 8,  0,  2}, { 9,  1,  2}, { 0,  3,  3}, { 1,  4,  3}, { 2,  5,  3}, { 3,  6,  3},
                                             { 4,  7,  3}, { 5,  8,  3}, { 6,  9,  3}, { 7,  0,  3}, { 8,  1,  3}, { 9,  2,  3}, { 0,  4,  4}, { 1,  5,  4},
                                             { 2,  6,  4}, { 3,  7,  4}, { 4,  8,  4}, { 5,  9,  4}, { 6,  0,  4}, { 7,  1,  4}, { 8,  2,  4}, { 9,  3,  4},
                                             { 0,  5,  5}, { 1,  6,  5}, { 2,  7,  5}, { 3,  8,  5}, { 4,  9,  5}, { 0,  1,  1}, { 1,  2,  1}, { 2,  3,  1},
                                             { 3,  4,  1}, { 4,  5,  1}, { 5,  6,  1}, { 6,  7,  1}, { 7,  8,  1}, { 8,  9,  1}, { 9, 10,  1}, {10,  0,  1},
                                             { 0,  2,  2}, { 1,  3,  2}, { 2,  4,  2}, { 3,  5,  2}, { 4,  6,  2}, { 5,  7,  2}, { 6,  8,  2}, { 7,  9,  2},
                                             { 8, 10,  2}, { 9,  0,  2}, {10,  1,  2}, { 0,  3,  3}, { 1,  4,  3}, { 2,  5,  3}, { 3,  6,  3}, { 4,  7,  3},
                                             { 5,  8,  3}, { 6,  9,  3}, { 7, 10,  3}, { 8,  0,  3}, { 9,  1,  3}, {10,  2,  3}, { 0,  4,  4}, { 1,  5,  4},
                                             { 2,  6,  4}, { 3,  7,  4}, { 4,  8,  4}, { 5,  9,  4}, { 6, 10,  4}, { 7,  0,  4}, { 8,  1,  4}, { 9,  2,  4},
                                             {10,  3,  4}, { 0,  5,  5}, { 1,  6,  5}, { 2,  7,  5}, { 3,  8,  5}, { 4,  9,  5}, { 5, 10,  5}, { 6,  0,  5},
                                             { 7,  1,  5}, { 8,  2,  5}, { 9,  3,  5}, {10,  4,  5}, { 0,  1,  1}, { 1,  2,  1}, { 2,  3,  1}, { 3,  4,  1},
                                             { 4,  5,  1}, { 5,  6,  1}, { 6,  7,  1}, { 7,  8,  1}, { 8,  9,  1}, { 9, 10,  1}, {10, 11,  1}, {11,  0,  1},
                                             { 0,  2,  2}, { 1,  3,  2}, { 2,  4,  2}, { 3,  5,  2}, { 4,  6,  2}, { 5,  7,  2}, { 6,  8,  2}, { 7,  9,  2},
                                             { 8, 10,  2}, { 9, 11,  2}, {10,  0,  2}, {11,  1,  2}, { 0,  3,  3}, { 1,  4,  3}, { 2,  5,  3}, { 3,  6,  3},
                                             { 4,  7,  3}, { 5,  8,  3}, { 6,  9,  3}, { 7, 10,  3}, { 8, 11,  3}, { 9,  0,  3}, {10,  1,  3}, {11,  2,  3},
                                             { 0,  4,  4}, { 1,  5,  4}, { 2,  6,  4}, { 3,  7,  4}, { 4,  8,  4}, { 5,  9,  4}, { 6, 10,  4}, { 7, 11,  4},
                                             { 8,  0,  4}, { 9,  1,  4}, {10,  2,  4}, {11,  3,  4}, { 0,  5,  5}, { 1,  6,  5}, { 2,  7,  5}, { 3,  8,  5},
                                             { 4,  9,  5}, { 5, 10,  5}, { 6, 11,  5}, { 7,  0,  5}, { 8,  1,  5}, { 9,  2,  5}, {10,  3,  5}, {11,  4,  5},
                                             { 0,  6,  6}, { 1,  7,  6}, { 2,  8,  6}, { 3,  9,  6}, { 4, 10,  6}, { 5, 11,  6}, { 0,  1,  1}, { 1,  2,  1},
                                             { 2,  3,  1}, { 3,  4,  1}, { 4,  5,  1}, { 5,  6,  1}, { 6,  7,  1}, { 7,  8,  1}, { 8,  9,  1}, { 9, 10,  1},
                                             {10, 11,  1}, {11, 12,  1}, {12,  0,  1}, { 0,  2,  2}, { 1,  3,  2}, { 2,  4,  2}, { 3,  5,  2}, { 4,  6,  2},
                                             { 5,  7,  2}, { 6,  8,  2}, { 7,  9,  2}, { 8, 10,  2}, { 9, 11,  2}, {10, 12,  2}, {11,  0,  2}, {12,  1,  2},
                                             { 0,  3,  3}, { 1,  4,  3}, { 2,  5,  3}, { 3,  6,  3}, { 4,  7,  3}, { 5,  8,  3}, { 6,  9,  3}, { 7, 10,  3},
                                             { 8, 11,  3}, { 9, 12,  3}, {10,  0,  3}, {11,  1,  3}, {12,  2,  3}, { 0,  4,  4}, { 1,  5,  4}, { 2,  6,  4},
                                             { 3,  7,  4}, { 4,  8,  4}, { 5,  9,  4}, { 6, 10,  4}, { 7, 11,  4}, { 8, 12,  4}, { 9,  0,  4}, {10,  1,  4},
                                             {11,  2,  4}, {12,  3,  4}, { 0,  5,  5}, { 1,  6,  5}, { 2,  7,  5}, { 3,  8,  5}, { 4,  9,  5}, { 5, 10,  5},
                                             { 6, 11,  5}, { 7, 12,  5}, { 8,  0,  5}, { 9,  1,  5}, {10,  2,  5}, {11,  3,  5}, {12,  4,  5}, { 0,  6,  6},
                                             { 1,  7,  6}, { 2,  8,  6}, { 3,  9,  6}, { 4, 10,  6}, { 5, 11,  6}, { 6, 12,  6}, { 7,  0,  6}, { 8,  1,  6},
                                             { 9,  2,  6}, {10,  3,  6}, {11,  4,  6}, {12,  5,  6}, { 0,  1,  1}, { 1,  2,  1}, { 2,  3,  1}, { 3,  4,  1},
                                             { 4,  5,  1}, { 5,  6,  1}, { 6,  7,  1}, { 7,  8,  1}, { 8,  9,  1}, { 9, 10,  1}, {10, 11,  1}, {11, 12,  1},
                                             {12, 13,  1}, {13,  0,  1}, { 0,  2,  2}, { 1,  3,  2}, { 2,  4,  2}, { 3,  5,  2}, { 4,  6,  2}, { 5,  7,  2},
                                             { 6,  8,  2}, { 7,  9,  2}, { 8, 10,  2}, { 9, 11,  2}, {10, 12,  2}, {11, 13,  2}, {12,  0,  2}, {13,  1,  2},
                                             { 0,  3,  3}, { 1,  4,  3}, { 2,  5,  3}, { 3,  6,  3}, { 4,  7,  3}, { 5,  8,  3}, { 6,  9,  3}, { 7, 10,  3},
                                             { 8, 11,  3}, { 9, 12,  3}, {10, 13,  3}, {11,  0,  3}, {12,  1,  3}, {13,  2,  3}, { 0,  4,  4}, { 1,  5,  4},
                                             { 2,  6,  4}, { 3,  7,  4}, { 4,  8,  4}, { 5,  9,  4}, { 6, 10,  4}, { 7, 11,  4}, { 8, 12,  4}, { 9, 13,  4},
                                             {10,  0,  4}, {11,  1,  4}, {12,  2,  4}, {13,  3,  4}, { 0,  5,  5}, { 1,  6,  5}, { 2,  7,  5}, { 3,  8,  5},
                                             { 4,  9,  5}, { 5, 10,  5}, { 6, 11,  5}, { 7, 12,  5}, { 8, 13,  5}, { 9,  0,  5}, {10,  1,  5}, {11,  2,  5},
                                             {12,  3,  5}, {13,  4,  5}, { 0,  6,  6}, { 1,  7,  6}, { 2,  8,  6}, { 3,  9,  6}, { 4, 10,  6}, { 5, 11,  6},
                                             { 6, 12,  6}, { 7, 13,  6}, { 8,  0,  6}, { 9,  1,  6}, {10,  2,  6}, {11,  3,  6}, {12,  4,  6}, {13,  5,  6},
                                             { 0,  7,  7}, { 1,  8,  7}, { 2,  9,  7}, { 3, 10,  7}, { 4, 11,  7}, { 5, 12,  7}, { 6, 13,  7}, { 0,  1,  1},
                                             { 1,  2,  1}, { 2,  3,  1}, { 3,  4,  1}, { 4,  5,  1}, { 5,  6,  1}, { 6,  7,  1}, { 7,  8,  1}, { 8,  9,  1},
                                             { 9, 10,  1}, {10, 11,  1}, {11, 12,  1}, {12, 13,  1}, {13, 14,  1}, {14,  0,  1}, { 0,  2,  2}, { 1,  3,  2},
                                             { 2,  4,  2}, { 3,  5,  2}, { 4,  6,  2}, { 5,  7,  2}, { 6,  8,  2}, { 7,  9,  2}, { 8, 10,  2}, { 9, 11,  2},
                                             {10, 12,  2}, {11, 13,  2}, {12, 14,  2}, {13,  0,  2}, {14,  1,  2}, { 0,  3,  3}, { 1,  4,  3}, { 2,  5,  3},
                                             { 3,  6,  3}, { 4,  7,  3}, { 5,  8,  3}, { 6,  9,  3}, { 7, 10,  3}, { 8, 11,  3}, { 9, 12,  3}, {10, 13,  3},
                                             {11, 14,  3}, {12,  0,  3}, {13,  1,  3}, {14,  2,  3}, { 0,  4,  4}, { 1,  5,  4}, { 2,  6,  4}, { 3,  7,  4},
                                             { 4,  8,  4}, { 5,  9,  4}, { 6, 10,  4}, { 7, 11,  4}, { 8, 12,  4}, { 9, 13,  4}, {10, 14,  4}, {11,  0,  4},
                                             {12,  1,  4}, {13,  2,  4}, {14,  3,  4}, { 0,  5,  5}, { 1,  6,  5}, { 2,  7,  5}, { 3,  8,  5}, { 4,  9,  5},
                                             { 5, 10,  5}, { 6, 11,  5}, { 7, 12,  5}, { 8, 13,  5}, { 9, 14,  5}, {10,  0,  5}, {11,  1,  5}, {12,  2,  5},
                                             {13,  3,  5}, {14,  4,  5}, { 0,  6,  6}, { 1,  7,  6}, { 2,  8,  6}, { 3,  9,  6}, { 4, 10,  6}, { 5, 11,  6},
                                             { 6, 12,  6}, { 7, 13,  6}, { 8, 14,  6}, { 9,  0,  6}, {10,  1,  6}, {11,  2,  6}, {12,  3,  6}, {13,  4,  6},
                                             {14,  5,  6}, { 0,  7,  7}, { 1,  8,  7}, { 2,  9,  7}, { 3, 10,  7}, { 4, 11,  7}, { 5, 12,  7}, { 6, 13,  7},
                                             { 7, 14,  7}, { 8,  0,  7}, { 9,  1,  7}, {10,  2,  7}, {11,  3,  7}, {12,  4,  7}, {13,  5,  7}, {14,  6,  7},
                                             { 0,  1,  1}, { 1,  2,  1}, { 2,  3,  1}, { 3,  4,  1}, { 4,  5,  1}, { 5,  6,  1}, { 6,  7,  1}, { 7,  8,  1},
                                             { 8,  9,  1}, { 9, 10,  1}, {10, 11,  1}, {11, 12,  1}, {12, 13,  1}, {13, 14,  1}, {14, 15,  1}, {15,  0,  1},
                                             { 0,  2,  2}, { 1,  3,  2}, { 2,  4,  2}, { 3,  5,  2}, { 4,  6,  2}, { 5,  7,  2}, { 6,  8,  2}, { 7,  9,  2},
                                             { 8, 10,  2}, { 9, 11,  2}, {10, 12,  2}, {11, 13,  2}, {12, 14,  2}, {13, 15,  2}, {14,  0,  2}, {15,  1,  2},
                                             { 0,  3,  3}, { 1,  4,  3}, { 2,  5,  3}, { 3,  6,  3}, { 4,  7,  3}, { 5,  8,  3}, { 6,  9,  3}, { 7, 10,  3},
                                             { 8, 11,  3}, { 9, 12,  3}, {10, 13,  3}, {11, 14,  3}, {12, 15,  3}, {13,  0,  3}, {14,  1,  3}, {15,  2,  3},
                                             { 0,  4,  4}, { 1,  5,  4}, { 2,  6,  4}, { 3,  7,  4}, { 4,  8,  4}, { 5,  9,  4}, { 6, 10,  4}, { 7, 11,  4},
                                             { 8, 12,  4}, { 9, 13,  4}, {10, 14,  4}, {11, 15,  4}, {12,  0,  4}, {13,  1,  4}, {14,  2,  4}, {15,  3,  4},
                                             { 0,  5,  5}, { 1,  6,  5}, { 2,  7,  5}, { 3,  8,  5}, { 4,  9,  5}, { 5, 10,  5}, { 6, 11,  5}, { 7, 12,  5},
                                             { 8, 13,  5}, { 9, 14,  5}, {10, 15,  5}, {11,  0,  5}, {12,  1,  5}, {13,  2,  5}, {14,  3,  5}, {15,  4,  5},
                                             { 0,  6,  6}, { 1,  7,  6}, { 2,  8,  6}, { 3,  9,  6}, { 4, 10,  6}, { 5, 11,  6}, { 6, 12,  6}, { 7, 13,  6},
                                             { 8, 14,  6}, { 9, 15,  6}, {10,  0,  6}, {11,  1,  6}, {12,  2,  6}, {13,  3,  6}, {14,  4,  6}, {15,  5,  6},
                                             { 0,  7,  7}, { 1,  8,  7}, { 2,  9,  7}, { 3, 10,  7}, { 4, 11,  7}, { 5, 12,  7}, { 6, 13,  7}, { 7, 14,  7},
                                             { 8, 15,  7}, { 9,  0,  7}, {10,  1,  7}, {11,  2,  7}, {12,  3,  7}, {13,  4,  7}, {14,  5,  7}, {15,  6,  7},
                                             { 0,  8,  8}, { 1,  9,  8}, { 2, 10,  8}, { 3, 11,  8}, { 4, 12,  8}, { 5, 13,  8}, { 6, 14,  8}, { 7, 15,  8}};

__device__ void convertCorrectTile(
//		int                   num_cams,
		int                   num_colors,
		struct CltExtra     * gpu_kernel_offsets, // [tileY][tileX][color]
		float               * gpu_kernels,        // [tileY][tileX][color]
		float               * gpu_images,
		float               * gpu_clt,
		const int             color,
		const int             lpf_mask,
		const float           centerX,
		const float           centerY,
		const int             txy,
		const float           tscale,
		const size_t          dstride, // in floats (pixels)
		float               * clt_tile, //        [4][DTT_SIZE][DTT_SIZE1], // +1 to alternate column ports
		float               * clt_kernels, //      [4][DTT_SIZE][DTT_SIZE1], // +1 to alternate column ports
		int   int_topleft     [2],
		float residual_shift  [2],
	    float window_hor_cos  [2*DTT_SIZE],
	    float window_hor_sin  [2*DTT_SIZE],
	    float window_vert_cos [2*DTT_SIZE],
	    float window_vert_sin [2*DTT_SIZE],
		int                woi_width,
		int                woi_height,
		int                kernels_hor,
		int                kernels_vert,
		int                tilesx);

__device__ void debug_print_lpf(
		float * lpf_tile);

__device__ void debug_print_clt1(
		float * clt_tile, //         [4][DTT_SIZE][DTT_SIZE1], // +1 to alternate column ports)
		const int color,
		int mask);

__device__ void debug_print_clt_scaled(
		float * clt_tile, //         [4][DTT_SIZE][DTT_SIZE1], // +1 to alternate column ports)
		const int color,
		int mask,
		float scale); // scale printed results

__device__ void debug_print_mclt(
		float * mclt_tile, //         [4][DTT_SIZE][DTT_SIZE1], // +1 to alternate column ports)
		const int color);

__device__ void debug_print_corr_15x15(
		int     corr_radius,
		float * mclt_tile, //DTT_SIZE2M1 x DTT_SIZE2M1
		const int color);

__device__ void shiftTileHor( // implemented, used
		float * clt_tile, //        [4][DTT_SIZE][DTT_SIZE1], // +1 to alternate column ports
		float residual_shift                         );

__device__ void shiftTileVert( // implemented, used
		float *clt_tile, //        [4][DTT_SIZE][DTT_SIZE1], // +1 to alternate column ports
		float residual_shift                         );

__device__ void convolveTiles( // implemented, used
		float* clt_tile, //    [4][DTT_SIZE][DTT_SIZE1], // 4 quadrants of the clt data, rows extended to optimize shared ports
		float* kernel); //      [4][DTT_SIZE][DTT_SIZE1]) // 4 quadrants of the CLT kernel (DTT3 converted)

__device__ void correlateAccumulateTiles(
		float  scale,      //    scale correlation
		float* clt_tile1,  //    [4][DTT_SIZE][DTT_SIZE1], // 4 quadrants of the clt data 1, rows extended to optimize shared ports
		float* clt_tile2,  //    [4][DTT_SIZE][DTT_SIZE1], // 4 quadrants of the clt data 2, rows extended to optimize shared ports
		float* corr_tile); //    [4][DTT_SIZE][DTT_SIZE1]) // 4 quadrants of the correlation result

__device__ void resetCorrelation(
		float* corr_tile); //    [4][DTT_SIZE][DTT_SIZE1]) // 4 quadrants of the correlation result

__device__ void normalizeTileAmplitude(
		float * clt_tile, //       [4][DTT_SIZE][DTT_SIZE1], // +1 to alternate column ports
		float fat_zero2);  // fat zero is absolute, scale it outside

__device__ void imclt8threads(// for 8 threads
		int     do_acc,     // 1 - add to previous value, 0 - overwrite
		float * clt_tile,   //        [4][DTT_SIZE][DTT_SIZE1], // +1 to alternate column ports [4][8][9]
		float * mclt_tile,  //           [2* DTT_SIZE][DTT_SIZE1+ DTT_SIZE], // +1 to alternate column ports[16][17]
		int debug);

__device__ void debayer(
		const int rb_mode,   // 0 - green, 1 - r/b
		float * mclt_src,  // [2* DTT_SIZE][DTT_SIZE1+ DTT_SIZE], // +1 to alternate column ports[16][17]
		float * mclt_dst,  // [2* DTT_SIZE][DTT_SIZE1+ DTT_SIZE], // +1 to alternate column ports[16][17]
		int debug);

__device__ void debayer_shot(
		const int rb_mode,   // 0 - green, 1 - r/b
		float     min_shot,  // 10.0
		float     shot_corr, // 3.0 (0.0 for mono)
		float   * mclt_src,  // [2* DTT_SIZE][DTT_SIZE1+ DTT_SIZE], // +1 to alternate column ports[16][17]
		float   * mclt_dst,  // [2* DTT_SIZE][DTT_SIZE1+ DTT_SIZE], // +1 to alternate column ports[16][17]
		float   * mclt_tmp,
		int       debug);

__device__ void tile_combine_rgba(
		int     num_cams,      //  number of cameras used <= NUM_CAMS
		int     colors,        // number of colors
		float * mclt_tile,     // debayer // has gaps to align with union !
		float * rbg_tile,      // if not null - original (not-debayered) rbg tile to use for the output
		float * rgba,          // result
		int     calc_extra,    // 1 - calcualate ports_rgb, max_diff
///		float ports_rgb_shared [NUM_COLORS][NUM_CAMS], // return to system memory (optionally pass null to skip calculation)
///		float max_diff_shared  [NUM_CAMS], // return to system memory (optionally pass null to skip calculation)
///		float max_diff_tmp     [NUM_CAMS][TEXTURE_THREADS_PER_TILE],
///		float ports_rgb_tmp    [NUM_COLORS][NUM_CAMS][TEXTURE_THREADS_PER_TILE], // [4*3][8]
		float * ports_rgb_shared, //  [colors][num_cams], // return to system memory (optionally pass null to skip calculation)
		float * max_diff_shared,  //  [num_cams], // return to system memory (optionally pass null to skip calculation)
		float * max_diff_tmp,     //  [num_cams][TEXTURE_THREADS_PER_TILE],
		float * ports_rgb_tmp,    //  [colors][num_cams][TEXTURE_THREADS_PER_TILE], // [4*3][8]

		float * port_offsets,  // [port]{x_off, y_off} - just to scale pixel value differences
		//		int           port_mask,      // which port to use, 0xf - all 4 (will modify as local variable)
		float   diff_sigma,     // pixel value/pixel change
		float   diff_threshold, // pixel value/pixel change
		// next not used
		//		boolean       diff_gauss,     // when averaging images, use gaussian around average as weight (false - sharp all/nothing)
		float   min_agree,      // minimal number of channels to agree on a point (real number to work with fuzzy averages)
		float * chn_weights,     // color channel weights, sum == 1.0
		int     dust_remove,     // Do not reduce average weight when only one image differs much from the average
		int     keep_weights,   // eturn channel weights and rms after A in RGBA (weight are always calculated, not so for the crms)
		int     debug);

__device__ void imclt_plane( // not implemented, not used
		int               color,
		float           * gpu_clt,   // [TILES-Y][TILES-X][NUM_COLORS][DTT_SIZE*DTT_SIZE]
		float           * gpu_rbg,            // WIDTH, HEIGHT
		const size_t      dstride);            // in floats (pixels)

extern "C" __global__ void clear_texture_list(
		int              * gpu_texture_indices,// packed tile + bits (now only (1 << 7)
		int                width,  // <= TILES-X, use for faster processing of LWIR images
		int                height); // <= TILES-Y, use for faster processing of LWIR images

extern "C" __global__ void mark_texture_tiles(
		int                num_cams,
		float            * gpu_ftasks,         // flattened tasks, 27 floats for quad EO, 99 floats for LWIR16
		int                num_tiles,           // number of tiles in task list
		int                width,               // number of tiles in a row
		int              * gpu_texture_indices);// packed tile + bits (now only (1 << 7)

extern "C" __global__ void mark_texture_neighbor_tiles( // TODO: remove __global__?
		int                num_cams,
		float            * gpu_ftasks,          // flattened tasks, 27 floats for quad EO, 99 floats for LWIR16
		int                num_tiles,           // number of tiles in task list
		int                width,               // number of tiles in a row
		int                height,              // number of tiles rows
		int              * gpu_texture_indices, // packed tile + bits (now only (1 << 7)
		int              * woi);                  // x,y,width,height of the woi

extern "C" __global__ void gen_texture_list(
		int                num_cams,
		float            * gpu_ftasks,          // flattened tasks, 27 floats for quad EO, 99 floats for LWIR16
		int                num_tiles,            // number of tiles in task list
		int                width,                // number of tiles in a row
		int                height,               // number of tiles rows
		int              * gpu_texture_indices,  // packed tile + bits (now only (1 << 7)
		int              * num_texture_tiles,    // number of texture tiles to process
		int              * woi);                 // min_x, min_y, max_x, max_y input

extern "C" __global__ void clear_texture_rbga(
		int               texture_width,
		int               texture_slice_height,
		const size_t      texture_rbga_stride,     // in floats 8*stride
		float           * gpu_texture_tiles);  // (number of colors +1 + ?)*16*16 rgba texture tiles

//inline __device__ int get_task_size(int num_cams);
inline __device__ int get_task_task(int num_tile, float * gpu_ftasks, int num_cams);
inline __device__ int get_task_txy(int num_tile, float * gpu_ftasks, int num_cams);

__global__ void index_direct(
		int                task_size,          // flattened task size in 4-byte floats
		float            * gpu_ftasks,         // flattened tasks, 27 floats for quad EO, 99 floats for LWIR16
//		struct tp_task   * gpu_tasks,
		int                num_tiles,          // number of tiles in task
		int *              active_tiles,       // pointer to the calculated number of non-zero tiles
		int *              pnum_active_tiles); //  indices to gpu_tasks  // should be initialized to zero

__global__ void index_correlate(
		int               num_cams,
//		int *             sel_pairs,           // unused bits should be 0
		int               sel_pairs0,
		int               sel_pairs1,
		int               sel_pairs2,
		int               sel_pairs3,
		float            * gpu_ftasks,         // flattened tasks, 27 floats for quad EO, 99 floats for LWIR16
//		struct tp_task   * gpu_tasks,
		int                num_tiles,         // number of tiles in task
		int                width,                // number of tiles in a row
		int *              gpu_corr_indices,  // array of correlation tasks
		int *              pnum_corr_tiles);   // pointer to the length of correlation tasks array

__global__ void index_inter_correlate(
		int               num_cams,
		int               sel_sensors,
		float            * gpu_ftasks,        // flattened tasks, 27 floats for quad EO, 99 floats for LWIR16
		int                num_tiles,         // number of tiles in task
		int                width,             // number of tiles in a row
		int *              gpu_corr_indices,  // array of correlation tasks
		int *              pnum_corr_tiles);  // pointer to the length of correlation tasks array

extern "C" __global__ void create_nonoverlap_list(
		int                num_cams,
		float            * gpu_ftasks ,         // flattened tasks, 27 floats for quad EO, 99 floats for LWIR16
		int                num_tiles,           // number of tiles in task
		int                width,               // number of tiles in a row
		int *              nonoverlap_list,     // pointer to the calculated number of non-zero tiles
		int *              pnonoverlap_length); //  indices to gpu_tasks  // should be initialized to zero

__global__ void convert_correct_tiles(
	    int                num_cams,           // actual number of cameras
	    int                num_colors,         // actual number of colors: 3 for RGB, 1 for LWIR/mono
		float           ** gpu_kernel_offsets, // [num_cams],
		float           ** gpu_kernels,        // [num_cams],
		float           ** gpu_images,         // [num_cams],
		float            * gpu_ftasks,         // flattened tasks, 27 floats for quad EO, 99 floats for LWIR16
//		struct tp_task   * gpu_tasks,
		int              * gpu_active_tiles,   // indices in gpu_tasks to non-zero tiles
		int                num_active_tiles,   // number of tiles in task
		float           ** gpu_clt,            // [num_cams][TILES-Y][TILES-X][NUM_COLORS][DTT_SIZE*DTT_SIZE]
		size_t             dstride,            // in floats (pixels)
		int                lpf_mask,           // apply lpf to colors : bit 0 - red, bit 1 - blue, bit2 - green. Now - always 0 !
		int                woi_width,
		int                woi_height,
		int                kernels_hor,
		int                kernels_vert, //);
		int                tilesx);

extern "C" __global__ void combine_inter(     // combine per-senor interscene correlations
		int               num_sel_sensors,    // number of sensors to correlate (not counting sum of all)
		int               num_corr_tiles,     // number of correlation tiles to process (here it includes sum)
		int             * gpu_corr_indices,   // packed tile+pair
		size_t            corr_stride,        // in floats
		float           * gpu_corrs);          // correlation output data (either pixel domain or transform domain

extern "C" __global__ void correlate2D_inter_inner( // will only process to TD, no normalisations and back conversion
		int               num_cams,
		int               num_sel_sensors,    // number of sensors to correlate (not counting sum of all)
		float          ** gpu_clt,            // [num_cams] ->[TILES-Y][TILES-X][colors][DTT_SIZE*DTT_SIZE]
		float          ** gpu_clt_ref,        // [num_cams] ->[TILES-Y][TILES-X][colors][DTT_SIZE*DTT_SIZE]
		int               colors,             // number of colors (3/1)
		float             scale0,             // scale for R
		float             scale1,             // scale for B
		float             scale2,             // scale for G
		int               num_corr_tiles,     // number of correlation tiles to process (here it includes sum)
		int             * gpu_corr_indices,   // packed tile+pair
		size_t            corr_stride,        // in floats
		float           * gpu_corrs);          // correlation output data (either pixel domain or transform domain

extern "C" __global__ void correlate2D_inner(
		int               num_cams,
		float          ** gpu_clt,            // [num_cams] ->[TILES-Y][TILES-X][colors][DTT_SIZE*DTT_SIZE]
		int               colors,             // number of colors (3/1)
		float             scale0,             // scale for R
		float             scale1,             // scale for B
		float             scale2,             // scale for G
		float             fat_zero2,          // here - absolute
		int            num_corr_tiles,     // number of correlation tiles to process
		int             * gpu_corr_indices,   // packed tile+pair
		size_t      corr_stride,        // in floats
		int               corr_radius0,        // radius of the output correlation (7 for 15x15)
		float           * gpu_corrs);          // correlation output data (either pixel domain or transform domain

extern "C" __global__ void corr2D_normalize_inner(
		int               num_corr_tiles,     // number of correlation tiles to process
		const size_t      corr_stride_td,     // (in floats) stride for the input TD correlations
		float           * gpu_corrs_td,       // correlation tiles in transform domain
		float           * corr_weights,       // null or per-tile weight (fat_zero2 will be divided by it)
		const size_t      corr_stride,        // in floats
		float           * gpu_corrs,          // correlation output data (either pixel domain or transform domain
		float             fat_zero2,          // here - absolute
		int               corr_radius);        // radius of the output correlation (7 for 15x15)

extern "C" __global__ void corr2D_combine_inner(
		int               num_tiles,          // number of tiles to process (each with num_pairs)
		int               num_pairs,          // num pairs per tile (should be the same)
		int               init_output,        // !=0 - reset output tiles to zero before accumulating
		int               pairs_mask,         // selected pairs (0x3 - horizontal, 0xc - vertical, 0xf - quad, 0x30 - cross)
		int             * gpu_corr_indices,   // packed tile+pair
		int             * gpu_combo_indices,  // output if noty null: packed tile+pairs_mask (will point to the first used pair
		const size_t      corr_stride,        // (in floats) stride for the input TD correlations
		float           * gpu_corrs,          // input correlation tiles
		const size_t      corr_stride_combo,  // (in floats) stride for the output TD correlations (same as input)
		float           * gpu_corrs_combo);    // combined correlation output (one per tile)

extern "C" __global__ void textures_accumulate( // (8,4,1) (N,1,1)
		int               num_cams,           // number of cameras used
		int             * woi,                // x, y, width,height
		float          ** gpu_clt,            // [num_cams] ->[TILES-Y][TILES-X][colors][DTT_SIZE*DTT_SIZE]
		size_t            num_texture_tiles,  // number of texture tiles to process
		int               gpu_texture_indices_offset,// add to gpu_texture_indices
		int             * gpu_texture_indices,// packed tile + bits (now only (1 << 7)
		// TODO: use geometry_correction rXY !
		struct gc       * gpu_geometry_correction,
		int               colors,             // number of colors (3/1)
		int               is_lwir,            // do not perform shot correction
		float             min_shot,           // 10.0
		float             scale_shot,         // 3.0
		float             diff_sigma,         // pixel value/pixel change
		float             diff_threshold,     // pixel value/pixel change
		float             min_agree,          // minimal number of channels to agree on a point (real number to work with fuzzy averages)
		float             weights[3],         // scale for R,B,G
		int               dust_remove,        // Do not reduce average weight when only one image differs much from the average
		int               keep_weights,       // return channel weights after A in RGBA (was removed) (should be 0 if gpu_texture_rbg)?
// combining both non-overlap and overlap (each calculated if pointer is not null )
		size_t            texture_rbg_stride, // in floats
		float           * gpu_texture_rbg,    // (number of colors +1 + ?)*16*16 rgba texture tiles
		size_t            texture_stride,     // in floats (now 256*4 = 1024)
		float           * gpu_texture_tiles,  // (number of colors +1 + ?)*16*16 rgba texture tiles
		int               linescan_order,     // if !=0 then output gpu_diff_rgb_combo in linescan order, else  - in gpu_texture_indices order
		float           * gpu_diff_rgb_combo, //) // diff[num_cams], R[num_cams], B[num_cams],G[num_cams]
		int               tilesx);

__device__ int get_textures_shared_size( // in bytes
	    int                num_cams,     // actual number of cameras
	    int                num_colors,   // actual number of colors: 3 for RGB, 1 for LWIR/mono
		int *              offsets);     // in floats


// ====== end of local declarations ====

/**
 * Calculate  2D phase correlation pairs from CLT representation. This is an outer kernel that calls other
 * ones with CDP, this one should be configured as correlate2D<<<1,1>>>
 *
 * @param num_cams         number of cameras <= NUM_CAMS
 * @param sel_pairs        array of length to accommodate all pairs (4  for 16 cameras, 120 pairs).
 * @param gpu_clt          array of num_cams pointers to the CLT (frequency domain) data [TILES-Y][TILES-X][NUM_COLORS][DTT_SIZE*DTT_SIZE]
 * @param colors           number of colors used:  3 for RGB or 1 for monochrome
 * @param scale0           scale red (or mono) component before mixing
 * @param scale1           scale blue (if colors = 3) component before mixing
 * @param scale2           scale green (if colors = 3) component before mixing
 * @param fat_zero2        add this value squared to the sum of squared components before normalization (squared)
 * @param gpu_ftasks           flattened tasks, 29 floats for quad EO, 101 floats for LWIR16
// * @param gpu_tasks        array of per-tile tasks (now bits 4..9 - correlation pairs)
 * @param num_tiles        number of tiles int gpu_tasks array prepared for processing
 * @param tilesx           number of tile rows
 * @param gpu_corr_indices allocated array for per-tile correlation tasks (4 bytes per tile)
 * @param pnum_corr_tiles  allocated space for pointer to a number of number of correlation tiles to process
 * @param corr_stride,     stride (in floats) for correlation outputs.
 * @param corr_radius,     radius of the output correlation (maximal 7 for 15x15)
 * @param gpu_corrs)       allocated array for the correlation output data (each element stride, payload: (2*corr_radius+1)^2
 */
extern "C" __global__ void correlate2D(
		int               num_cams,
//		int *             sel_pairs,
		int               sel_pairs0,
		int               sel_pairs1,
		int               sel_pairs2,
		int               sel_pairs3,
		float          ** gpu_clt,            // [num_cams] ->[TILES-Y][TILES-X][colors][DTT_SIZE*DTT_SIZE]
		int               colors,             // number of colors (3/1)
		float             scale0,             // scale for R
		float             scale1,             // scale for B
		float             scale2,            // scale for G
		float             fat_zero2,           // here - absolute
		float            * gpu_ftasks,         // flattened tasks, 27 floats for quad EO, 99 floats for LWIR16
		int               num_tiles,          // number of tiles in task
		int               tilesx,             // number of tile rows
		int             * gpu_corr_indices,   // packed tile+pair
		int             * pnum_corr_tiles,    // pointer to a number of correlation tiles to process
//		const size_t      corr_stride,        // in floats
		size_t      corr_stride,        // in floats
		int               corr_radius,        // radius of the output correlation (7 for 15x15)
		float           * gpu_corrs)          // correlation output data
{
	 dim3 threads0(CONVERT_DIRECT_INDEXING_THREADS, 1, 1);
	 dim3 blocks0 ((num_tiles + CONVERT_DIRECT_INDEXING_THREADS -1) >> CONVERT_DIRECT_INDEXING_THREADS_LOG2,1, 1);
	 if (threadIdx.x == 0) { // only 1 thread, 1 block
		 *pnum_corr_tiles = 0;
		 index_correlate<<<blocks0,threads0>>>(
				 num_cams,            // int               num_cams,
				 sel_pairs0,          // int               sel_pairs0,
				 sel_pairs1,          // int               sel_pairs1,
				 sel_pairs2,          // int               sel_pairs2,
				 sel_pairs3,          // int               sel_pairs3,

				 gpu_ftasks,          // float            * gpu_ftasks,         // flattened tasks, 29 floats for quad EO, 101 floats for LWIR16
				 num_tiles,           // int                num_tiles,          // number of tiles in task
				 tilesx,              // int                width,                // number of tiles in a row
				 gpu_corr_indices,    // int *              gpu_corr_indices,  // array of correlation tasks
				 pnum_corr_tiles);    // int *              pnum_corr_tiles);   // pointer to the length of correlation tasks array
		 cudaDeviceSynchronize();
		 dim3 threads_corr(CORR_THREADS_PER_TILE, CORR_TILES_PER_BLOCK, 1);
		 dim3 grid_corr((*pnum_corr_tiles + CORR_TILES_PER_BLOCK-1) / CORR_TILES_PER_BLOCK,1,1);
		 correlate2D_inner<<<grid_corr,threads_corr>>>(
				 num_cams,           // int               num_cams,
				 gpu_clt,            // float          ** gpu_clt,            // [num_cams] ->[TILES-Y][TILES-X][colors][DTT_SIZE*DTT_SIZE]
				 colors,             // int               colors,             // number of colors (3/1)
				 scale0,             // float             scale0,             // scale for R
				 scale1,             // float             scale1,             // scale for B
				 scale2,             // float             scale2,             // scale for G
				 fat_zero2,          // float             fat_zero2,           // here - absolute
				 *pnum_corr_tiles,   // size_t            num_corr_tiles,     // number of correlation tiles to process
				 gpu_corr_indices,   //  int             * gpu_corr_indices,  // packed tile+pair
				 corr_stride,        // const size_t      corr_stride,        // in floats
				 corr_radius,        // int               corr_radius,        // radius of the output correlation (7 for 15x15)
				 gpu_corrs);         // float           * gpu_corrs);         // correlation output data
	 }
}

/**
 * Calculate  2D phase correlation pairs from CLT representation. This is an outer kernel that calls other
 * ones with CDP, this one should be configured as correlate2D<<<1,1>>>
 *
 * @param num_cams         number of cameras <= NUM_CAMS
 * @param sel_pairs        array of length to accommodate all pairs (4  for 16 cameras, 120 pairs).
 * @param gpu_clt          array of num_cams pointers to the CLT (frequency domain) data [TILES-Y][TILES-X][NUM_COLORS][DTT_SIZE*DTT_SIZE]
 * @param colors           number of colors used:  3 for RGB or 1 for monochrome
 * @param scale0           scale red (or mono) component before mixing
 * @param scale1           scale blue (if colors = 3) component before mixing
 * @param scale2           scale green (if colors = 3) component before mixing
 * @param fat_zero2        add this value squared to the sum of squared components before normalization (squared)
 * @param gpu_ftasks           flattened tasks, 29 floats for quad EO, 101 floats for LWIR16
 * @param num_tiles        number of tiles int gpu_tasks array prepared for processing
 * @param tilesx           number of tile rows
 * @param gpu_corr_indices allocated array for per-tile correlation tasks (4 bytes per tile)
 * @param pnum_corr_tiles  allocated space for pointer to a number of number of correlation tiles to process
 * @param corr_stride,     stride (in floats) for correlation outputs.
 * @param gpu_corrs)       allocated array for the correlation output data (each element stride, payload: (2*corr_radius+1)^2
 */
extern "C" __global__ void correlate2D_inter( // only results in TD
		int               num_cams,
		int               sel_sensors,
		float          ** gpu_clt,            // [num_cams] ->[TILES-Y][TILES-X][colors][DTT_SIZE*DTT_SIZE]
		float          ** gpu_clt_ref,        // [num_cams] ->[TILES-Y][TILES-X][colors][DTT_SIZE*DTT_SIZE]
		int               colors,             // number of colors (3/1)
		float             scale0,             // scale for R
		float             scale1,             // scale for B
		float             scale2,             // scale for G
		float            * gpu_ftasks,        // flattened tasks, 27 floats for quad EO, 99 floats for LWIR16
		int               num_tiles,          // number of tiles in task
		int               tilesx,             // number of tile rows
		int             * gpu_corr_indices,   // packed tile+pair
		int             * pnum_corr_tiles,    // pointer to a number of correlation tiles to process
		size_t            corr_stride,        // in floats
		float           * gpu_corrs)          // correlation output data
{
	dim3 threads0(CONVERT_DIRECT_INDEXING_THREADS, 1, 1);
	dim3 blocks0 ((num_tiles + CONVERT_DIRECT_INDEXING_THREADS -1) >> CONVERT_DIRECT_INDEXING_THREADS_LOG2,1, 1);
	if (threadIdx.x == 0) { // only 1 thread, 1 block
		int num_sel_sensors = __popc (sel_sensors); // number of non-zero bits
		if (num_sel_sensors > 0){
// try with null tp_tasks to use same sequence from GPU memory
			*pnum_corr_tiles = 0;
			index_inter_correlate<<<blocks0,threads0>>>(
					num_cams,            // int               num_cams,
					sel_sensors,         // int               sel_sensors,
					gpu_ftasks,          // float            * gpu_ftasks,         // flattened tasks, 29 floats for quad EO, 101 floats for LWIR16
					num_tiles,           // int                num_tiles,          // number of tiles in task
					tilesx,              // int                width,                // number of tiles in a row
					gpu_corr_indices,    // int *              gpu_corr_indices,  // array of correlation tasks
					pnum_corr_tiles);    // int *              pnum_corr_tiles);   // pointer to the length of correlation tasks array
			cudaDeviceSynchronize();
			int num_corr_tiles_with_sum = (*pnum_corr_tiles);
			int num_corr_tiles_wo_sum =   num_corr_tiles_with_sum * num_sel_sensors/ (num_sel_sensors + 1); // remove sum from count
			dim3 threads_corr(CORR_THREADS_PER_TILE, CORR_TILES_PER_BLOCK, 1);
			dim3 grid_corr((num_corr_tiles_wo_sum + CORR_TILES_PER_BLOCK-1) / CORR_TILES_PER_BLOCK,1,1);

			correlate2D_inter_inner<<<grid_corr,threads_corr>>>( // will only process to TD, no normalisations and back conversion
					num_cams,                // int          num_cams,
					num_sel_sensors,         // int          num_sel_sensors,    // number of sensors to correlate (not counting sum of all)
					gpu_clt,                 // float     ** gpu_clt,            // [num_cams] ->[TILES-Y][TILES-X][colors][DTT_SIZE*DTT_SIZE]
					gpu_clt_ref,             // float     ** gpu_clt_ref,        // [num_cams] ->[TILES-Y][TILES-X][colors][DTT_SIZE*DTT_SIZE]
					colors,                  // int          colors,             // number of colors (3/1)
					scale0,                  // float        scale0,             // scale for R
					scale1,                  // float        scale1,             // scale for B
					scale2,                  // float        scale2,             // scale for G
					num_corr_tiles_with_sum, // int          num_corr_tiles,     // number of correlation tiles to process (here it includes sum for compatibility with intra format)
					gpu_corr_indices,        // int        * gpu_corr_indices,   // packed tile + sensor (0xff - sum)
					corr_stride,             // size_t       corr_stride,        // in floats
					gpu_corrs);              // float      * gpu_corrs)          // correlation output data (either pixel domain or transform domain
			dim3 grid_combine((num_tiles + CORR_TILES_PER_BLOCK-1) / CORR_TILES_PER_BLOCK,1,1);
			combine_inter<<<grid_combine,threads_corr>>>(     // combine per-senor interscene correlations
					num_sel_sensors,         // int          num_sel_sensors,    // number of sensors to correlate (not counting sum of all)
					num_corr_tiles_with_sum, // int          num_corr_tiles,     // number of correlation tiles to process (here it includes sum)
					gpu_corr_indices,        // int        * gpu_corr_indices,   // packed tile+pair NOT USED
					corr_stride,             // size_t       corr_stride,        // in floats
					gpu_corrs);              // float      * gpu_corrs);          // correlation output data (either pixel domain or transform domain
		}
	}
}

/**
 * Used for interscene correlations (for motion vector calculation).
 * Calculate sum of selected correlation (in TD) and place it after individual (per-sensor) correlations.
 * Configuration
 *    threads: dim3 (CORR_THREADS_PER_TILE, CORR_TILES_PER_BLOCK, 1)
 *    grids:   dim3 ((number_of_task_tiles + CORR_TILES_PER_BLOCK-1) / CORR_TILES_PER_BLOCK,1,1)
 *
 * @param num_sel_sensors  number of sensors to correlate
 * @param num_corr_tiles   number of correlation tiles to process (here it includes sum)
 * @param gpu_corr_indices packed tile+pair, similar format as intrascene (tile number  << 8), low byte
 *                         is now sensor number or 0xff (last one for each tile) for the sum of all individual
 *                         correlations. Entries for each tile go in the same order (increasing sensor number)
 *                         followed by the sum of all the selected correlations. Entries for different tiles
 *                         are not ordered.
 * @param corr_stride      stride (in floats) for correlation outputs.
 * @param gpu_corrs        allocated array for the correlation output data, first num_sel_sensors for each tile
 *                         should be calculated by correlate2D_inter_inner() leaving gaps for sums, calculated here
 *
 */
extern "C" __global__ void combine_inter(     // combine per-senor interscene correlations
		int               num_sel_sensors,    // number of sensors to correlate (not counting sum of all)
		int               num_corr_tiles,     // number of correlation tiles to process (here it includes sum)
		int             * gpu_corr_indices,   // packed tile+pair
		size_t            corr_stride,        // in floats
		float           * gpu_corrs)          // correlation output data (either pixel domain or transform domain
{
	int corr_in_block = threadIdx.y;
	int itile = blockIdx.x * CORR_TILES_PER_BLOCK + corr_in_block; // correlation tile index
	int corr_offset = itile * (num_sel_sensors + 1); // index of the first correlation for this task;
	if (corr_offset >= (num_corr_tiles - num_sel_sensors)) {
		return;
	}
//    __syncthreads();// __syncwarp();
    __shared__ float clt_corrs   [CORR_TILES_PER_BLOCK][4][DTT_SIZE][DTT_SIZE1];
    float * clt_corr =  ((float *) clt_corrs) +  corr_in_block * (4 * DTT_SIZE * DTT_SIZE1); // top left quadrant0
    resetCorrelation(clt_corr);
    __syncthreads(); /// ***** Was not here: probably not needed
    for (int isens = 0; isens < num_sel_sensors; isens++){
        float *mem_corr = gpu_corrs + corr_stride * corr_offset + threadIdx.x;
        float *clt = clt_corr + threadIdx.x;
    #pragma unroll
        for (int i = 0; i < DTT_SIZE4; i++){
        	(*clt) += (*mem_corr);
        	clt        += DTT_SIZE1;
        	mem_corr   += DTT_SIZE;
        }
        corr_offset++;
    }
    // Now corr_offset points to the sum of correlations
    float *mem_corr = gpu_corrs + corr_stride * corr_offset + threadIdx.x;
    float *clt = clt_corr + threadIdx.x;
#pragma unroll
    for (int i = 0; i < DTT_SIZE4; i++){
    	(*mem_corr) = (*clt);
    	clt        += DTT_SIZE1;
    	mem_corr   += DTT_SIZE;
    }

}

/**
 * Calculate interscene 2D phase correlation pairs from CLT representation.
 * This is an inner kernel that is called from correlate2D_inter.
 * Configuration
 *    threads: dim3 (CORR_THREADS_PER_TILE, CORR_TILES_PER_BLOCK, 1)
 *    grids:   dim3 ((number_of_corr_tiles_excluding_sums + CORR_TILES_PER_BLOCK-1) / CORR_TILES_PER_BLOCK,1,1)
 *
 * @param num_cams         number of cameras
 * @param num_sel_sensors  number of sensors to correlate
 * @param gpu_clt          array of num_cams pointers to the CLT (frequency domain) data [TILES-Y][TILES-X][NUM_COLORS][DTT_SIZE*DTT_SIZE]
 * @param gpu_clt_ref      array of num_cams pointers to the CLT (frequency domain) data [TILES-Y][TILES-X][NUM_COLORS][DTT_SIZE*DTT_SIZE]
 *                         gpu_clt_ref and gpu_clt_ref correspond to two scenes, the reference is the first in correlation.
 * @param colors           number of colors used:  3 for RGB or 1 for monochrome
 * @param scale0           scale red (or mono) component before mixing
 * @param scale1           scale blue (if colors = 3) component before mixing
 * @param scale2           scale green (if colors = 3) component before mixing
 * @param num_corr_tiles   number of correlation tiles to process (here it includes sum)
 * @param gpu_corr_indices packed tile+pair, similar format as intrascene (tile number  << 8), low byte
 *                         is now sensor number or 0xff (last one for each tile) for the sum of all individual
 *                         correlations. Entries for each tile go in the same order (increasing sensor number)
 *                         followed by the sum of all the selected correlations. Entries for different tiles
 *                         are not ordered.
 * @param corr_stride      stride (in floats) for correlation outputs.
 * @param gpu_corrs        allocated array for the correlation output data, first num_sel_sensors for each tile
 *                         will be calculated here leaving gaps for sums, calculated by combine_inter()
 */
extern "C" __global__ void correlate2D_inter_inner( // will only process to TD, no normalisations and back conversion
		int               num_cams,
		int               num_sel_sensors,    // number of sensors to correlate (not counting sum of all)
		float          ** gpu_clt,            // [num_cams] ->[TILES-Y][TILES-X][colors][DTT_SIZE*DTT_SIZE]
		float          ** gpu_clt_ref,        // [num_cams] ->[TILES-Y][TILES-X][colors][DTT_SIZE*DTT_SIZE]
		int               colors,             // number of colors (3/1)
		float             scale0,             // scale for R
		float             scale1,             // scale for B
		float             scale2,             // scale for G
		int               num_corr_tiles,     // number of correlation tiles to process (here it includes sum)
		int             * gpu_corr_indices,   // packed tile+pair
		size_t            corr_stride,        // in floats
		float           * gpu_corrs)          // correlation output data (either pixel domain or transform domain
{
	float scales[3] = {scale0, scale1, scale2};
	int corr_in_block = threadIdx.y;
	int corr_num = blockIdx.x * CORR_TILES_PER_BLOCK + corr_in_block; // 4
	int tile_index =  corr_num / num_sel_sensors;
	int corr_offset = tile_index + corr_num; // added for missing sum correlation tiles.
	if (corr_offset >= num_corr_tiles){
		return; // nothing to do
	}

	// get number of pair and number of tile
	int corr_sensor = gpu_corr_indices[corr_offset]; // corr_num];

	int tile_num = corr_sensor >> CORR_NTILE_SHIFT;
	corr_sensor &= (corr_sensor & ((1 << CORR_NTILE_SHIFT) - 1));
	if (corr_sensor >= num_cams){
		return; // BUG - should not happen
	}

    __syncthreads();// __syncwarp();
    __shared__ float clt_tiles1  [CORR_TILES_PER_BLOCK][4][DTT_SIZE][DTT_SIZE1];
    __shared__ float clt_tiles2  [CORR_TILES_PER_BLOCK][4][DTT_SIZE][DTT_SIZE1];
    __shared__ float clt_corrs   [CORR_TILES_PER_BLOCK][4][DTT_SIZE][DTT_SIZE1];
//    __shared__ float mlt_corrs   [CORR_TILES_PER_BLOCK][DTT_SIZE2M1][DTT_SIZE2M1]; // result correlation
    // set clt_corr to all zeros
    float * clt_corr =  ((float *) clt_corrs) +  corr_in_block * (4 * DTT_SIZE * DTT_SIZE1); // top left quadrant0
//    float * mclt_corr = ((float *) mlt_corrs) +  corr_in_block * (DTT_SIZE2M1*DTT_SIZE2M1);
    resetCorrelation(clt_corr);
    __syncthreads(); /// ***** Was not here: probably not needed
    for (int color = 0; color < colors; color++){
        // copy clt (frequency domain data)
        float * clt_tile1 = ((float *) clt_tiles1) +  corr_in_block * (4 * DTT_SIZE * DTT_SIZE1);
        float * clt_tile2 = ((float *) clt_tiles2) +  corr_in_block * (4 * DTT_SIZE * DTT_SIZE1);
        int offs = (tile_num * colors + color) * (4 * DTT_SIZE * DTT_SIZE) + threadIdx.x;
        float * gpu_tile1 = ((float *) gpu_clt_ref[corr_sensor]) + offs;
        float * gpu_tile2 = ((float *) gpu_clt    [corr_sensor]) + offs;
		float * clt_tile1i = clt_tile1 + threadIdx.x;
		float * clt_tile2i = clt_tile2 + threadIdx.x;
#pragma unroll
#define USE_LOG
#ifdef 	USE_LOG
		// Apply high-pass filter to correlation inputs to reduce dynamic range before multiplication
		for (int q = 0; q < 4; q++){
    		float *log = LoG_corr + threadIdx.x;
			for (int i = 0; i < DTT_SIZE; i++){ // copy 32 rows (4 quadrants of 8 rows)
				*clt_tile1i= (*gpu_tile1) * (*log);
				*clt_tile2i= (*gpu_tile2) * (*log);
				clt_tile1i += DTT_SIZE1;
				clt_tile2i += DTT_SIZE1;
				gpu_tile1 += DTT_SIZE;
				gpu_tile2 += DTT_SIZE;
				log +=       DTT_SIZE;
			}
		}
#else
		for (int i = 0; i < DTT_SIZE4; i++){ // copy 32 rows (4 quadrants of 8 rows)
			*clt_tile1i= *gpu_tile1;
			*clt_tile2i= *gpu_tile2;
			clt_tile1i += DTT_SIZE1;
			clt_tile2i += DTT_SIZE1;
			gpu_tile1 += DTT_SIZE;
			gpu_tile2 += DTT_SIZE;
    	}
#endif //USE_LOG
		__syncthreads();
		// each thread should get the same pointers here, offsets are inside
        correlateAccumulateTiles(
        		scales[color], // float  scale,     // scale correlation
				clt_tile1, // float* clt_tile1, //    [4][DTT_SIZE][DTT_SIZE1], // 4 quadrants of the clt data 1, rows extended to optimize shared ports
				clt_tile2, // float* clt_tile2, //    [4][DTT_SIZE][DTT_SIZE1], // 4 quadrants of the clt data 2, rows extended to optimize shared ports
				clt_corr); // float* corr_tile) //    [4][DTT_SIZE][DTT_SIZE1]) // 4 quadrants of the correlation result
        __syncthreads();

        if (color == 1){ // LPF only after B (nothing in mono)
        	float *clt = clt_corr + threadIdx.x;
#pragma unroll
        	for (int q = 0; q < 4; q++){
        		float *lpf_rb = lpf_rb_corr + threadIdx.x;
#pragma unroll
        		for (int i = 0; i < DTT_SIZE; i++){
        			(*clt) *= (*lpf_rb);
        			clt    += DTT_SIZE1;
        			lpf_rb += DTT_SIZE;
        		}
        	}
        	__syncthreads();// __syncwarp();
        } // if (color == 1){ // LPF only after B (nothing in mono)
    } // for (int color = 0; color < colors; color++){
	__syncthreads();// __syncwarp();
    float *mem_corr = gpu_corrs + corr_stride * corr_offset + threadIdx.x;
    float *clt = clt_corr + threadIdx.x;
#pragma unroll
    for (int i = 0; i < DTT_SIZE4; i++){
    	(*mem_corr) = (*clt);
    	clt        += DTT_SIZE1;
    	mem_corr   += DTT_SIZE;
    }
    __syncthreads();// __syncwarp();
}





/**
 * Calculate  2D phase correlation pairs from CLT representation. This is an inner kernel that is called
 * from correlate2D. If called from the CPU: <<<ceil(number_of_tiles/32),32>>>.
 * If corr_radius==0, skip normalization and inverse transform, output transform domain tiles
 *
 * @param num_cams         number of cameras
 * @param gpu_clt          array of num_cams pointers to the CLT (frequency domain) data [TILES-Y][TILES-X][NUM_COLORS][DTT_SIZE*DTT_SIZE]
 * @param colors           number of colors used:  3 for RGB or 1 for monochrome
 * @param scale0           scale red (or mono) component before mixing
 * @param scale1           scale blue (if colors = 3) component before mixing
 * @param scale2           scale green (if colors = 3) component before mixing
 * @param fat_zero2        add this value squared to the sum of squared components before normalization
 * @param num_corr_tiles   number of correlation tiles to process
 * @param gpu_corr_indices packed array (each element, integer contains tile+pair) of correlation tasks
 * @param corr_stride      stride (in floats) for correlation outputs.
 * @param corr_radius      radius of the output correlation (maximal 7 for 15x15). If 0 - output Transform Domain tiles, no normalization
 * @param gpu_corrs        allocated array for the correlation output data (each element stride, payload: (2*corr_radius+1)^2
 */
extern "C" __global__ void correlate2D_inner(
		int               num_cams,
		float          ** gpu_clt,            // [num_cams] ->[TILES-Y][TILES-X][colors][DTT_SIZE*DTT_SIZE]
		int               colors,             // number of colors (3/1)
		float             scale0,             // scale for R
		float             scale1,             // scale for B
		float             scale2,             // scale for G
		float             fat_zero2,          // here - absolute
		int            num_corr_tiles,     // number of correlation tiles to process
		int             * gpu_corr_indices,   // packed tile+pair
		size_t         corr_stride,        // in floats
		int               corr_radius0,        // radius of the output correlation (7 for 15x15)
		float           * gpu_corrs)          // correlation output data (either pixel domain or transform domain
{
//	int corr_radius = corr_radius0 & 0x1f;// minimal "bad"
//	int corr_radius = corr_radius0 & 0xf; // maximal "good"
	int corr_radius = corr_radius0 & 0x7; // actual never >7. Still did not understand where is the problem,
	// providing literal "7" in the call does not fix the problem
	float scales[3] = {scale0, scale1, scale2};
	int corr_in_block = threadIdx.y;
	int corr_num = blockIdx.x * CORR_TILES_PER_BLOCK + corr_in_block; // 4
	if (corr_num >= num_corr_tiles){
		return; // nothing to do
	}
	int pair_list_start = pairs_offsets[num_cams];
	int pair_list_len =   pairs_offsets[num_cams + 1] - pair_list_start;
	// get number of pair and number of tile
	int corr_pair = gpu_corr_indices[corr_num];
	int tile_num = corr_pair >> CORR_NTILE_SHIFT;
	corr_pair &= (corr_pair & ((1 << CORR_NTILE_SHIFT) - 1));
	if (corr_pair > pair_list_len){
		return; // BUG - should not happen
	}
	int cam1 = all_pairs[pair_list_start + corr_pair][0]; // number of the first camera in a pair
	int cam2 = all_pairs[pair_list_start + corr_pair][1]; // number of the second camera in a pair
    __syncthreads();// __syncwarp();
    __shared__ float clt_tiles1  [CORR_TILES_PER_BLOCK][4][DTT_SIZE][DTT_SIZE1];
    __shared__ float clt_tiles2  [CORR_TILES_PER_BLOCK][4][DTT_SIZE][DTT_SIZE1];
    __shared__ float clt_corrs   [CORR_TILES_PER_BLOCK][4][DTT_SIZE][DTT_SIZE1];
    __shared__ float mlt_corrs   [CORR_TILES_PER_BLOCK][DTT_SIZE2M1][DTT_SIZE2M1]; // result correlation
    // set clt_corr to all zeros
    float * clt_corr =  ((float *) clt_corrs) +  corr_in_block * (4 * DTT_SIZE * DTT_SIZE1); // top left quadrant0
    float * mclt_corr = ((float *) mlt_corrs) +  corr_in_block * (DTT_SIZE2M1*DTT_SIZE2M1);
    resetCorrelation(clt_corr);
    __syncthreads(); /// ***** Was not here: probably not needed
    for (int color = 0; color < colors; color++){
        // copy clt (frequency domain data)
        float * clt_tile1 = ((float *) clt_tiles1) +  corr_in_block * (4 * DTT_SIZE * DTT_SIZE1);
        float * clt_tile2 = ((float *) clt_tiles2) +  corr_in_block * (4 * DTT_SIZE * DTT_SIZE1);
        int offs = (tile_num * colors + color) * (4 * DTT_SIZE * DTT_SIZE) + threadIdx.x;
        float * gpu_tile1 = ((float *) gpu_clt[cam1]) + offs;
        float * gpu_tile2 = ((float *) gpu_clt[cam2]) + offs;
		float * clt_tile1i = clt_tile1 + threadIdx.x;
		float * clt_tile2i = clt_tile2 + threadIdx.x;
#pragma unroll
#define USE_LOG
#ifdef 	USE_LOG
		// Apply high-pass filter to correlation inputs to reduce dynamic range before multiplication
		for (int q = 0; q < 4; q++){
    		float *log = LoG_corr + threadIdx.x;
			for (int i = 0; i < DTT_SIZE; i++){ // copy 32 rows (4 quadrants of 8 rows)
				*clt_tile1i= (*gpu_tile1) * (*log);
				*clt_tile2i= (*gpu_tile2) * (*log);
				clt_tile1i += DTT_SIZE1;
				clt_tile2i += DTT_SIZE1;
				gpu_tile1 += DTT_SIZE;
				gpu_tile2 += DTT_SIZE;
				log +=       DTT_SIZE;
			}
		}
#else
		for (int i = 0; i < DTT_SIZE4; i++){ // copy 32 rows (4 quadrants of 8 rows)
			*clt_tile1i= *gpu_tile1;
			*clt_tile2i= *gpu_tile2;
			clt_tile1i += DTT_SIZE1;
			clt_tile2i += DTT_SIZE1;
			gpu_tile1 += DTT_SIZE;
			gpu_tile2 += DTT_SIZE;
    	}
#endif //USE_LOG
		__syncthreads();
#ifdef DBG_TILE
#ifdef DEBUG6
    if ((tile_num == DBG_TILE) && (corr_pair == 0) && (threadIdx.x == 0)){
        printf("\ncorrelate2D tile = %d, pair=%d, color = %d   CAMERA1\n",tile_num, corr_pair,color);
    	debug_print_clt1(clt_tile1, color,  0xf); //
        printf("\ncorrelate2D tile = %d, pair=%d, color = %d   CAMERA2\n",tile_num, corr_pair,color);
    	debug_print_clt1(clt_tile2, color,  0xf); //
    }
     __syncthreads();// __syncwarp();
#endif
#endif
		// each thread should get the same pointers here, offsets are inside
        correlateAccumulateTiles(
        		scales[color], // float  scale,     // scale correlation
				clt_tile1, // float* clt_tile1, //    [4][DTT_SIZE][DTT_SIZE1], // 4 quadrants of the clt data 1, rows extended to optimize shared ports
				clt_tile2, // float* clt_tile2, //    [4][DTT_SIZE][DTT_SIZE1], // 4 quadrants of the clt data 2, rows extended to optimize shared ports
				clt_corr); // float* corr_tile) //    [4][DTT_SIZE][DTT_SIZE1]) // 4 quadrants of the correlation result
        __syncthreads();

#ifdef DBG_TILE
#ifdef DEBUG6
        if ((tile_num == DBG_TILE) && (corr_pair == 0) && (threadIdx.x == 0)){
        	printf("\ncorrelate2D, color = %d CORRELATION\n", color);
        	debug_print_clt1(clt_corr, color,  0xf);
        }
        __syncthreads();// __syncwarp();
#endif
#endif
        if (color == 1){ // LPF only after B (nothing in mono)

#ifdef DBG_TILE
#ifdef DEBUG6
        	if ((tile_num == DBG_TILE) && (corr_pair == 0) && (threadIdx.x == 0)){
        		printf("\ncorrelate2D LPF for RB correlation\n");
        		debug_print_lpf(lpf_rb_corr);
        	}
        	__syncthreads();// __syncwarp();
#endif
#endif

        	float *clt = clt_corr + threadIdx.x;
#pragma unroll
        	for (int q = 0; q < 4; q++){
        		float *lpf_rb = lpf_rb_corr + threadIdx.x;
#pragma unroll
        		for (int i = 0; i < DTT_SIZE; i++){
        			(*clt) *= (*lpf_rb);
        			clt    += DTT_SIZE1;
        			lpf_rb += DTT_SIZE;
        		}
        	}
        	__syncthreads();// __syncwarp();
#ifdef DBG_TILE
#ifdef DEBUG6
        	if ((tile_num == DBG_TILE) && (corr_pair == 0) && (threadIdx.x == 0)){
        		printf("\ncorrelate2D CORRELATION RB LPF-ed\n");
        		debug_print_clt1(clt_corr, -1,  0xf);
        	}
        	__syncthreads();// __syncwarp();
#endif
#endif
        } // if (color == 1){ // LPF only after B (nothing in mono)
    } // for (int color = 0; color < colors; color++){
//	corr_radius = 7;
    // Skip normalization, lpf, inverse correction and unfolding if Transform Domain output is required
    if (corr_radius > 0) {
    	normalizeTileAmplitude(
    			clt_corr,   // float * clt_tile, //       [4][DTT_SIZE][DTT_SIZE1], // +1 to alternate column ports
				fat_zero2); // float fat_zero2 ) // fat zero is absolute, scale it outside
    	// Low Pass Filter from constant area (is it possible to replace?)
        __syncthreads(); /// ***** Was not here: probably not needed

#ifdef DBG_TILE
#ifdef DEBUG6
    	if ((tile_num == DBG_TILE) && (corr_pair == 0) && (threadIdx.x == 0)){
    		printf("\ncorrelate2D CORRELATION NORMALIZED, fat_zero2=%f\n",fat_zero2);
    		debug_print_clt1(clt_corr, -1,  0xf);
    	}
    	__syncthreads();// __syncwarp();
#endif
#endif

#ifdef DBG_TILE
#ifdef DEBUG6
    	if ((tile_num == DBG_TILE) && (corr_pair == 0) && (threadIdx.x == 0)){
    		printf("\ncorrelate2D LPF\n");
    		debug_print_lpf(lpf_corr);
    	}
    	__syncthreads();// __syncwarp();
#endif
#endif

    	float *clt = clt_corr + threadIdx.x;
#pragma unroll
    	for (int q = 0; q < 4; q++){
    		float *lpf = lpf_corr + threadIdx.x;
#pragma unroll
    		for (int i = 0; i < DTT_SIZE; i++){
    			(*clt) *= (*lpf);
    			clt   += DTT_SIZE1;
    			lpf   += DTT_SIZE;
    		}
    	}
    	__syncthreads();// __syncwarp();
//    	corr_radius = 7;

#ifdef DBG_TILE
#ifdef DEBUG6
    	if ((tile_num == DBG_TILE) && (corr_pair == 0) && (threadIdx.x == 0)){
    		printf("\ncorrelate2D CORRELATION LPF-ed\n");
    		debug_print_clt1(clt_corr, -1,  0xf);
    	}
    	__syncthreads();// __syncwarp();
#endif
#endif
    	dttii_2d(clt_corr);
//    has __syncthreads() inside

#ifdef DBG_TILE
#ifdef DEBUG6
    	if ((tile_num == DBG_TILE) && (corr_pair == 0) && (threadIdx.x == 4)){
    		printf("\ncorrelate2D AFTER HOSIZONTAL (VERTICAL) PASS, corr_radius=%d\n",corr_radius);
    		debug_print_clt1(clt_corr, -1,  0xf);
    	}
    	__syncthreads();// __syncwarp();

#endif
#endif
    	__syncthreads();
//    	corr_radius = 7;
    	corrUnfoldTile(
    			corr_radius, // int corr_radius,
				(float *) clt_corr,  // float* qdata0, //    [4][DTT_SIZE][DTT_SIZE1], // 4 quadrants of the clt data, rows extended to optimize shared ports
				(float *) mclt_corr); // float* rslt)  //   [DTT_SIZE2M1][DTT_SIZE2M1]) // 15x15
    	__syncthreads();

#ifdef DBG_TILE
#ifdef DEBUG6
    	if ((tile_num == DBG_TILE) && (corr_pair == 0) && (threadIdx.x == 0)){
    		printf("\ncorrelate2D after UNFOLD, corr_radius=%d\n",corr_radius);
    		debug_print_corr_15x15(
    				corr_radius, // int     corr_radius,
					mclt_corr,
					-1);
    	}
    	__syncthreads();// __syncwarp();
#endif
#endif

    	// copy 15x15 tile to main memory (2 * corr_radius +1) x (2 * corr_radius +1)
    	int size2r1 = 2 * corr_radius + 1;  //  15 for full corr tile
    	int len2r1x2r1 = size2r1 * size2r1; // 225 for full corr tile
    	int corr_tile_offset =  + corr_stride * corr_num;
    	float *mem_corr = gpu_corrs + corr_tile_offset;
#pragma unroll
    	//    for (int offs = threadIdx.x; offs < DTT_SIZE2M1*DTT_SIZE2M1; offs+=CORR_THREADS_PER_TILE){ // variable number of cycles per thread
    	for (int offs = threadIdx.x; offs < len2r1x2r1; offs+=CORR_THREADS_PER_TILE){ // variable number of cycles per thread
    		mem_corr[offs] = mclt_corr[offs]; // copy OK
    	}

    	__syncthreads();
#ifdef DBG_TILE
#ifdef DEBUG6
    	if ((tile_num == DBG_TILE) && (corr_pair == 0) && (threadIdx.x == 0)){
    		printf("\ncorrelate2D after copy to main memory\n");
    		//    	debug_print_clt1(clt_corr, -1,  0xf);
    	}
    	__syncthreads();// __syncwarp();
#endif
#endif
    } else { //     if (corr_radius > 0) { transform domain output
    	//    	int corr_tile_offset =  + corr_stride * corr_num;
    	float *mem_corr = gpu_corrs + corr_stride * corr_num + threadIdx.x;
    	float *clt = clt_corr + threadIdx.x;
#pragma unroll
    	for (int i = 0; i < DTT_SIZE4; i++){
    		(*mem_corr) = (*clt);
    		clt        += DTT_SIZE1;
    		mem_corr   += DTT_SIZE;
    	}
    	__syncthreads();// __syncwarp();
    } //     if (corr_radius > 0) ... else

}


/**
 * Combine multiple correlation pairs for quad (square) camera: 2 or 4 ortho into a single clt tile,
 * and separately the two diagonals into another single one
 * When adding vertical pairs to the horizontal, each quadrant is transposed, and the Q1 and Q2 are also swapped.
 * when combining two diagonals (down-right and up-right), the data in quadrants Q2 and Q3 is negated
 * (corresponds to a vertical flip).
 * Data can be added to the existing one (e.g. for the inter-scene accumulation of the compatible correlations).
 * This is an outer kernel that calls the inner one with CDP, this one should be configured as corr2D_combine<<<1,1>>>
 *
 * @param num_tiles,          // number of tiles to process (each with num_pairs)
 * @param num_pairs,          // num pairs per tile (should be the same)
 * @param init_output,        // & 1 - reset output tiles to zero before accumulating, &2 no transpose
 * @param pairs_mask,         // selected pairs (0x3 - horizontal, 0xc - vertical, 0xf - quad, 0x30 - cross)
 * @param gpu_corr_indices,   // packed tile+pair
 * @param gpu_combo_indices,  // output if noty null: packed tile+pairs_mask (will point to the first used pair
 * @param corr_stride,        // (in floats) stride for the input TD correlations
 * @param gpu_corrs,          // input correlation tiles
 * @param corr_stride_combo,  // (in floats) stride for the output TD correlations (same as input)
 * @param gpu_corrs_combo)    // combined correlation output (one per tile)
 */
extern "C" __global__ void corr2D_combine(
		int               num_tiles,          // number of tiles to process (each with num_pairs)
		int               num_pairs,          // num pairs per tile (should be the same)
		int               init_output,        // !=0 - reset output tiles to zero before accumulating
		int               pairs_mask,         // selected pairs (0x3 - horizontal, 0xc - vertical, 0xf - quad, 0x30 - cross)
		int             * gpu_corr_indices,   // packed tile+pair
		int             * gpu_combo_indices,  // output if noty null: packed tile+pairs_mask (will point to the first used pair
		const size_t      corr_stride,        // (in floats) stride for the input TD correlations
		float           * gpu_corrs,          // input correlation tiles
		const size_t      corr_stride_combo,  // (in floats) stride for the output TD correlations (same as input)
		float           * gpu_corrs_combo)    // combined correlation output (one per tile)
{
	 if (threadIdx.x == 0) { // only 1 thread, 1 block
		    dim3 threads_corr(CORR_THREADS_PER_TILE, CORR_TILES_PER_BLOCK_COMBINE, 1);
		    dim3 grid_corr((num_tiles + CORR_TILES_PER_BLOCK_COMBINE-1) / CORR_TILES_PER_BLOCK_COMBINE,1,1);
		    corr2D_combine_inner<<<grid_corr,threads_corr>>>(
		    		num_tiles,          // int               num_tiles,          // number of tiles to process (each with num_pairs)
					num_pairs,          // int               num_pairs,          // num pairs per tile (should be the same)
					init_output,        // int               init_output,        // !=0 - reset output tiles to zero before accumulating
					pairs_mask,         // int               pairs_mask,         // selected pairs (0x3 - horizontal, 0xc - vertical, 0xf - quad, 0x30 - cross)
					gpu_corr_indices,   // int             * gpu_corr_indices,   // packed tile+pair
					gpu_combo_indices,  // int             * gpu_combo_indices,  // output if noty null: packed tile+pairs_mask (will point to the first used pair
					corr_stride,        // const size_t      corr_stride,        // (in floats) stride for the input TD correlations
					gpu_corrs,          // float           * gpu_corrs,          // input correlation tiles
					corr_stride_combo,  // const size_t      corr_stride_combo,  // (in floats) stride for the output TD correlations (same as input)
					gpu_corrs_combo);    // float           * gpu_corrs_combo)    // combined correlation output (one per tile)
	 }
}

//#define CORR_TILES_PER_BLOCK_COMBINE   4 // increase to 8?
#define PAIRS_HOR_DIAG_MAIN 0x13
#define PAIRS_VERT          0x0c
#define PAIRS_DIAG_OTHER    0x20
/**
 * Combine multiple correlation pairs for quad (square) camera: 2 or 4 ortho into a single clt tile,
 * and separately the two diagonals into another single one
 * When adding vertical pairs to the horizontal, each quadrant is transposed, and the Q1 and Q2 are also swapped.
 * when combining tho diagonals (down-right and up-right), the data in quadrants Q2 and Q3 is negated
 * (corresponds to a vertical flip).
 * Data can be added to the existing one (e.g. for the inter-scene accumulation of the compatible correlations).
 * This is an inner kernel that is called from corr2D_combine.
 *
 * @param num_tiles,          // number of tiles to process (each with num_pairs)
 * @param num_pairs,          // num pairs per tile (should be the same)
 * @param init_output,        // !=0 - reset output tiles to zero before accumulating
 * @param pairs_mask,         // selected pairs (0x3 - horizontal, 0xc - vertical, 0xf - quad, 0x30 - cross)
 * @param gpu_corr_indices,   // packed tile+pair
 * @param gpu_combo_indices,  // output if noty null: packed tile+pairs_mask (will point to the first used pair
 * @param corr_stride,        // (in floats) stride for the input TD correlations
 * @param gpu_corrs,          // input correlation tiles
 * @param corr_stride_combo,  // (in floats) stride for the output TD correlations (same as input)
 * @param gpu_corrs_combo)    // combined correlation output (one per tile)
 */
extern "C" __global__ void corr2D_combine_inner(
		int               num_tiles,          // number of tiles to process (each with num_pairs)
		int               num_pairs,          // num pairs per tile (should be the same)
		int               init_output,        // !=0 - reset output tiles to zero before accumulating
		int               pairs_mask,         // selected pairs (0x3 - horizontal, 0xc - vertical, 0xf - quad, 0x30 - cross)
		int             * gpu_corr_indices,   // packed tile+pair
		int             * gpu_combo_indices,  // output if noty null: packed tile+pairs_mask (will point to the first used pair
		const size_t      corr_stride,        // (in floats) stride for the input TD correlations
		float           * gpu_corrs,          // input correlation tiles
		const size_t      corr_stride_combo,  // (in floats) stride for the output TD correlations (same as input)
		float           * gpu_corrs_combo)    // combined correlation output (one per tile)
{
	int tile_in_block = threadIdx.y;
	int tile_index = blockIdx.x * CORR_TILES_PER_BLOCK_COMBINE + tile_in_block;
	if (tile_index >= num_tiles){
		return; // nothing to do
	}
	int corr_tile_index0 = tile_index * num_pairs;
	if (gpu_combo_indices != 0){
		int corr_pair = gpu_corr_indices[corr_tile_index0];
		gpu_combo_indices[tile_index] = ((corr_pair >> CORR_NTILE_SHIFT) << CORR_NTILE_SHIFT) | pairs_mask;
	}
	float scale = 1.0/__popc(pairs_mask); // reverse to number of pairs to combine
    __syncthreads();// __syncwarp();
    __shared__ float clt_corrs   [CORR_TILES_PER_BLOCK_COMBINE][4][DTT_SIZE][DTT_SIZE1];
    // start of the block in shared memory
    float *clt_corr =  ((float *) clt_corrs) +  tile_in_block * (4 * DTT_SIZE * DTT_SIZE1); // top left quadrant0
	float *clt = clt_corr + threadIdx.x;
	float *mem_corr = gpu_corrs_combo + corr_stride_combo * tile_index + threadIdx.x;

	if (init_output & 1){ // reset combo
#pragma unroll
		for (int i = 0; i < DTT_SIZE4; i++){
			(*clt)         = 0.0f;
			clt           += DTT_SIZE1;
		}
	} else { // read previous from device memory
#pragma unroll
		for (int i = 0; i < DTT_SIZE4; i++){
			(*clt)      = (*mem_corr);
			clt        += DTT_SIZE1;
			mem_corr   += DTT_SIZE;
		}
	}
	__syncthreads();// __syncwarp();


	for (int ipair = 0; ipair < num_pairs; ipair++){ // only selected
		int corr_tile_index = corr_tile_index0 + ipair;
		// get number of pair
		int corr_pair = gpu_corr_indices[corr_tile_index];
//		int tile_num = corr_pair >> CORR_NTILE_SHIFT;
		corr_pair &= (corr_pair & ((1 << CORR_NTILE_SHIFT) - 1));
		int pair_bit = 1 << corr_pair;
		if ((pairs_mask & pair_bit) != 0) {
//			if (corr_pair > NUM_PAIRS){
//				return; // BUG - should not happen
//			}
			if ((PAIRS_HOR_DIAG_MAIN & pair_bit) || (init_output & 2)){ // just accumulate. This if-s will branch in all threads, no diversion
				clt = clt_corr + threadIdx.x;
				mem_corr = gpu_corrs + corr_stride_combo * corr_tile_index + threadIdx.x;
#pragma unroll
				for (int i = 0; i < DTT_SIZE4; i++){
					(*clt)     += (*mem_corr);
					clt        += DTT_SIZE1;
					mem_corr   += DTT_SIZE;
				}

			} else if (PAIRS_VERT & pair_bit) { // transpose and swap Q1 and Q2
				for (int q = 0; q < 4; q++){
					int qr = ((q & 1) << 1) | ((q >> 1) & 1);
					clt = clt_corr + qr * (DTT_SIZE1 * DTT_SIZE) + threadIdx.x;
					mem_corr = gpu_corrs + corr_stride_combo * corr_tile_index + q * (DTT_SIZE * DTT_SIZE) + DTT_SIZE * threadIdx.x;
#pragma unroll
					for (int i = 0; i < DTT_SIZE; i++){
						(*clt)     += (*mem_corr);
						clt        += DTT_SIZE1;
						mem_corr   += 1;
					}
				}

			} else if (PAIRS_DIAG_OTHER & pair_bit) {
				clt = clt_corr + threadIdx.x;
				mem_corr = gpu_corrs + corr_stride_combo * corr_tile_index + threadIdx.x;
#pragma unroll
				for (int i = 0; i < DTT_SIZE2; i++){ // CC, CS
					(*clt)     += (*mem_corr);
					clt        += DTT_SIZE1;
					mem_corr   += DTT_SIZE;
				}
#pragma unroll
				for (int i = 0; i < DTT_SIZE2; i++){ // SC, SS
					(*clt)     -= (*mem_corr); // negate
					clt        += DTT_SIZE1;
					mem_corr   += DTT_SIZE;
				}

			} //PAIRS_DIAG_OTHER
		}
	} //for (int ipair = 0; ipair < num_pairs; ipair++){ // only selected
	__syncthreads();// __syncwarp();
	// copy result to the device memory

	clt = clt_corr + threadIdx.x;
	mem_corr = gpu_corrs_combo + corr_stride_combo * tile_index + threadIdx.x;
#pragma unroll
	for (int i = 0; i < DTT_SIZE4; i++){
		(*mem_corr) = (*clt);
		clt        += DTT_SIZE1;
		mem_corr   += DTT_SIZE;
	}
	__syncthreads();// __syncwarp();
}

/**
 * Normalize, low-pass filter, convert to pixel domain and unfold correlation tiles.This is an outer kernel
 * that calls the inner one with CDP, this one should be configured as correlate2D<<<1,1>>>
 *
 * @param num_corr_tiles   number of correlation tiles to process
 * @param corr_stride_td,  stride (in floats) for correlation input (transform domain).
 * @param gpu_corrs_td     correlation data in transform domain
 * @param corr_weights     null or per-tile weight (fat_zero2 will be divided by it), length = num_corr_tiles
 * @param corr_stride,     stride (in floats) for correlation pixel-domain outputs.
 * @param gpu_corrs        allocated array for the correlation output data (each element stride, payload: (2*corr_radius+1)^2
 * @param fat_zero2        add this value squared to the sum of squared components before normalization (squared)
 * @param corr_radius,     radius of the output correlation (maximal 7 for 15x15)
 */
extern "C" __global__ void corr2D_normalize(
		int               num_corr_tiles,     // number of correlation tiles to process
		const size_t      corr_stride_td,     // in floats
		float           * gpu_corrs_td,       // correlation tiles in transform domain
		float           * corr_weights,       // null or per correlation tile weight (fat_zero2 will be divided by it), length = num_corr_tile
		const size_t      corr_stride,        // in floats
		float           * gpu_corrs,          // correlation output data (either pixel domain or transform domain
		float             fat_zero2,          // here - absolute, squared
		int               corr_radius)        // radius of the output correlation (7 for 15x15)
{
	 if (threadIdx.x == 0) { // only 1 thread, 1 block
		    dim3 threads_corr(CORR_THREADS_PER_TILE, CORR_TILES_PER_BLOCK_NORMALIZE, 1);
		    dim3 grid_corr((num_corr_tiles + CORR_TILES_PER_BLOCK_NORMALIZE-1) / CORR_TILES_PER_BLOCK_NORMALIZE,1,1);
		    corr2D_normalize_inner<<<grid_corr,threads_corr>>>(
	        		num_corr_tiles,      // int               num_corr_tiles,     // number of correlation tiles to process
					corr_stride_td,      // const size_t      corr_stride,        // in floats
					gpu_corrs_td,        // float           * gpu_corrs_td,       // correlation tiles in transform domain
					corr_weights,        // float           * corr_weights,       // null or per-tile weight (fat_zero2 will be divided by it)
					corr_stride,         // const size_t      corr_stride,        // in floats
					gpu_corrs,           // float           * gpu_corrs,          // correlation output data (either pixel domain or transform domain
					fat_zero2,            // float            fat_zero2,           // here - absolute
					corr_radius);        // int               corr_radius,        // radius of the output correlation (7 for 15x15)
	 }
}

/**
 * Normalize, low-pass filter, convert to pixel domain and unfold correlation tiles. This is an inner
 * kernel that is called from corr2D_normalize.
 *
 * @param num_corr_tiles   number of correlation tiles to process
 * @param corr_stride_td   stride (in floats) for correlation input (transform domain).
 * @param gpu_corrs_td     correlation data in transform domain
 * @param corr_weights     null or per-tile weight (fat_zero2 will be divided by it), length = num_corr_tiles
 * @param corr_stride      stride (in floats) for correlation pixel-domain outputs.
 * @param gpu_corrs        allocated array for the correlation output data (each element stride, payload: (2*corr_radius+1)^2
 * @param fat_zero2        add this value squared to the sum of squared components before normalization
 * @param corr_radius      radius of the output correlation (maximal 7 for 15x15)
 */

extern "C" __global__ void corr2D_normalize_inner(
		int               num_corr_tiles,     // number of correlation tiles to process
		const size_t      corr_stride_td,     // (in floats) stride for the input TD correlations
		float           * gpu_corrs_td,       // correlation tiles in transform domain
		float           * corr_weights,       // null or per-tile weight (fat_zero2 will be divided by it)
		const size_t      corr_stride,        // in floats
		float           * gpu_corrs,          // correlation output data (either pixel domain or transform domain
		float             fat_zero2,          // here - absolute, squared
		int               corr_radius)        // radius of the output correlation (7 for 15x15)
{
	corr_radius &= 0x7; // actual never >7. Still did not understand where is the problem,
	// providing literal "7" in the call does not fix the problem

	int corr_in_block = threadIdx.y;
	int corr_num = blockIdx.x * CORR_TILES_PER_BLOCK_NORMALIZE + corr_in_block; // 4
	if (corr_num >= num_corr_tiles){
		return; // nothing to do
	}
    __syncthreads();// __syncwarp();
    __shared__ float clt_corrs   [CORR_TILES_PER_BLOCK_NORMALIZE][4][DTT_SIZE][DTT_SIZE1];
    __shared__ float mlt_corrs   [CORR_TILES_PER_BLOCK_NORMALIZE][DTT_SIZE2M1][DTT_SIZE2M1]; // result correlation
    __shared__ float norm_fat_zero [CORR_TILES_PER_BLOCK_NORMALIZE];
    // set clt_corr to all zeros
    float * clt_corr =  ((float *) clt_corrs) +  corr_in_block * (4 * DTT_SIZE * DTT_SIZE1); // top left quadrant0
    float * mclt_corr = ((float *) mlt_corrs) +  corr_in_block * (DTT_SIZE2M1*DTT_SIZE2M1);

    // Read correlation tile from the device memory to the shared memory
	float *mem_corr = gpu_corrs_td + corr_stride_td * corr_num + threadIdx.x;
	float *clt = clt_corr + threadIdx.x;
#pragma unroll
	for (int i = 0; i < DTT_SIZE4; i++){
		(*clt)      = (*mem_corr);
		clt        += DTT_SIZE1;
		mem_corr   += DTT_SIZE;
	}
	__syncthreads();// __syncwarp();

	if (threadIdx.x == 0){
		norm_fat_zero[corr_in_block] = fat_zero2;
		if (corr_weights) { // same for all
			norm_fat_zero[corr_in_block] /= * (corr_weights + corr_num);
		}
	}
	__syncthreads();// __syncwarp();


	// normalize Amplitude
	normalizeTileAmplitude(
			clt_corr, // float * clt_tile, //       [4][DTT_SIZE][DTT_SIZE1], // +1 to alternate column ports
			norm_fat_zero[corr_in_block]); // fat_zero2); // float fat_zero2 ) // fat zero is absolute, scale it outside
	// Low Pass Filter from constant area (is it possible to replace?)

#ifdef DBG_TILE
#ifdef DEBUG6
	if ((tile_num == DBG_TILE) && (corr_pair == 0) && (threadIdx.x == 0)){
		printf("\ncorrelate2D CORRELATION NORMALIZED, fat_zero2=%f\n",fat_zero2);
		debug_print_clt1(clt_corr, -1,  0xf);
	}
	__syncthreads();// __syncwarp();
#endif
#endif

#ifdef DBG_TILE
#ifdef DEBUG6
	if ((tile_num == DBG_TILE) && (corr_pair == 0) && (threadIdx.x == 0)){
		printf("\ncorrelate2D LPF\n");
		debug_print_lpf(lpf_corr);
	}
	__syncthreads();// __syncwarp();
#endif
#endif

	// Apply LPF filter
	clt = clt_corr + threadIdx.x;
#pragma unroll
	for (int q = 0; q < 4; q++){
		float *lpf = lpf_corr + threadIdx.x;
#pragma unroll
		for (int i = 0; i < DTT_SIZE; i++){
			(*clt) *= (*lpf);
			clt   += DTT_SIZE1;
			lpf   += DTT_SIZE;
		}
	}
	__syncthreads();// __syncwarp();
#ifdef DBG_TILE
#ifdef DEBUG6
	if ((tile_num == DBG_TILE) && (corr_pair == 0) && (threadIdx.x == 0)){
		printf("\ncorrelate2D CORRELATION LPF-ed\n");
		debug_print_clt1(clt_corr, -1,  0xf);
	}
	__syncthreads();// __syncwarp();
#endif
#endif

	// Convert correlation to pixel domain with DTT-II
	dttii_2d(clt_corr);
#ifdef DBG_TILE
#ifdef DEBUG6
	if ((tile_num == DBG_TILE) && (corr_pair == 0) && (threadIdx.x == 4)){
		printf("\ncorrelate2D AFTER HOSIZONTAL (VERTICAL) PASS, corr_radius=%d\n",corr_radius);
		debug_print_clt1(clt_corr, -1,  0xf);
	}
	__syncthreads();// __syncwarp();

#endif
#endif

	// Unfold center area (2 * corr_radius + 1) * (2 * corr_radius + 1)
	corrUnfoldTile(
			corr_radius, // int corr_radius,
			(float *) clt_corr,   // float* qdata0, //    [4][DTT_SIZE][DTT_SIZE1], // 4 quadrants of the clt data, rows extended to optimize shared ports
			(float *) mclt_corr); // float* rslt)  //   [DTT_SIZE2M1][DTT_SIZE2M1]) // 15x15

	__syncthreads();

#ifdef DBG_TILE
#ifdef DEBUG6
	if ((tile_num == DBG_TILE) && (corr_pair == 0) && (threadIdx.x == 0)){
		printf("\ncorrelate2D after UNFOLD, corr_radius=%d\n",corr_radius);
		debug_print_corr_15x15(
				corr_radius, // int     corr_radius,
				mclt_corr,
				-1);
	}
	__syncthreads();// __syncwarp();
#endif
#endif

	// copy (2 * corr_radius +1) x (2 * corr_radius +1) (up to 15x15) tile to the main memory
	int size2r1 = 2 * corr_radius + 1;
	int len2r1x2r1 = size2r1 * size2r1;
	int corr_tile_offset =  + corr_stride * corr_num;
	mem_corr = gpu_corrs + corr_tile_offset;
#pragma unroll
	for (int offs = threadIdx.x; offs < len2r1x2r1; offs+=CORR_THREADS_PER_TILE){ // variable number of cycles per thread
		mem_corr[offs] = mclt_corr[offs];
	}
	__syncthreads();
#ifdef DBG_TILE
#ifdef DEBUG6
	if ((tile_num == DBG_TILE) && (corr_pair == 0) && (threadIdx.x == 0)){
		printf("\ncorrelate2D after copy to main memory\n");
		//    	debug_print_clt1(clt_corr, -1,  0xf);
	}
	__syncthreads();// __syncwarp();
#endif
#endif



}


/**
 * Calculate texture as RGBA (or YA for mono) from the in-memory frequency domain representation
 * and the per-tile task array (may be sparse).
 * Determines WoI from min/max Y,X of the selected tiles, returns calculated WoI in woi parameter
 * color is the outer index of the result, the image is moved to the top-left corner
 * (woi.x -> 0, woi.y -> 0, packed texture_rbga_stride per line, number of output lines per slice
 * is woi.height.
 *
 * This kernel launches others with CDP, from CPU it is just <<<1,1>>>
 *
 * @param num_cams             number of cameras
 * @param gpu_ftasks           flattened tasks, 27 floats for quad EO, 99 floats for LWIR16
 * @param num_tiles            number of tiles int gpu_tasks array prepared for processing
 * @param gpu_texture_indices  allocated array - 1 integer per tile to process
 * @param num_texture_tiles    allocated array - 8 integers (may be reduced to 4 later)
 * @param woi                  WoI for the output texture (x,y,width,height of the woi)
 * @param width                full image width in tiles <= TILES-X, use for faster processing of LWIR images (should be actual + 1)
 * @param height               full image height in tiles <= TILES-Y, use for faster processing of LWIR images
 * @param gpu_clt              array of num_cams pointers to the CLT (frequency domain) data [TILES-Y][TILES-X][NUM_COLORS][DTT_SIZE*DTT_SIZE]
 * @param gpu_geometry_correction geometry correction structure, used for rXY to determine pairs weight
 * @param colors               number of colors used:  3 for RGB or 1 for monochrome
 * @param is_lwir              do not perform shot correction
 * @param params               array of 5 float parameters (mitigating CUDA_ERROR_INVALID_PTX):
 *     	  min_shot             shot noise minimal value (10.0)
 *   	  scale_shot           scale shot noise (3.0)
 *   	  diff_sigma           pixel value/pixel change (1.5)
 *   	  diff_threshold       pixel value/pixel change (10)
 *   	  min_agree            minimal number of channels to agree on a point (real number to work with fuzzy averages) (3.0)
 * @param weights              scales for R,B,G {0.294118, 0.117647, 0.588235}
 * @param dust_remove          do not reduce average weight when only one image differs much from the average (true)
 * @param keep_weights         return channel weights after A in RGBA (was removed)
 * @param texture_rbga_stride  output stride (in floats)
 * @param gpu_texture_tiles    output array (number of colors +1 + ?) * woi.height * output stride(first woi.width valid) float values
 */
extern "C" __global__ void generate_RBGA(
		int                num_cams,           // number of cameras used
		// Parameters to generate texture tasks
		float            * gpu_ftasks,         // flattened tasks, 27 floats for quad EO, 99 floats for LWIR16
//		struct tp_task   * gpu_tasks,
		int                num_tiles,          // number of tiles in task list
		// declare arrays in device code?
		int              * gpu_texture_indices,// packed tile + bits (now only (1 << 7)
		int              * num_texture_tiles,  // number of texture tiles to process  (8 separate elements for accumulation)
		int              * woi,                // x,y,width,height of the woi
		int                width,  // <= TILES-X, use for faster processing of LWIR images (should be actual + 1)
		int                height, // <= TILES-Y, use for faster processing of LWIR images
		// Parameters for the texture generation
		float          ** gpu_clt,            // [num_cams] ->[TILES-Y][TILES-X][colors][DTT_SIZE*DTT_SIZE]
		// TODO: use geometry_correction rXY !
		struct gc       * gpu_geometry_correction,
		int               colors,             // number of colors (3/1)
		int               is_lwir,            // do not perform shot correction
		float             params[5],          // mitigating CUDA_ERROR_INVALID_PTX
		float             weights[3],         // scale for R,B,G
		int               dust_remove,        // Do not reduce average weight when only one image differs much from the average
		int               keep_weights,       // return channel weights after A in RGBA (was removed)
		const size_t      texture_rbga_stride,     // in floats
		float           * gpu_texture_tiles)  // (number of colors +1 + ?)*16*16 rgba texture tiles
{
	float             min_shot = params[0];           // 10.0
	float             scale_shot = params[1];         // 3.0
	float             diff_sigma = params[2];         // pixel value/pixel change
	float             diff_threshold = params[3];     // pixel value/pixel change
	float             min_agree = params[4];          // minimal number of channels to agree on a point (real number to work with fuzzy averages)
	int               tilesya =  ((height +3) & (~3)); //#define TILES-YA       ((TILES-Y +3) & (~3))
	dim3 threads0((1 << THREADS_DYNAMIC_BITS), 1, 1);
    int blocks_x = (width + ((1 << THREADS_DYNAMIC_BITS) - 1)) >> THREADS_DYNAMIC_BITS;
    dim3 blocks0 (blocks_x, height, 1);

	if (threadIdx.x == 0) {
		clear_texture_list<<<blocks0,threads0>>>(
				gpu_texture_indices,
				width,
				height);
		cudaDeviceSynchronize(); // not needed yet, just for testing
		dim3 threads((1 << THREADS_DYNAMIC_BITS), 1, 1);
		int blocks_t =   (num_tiles + ((1 << THREADS_DYNAMIC_BITS)) -1) >> THREADS_DYNAMIC_BITS;//
	    dim3 blocks(blocks_t, 1, 1);
	    // mark used tiles in gpu_texture_indices memory
		mark_texture_tiles <<<blocks,threads>>>(
				num_cams,           // int                num_cams,
				gpu_ftasks,         // float            * gpu_ftasks,         // flattened tasks, 27 floats for quad EO, 99 floats for LWIR16
				num_tiles,          // number of tiles in task list
				width,              // number of tiles in a row
				gpu_texture_indices); // packed tile + bits (now only (1 << 7)
		cudaDeviceSynchronize();
	    // mark n/e/s/w used tiles from gpu_texture_indices memory to gpu_tasks lower 4 bits
		*(woi + 0) = width;  // TILES-X;
		*(woi + 1) = height; // TILES-Y;
		*(woi + 2) = 0; // maximal x
		*(woi + 3) = 0; // maximal y
		mark_texture_neighbor_tiles <<<blocks,threads>>>(
				num_cams,           // int                num_cams,
				gpu_ftasks,         // float            * gpu_ftasks,         // flattened tasks, 27 floats for quad EO, 99 floats for LWIR16
				num_tiles,           // number of tiles in task list
				width,               // number of tiles in a row
				height,              // number of tiles rows
				gpu_texture_indices, // packed tile + bits (now only (1 << 7)
				woi);                // min_x, min_y, max_x, max_y

		cudaDeviceSynchronize();
	    // Generate tile indices list, upper 24 bits - tile index, lower 4 bits: n/e/s/w neighbors, bit 7 - set to 1
		*(num_texture_tiles+0) = 0;
		*(num_texture_tiles+1) = 0;
		*(num_texture_tiles+2) = 0;
		*(num_texture_tiles+3) = 0;
		*(num_texture_tiles+4) = 0;
		*(num_texture_tiles+5) = 0;
		*(num_texture_tiles+6) = 0;
		*(num_texture_tiles+7) = 0;

		gen_texture_list <<<blocks,threads>>>(
				num_cams,            // int                num_cams,
				gpu_ftasks,          // float            * gpu_ftasks,         // flattened tasks, 27 floats for quad EO, 99 floats for LWIR16
				num_tiles,           // number of tiles in task list
				width,               // number of tiles in a row
				height,              // int                height,               // number of tiles rows
				gpu_texture_indices, // packed tile + bits (now only (1 << 7)
				num_texture_tiles,   // number of texture tiles to process
				woi);                // x,y, here woi[2] = max_X, woi[3] - max-Y

		cudaDeviceSynchronize(); // not needed yet, just for testing
		*(woi + 2) += 1 - *(woi + 0); // width
		*(woi + 3) += 1 - *(woi + 1); // height
	}
	 __syncthreads();
// Zero output textures. Trim
// texture_rbga_stride
	 int texture_width =        (*(woi + 2) + 1)* DTT_SIZE;
	 int texture_tiles_height = (*(woi + 3) + 1) * DTT_SIZE;
	 int texture_slices =       colors + 1;
	 if (keep_weights & 2){
		 texture_slices += colors * num_cams;
	 }

	 if (threadIdx.x == 0) {

		    dim3 threads2((1 << THREADS_DYNAMIC_BITS), 1, 1);
		    int blocks_x = (texture_width + ((1 << (THREADS_DYNAMIC_BITS + DTT_SIZE_LOG2 )) - 1)) >> (THREADS_DYNAMIC_BITS + DTT_SIZE_LOG2);
		    dim3 blocks2 (blocks_x, texture_tiles_height * texture_slices, 1); // each thread - 8 vertical
		    clear_texture_rbga<<<blocks2,threads2>>>( // add clearing of multi-sensor output (keep_weights & 2 !=0)
		    		texture_width,
					texture_tiles_height * texture_slices, // int               texture_slice_height,
					texture_rbga_stride,                   // const size_t      texture_rbga_stride,     // in floats 8*stride
					gpu_texture_tiles) ;                   // float           * gpu_texture_tiles);
// Run 8 times - first 4 1-tile offsets  inner tiles (w/o verifying margins), then - 4 times with verification and ignoring 4-pixel
// oversize (border 16x116 tiles overhang by 4 pixels)
			cudaDeviceSynchronize(); // not needed yet, just for testing
			for (int pass = 0; pass < 8; pass++){
//			    dim3 threads_texture(TEXTURE_THREADS_PER_TILE, NUM_CAMS, 1); // TEXTURE_TILES_PER_BLOCK, 1);
//			    dim3 threads_texture(TEXTURE_THREADS_PER_TILE, num_cams, 1); // TEXTURE_TILES_PER_BLOCK, 1);
//			    dim3 threads_texture(TEXTURE_THREADS/num_cams, num_cams, 1); // TEXTURE_TILES_PER_BLOCK, 1);
				int num_cams_per_thread = NUM_THREADS / TEXTURE_THREADS_PER_TILE; // 4 cameras parallel, then repeat
				dim3 threads_texture(TEXTURE_THREADS_PER_TILE, num_cams_per_thread, 1); // TEXTURE_TILES_PER_BLOCK, 1);
		//		 dim3 threads_texture(TEXTURE_THREADS_PER_TILE, NUM_CAMS, 1); // TEXTURE_TILES_PER_BLOCK, 1);
		//	     dim3 threads_texture(TEXTURE_THREADS/num_cams, num_cams, 1); // TEXTURE_TILES_PER_BLOCK, 1);

			    int border_tile =  pass >> 2;
			    int ntt = *(num_texture_tiles + ((pass & 3) << 1) + border_tile);
			    dim3 grid_texture((ntt + TEXTURE_TILES_PER_BLOCK-1) / TEXTURE_TILES_PER_BLOCK,1,1); // TEXTURE_TILES_PER_BLOCK = 1
			    int ti_offset = (pass & 3) * (width * (tilesya >> 2)); //  (TILES-X * (TILES-YA >> 2));  // 1/4
			    if (border_tile){
			    	ti_offset += width * (tilesya >> 2) - ntt; // TILES-X * (TILES-YA >> 2) - ntt;
			    }
#ifdef DEBUG12
				printf("\ngenerate_RBGA() pass= %d, border_tile= %d, ti_offset= %d, ntt=%d\n",
						pass, border_tile,ti_offset, ntt);
				printf("\ngenerate_RBGA() gpu_texture_indices= %p, gpu_texture_indices + ti_offset= %p\n",
						(void *) gpu_texture_indices, (void *) (gpu_texture_indices + ti_offset));
				printf("\ngenerate_RBGA() grid_texture={%d, %d, %d)\n",
						grid_texture.x, grid_texture.y, grid_texture.z);
				printf("\ngenerate_RBGA() threads_texture={%d, %d, %d)\n",
						threads_texture.x, threads_texture.y, threads_texture.z);
				printf("\n");
#endif
			    /* */
				int shared_size = get_textures_shared_size( // in bytes
					    num_cams,     // int                num_cams,     // actual number of cameras
						colors,   // int                num_colors,   // actual number of colors: 3 for RGB, 1 for LWIR/mono
						0);           // int *              offsets);     // in floats
			    textures_accumulate <<<grid_texture,threads_texture, shared_size>>>(
			    		num_cams,                        // int               num_cams,           // number of cameras used
			    		woi,                             // int             * woi,                // x, y, width,height
						gpu_clt,                         // float          ** gpu_clt,            // [num_cams] ->[TILES-Y][TILES-X][colors][DTT_SIZE*DTT_SIZE]
						ntt,                             // size_t            num_texture_tiles,  // number of texture tiles to process
						ti_offset,                       //                   gpu_texture_indices_offset,// add to gpu_texture_indices
						gpu_texture_indices, //  + ti_offset, // int             * gpu_texture_indices,// packed tile + bits (now only (1 << 7)
						gpu_geometry_correction,         // struct gc       * gpu_geometry_correction,
						colors,                          // int               colors,             // number of colors (3/1)
						is_lwir,                         // int               is_lwir,            // do not perform shot correction
						min_shot,                        // float             min_shot,           // 10.0
						scale_shot,                      // float             scale_shot,         // 3.0
						diff_sigma,                      // float             diff_sigma,         // pixel value/pixel change
						diff_threshold,                  // float             diff_threshold,     // pixel value/pixel change
						min_agree,                       // float             min_agree,          // minimal number of channels to agree on a point (real number to work with fuzzy averages)
						weights,                         // float             weights[3],         // scale for R,B,G
						dust_remove,                     // int               dust_remove,        // Do not reduce average weight when only one image differs much from the average
						keep_weights,                    // int               keep_weights,       // return channel weights after A in RGBA (was removed) (should be 0 if gpu_texture_rbg)?
			    // combining both non-overlap and overlap (each calculated if pointer is not null )
						texture_rbga_stride,             // size_t      texture_rbg_stride, // in floats
						gpu_texture_tiles,               // float           * gpu_texture_rbg,     // (number of colors +1 + ?)*16*16 rgba texture tiles
			    		0,                               // size_t      texture_stride,     // in floats (now 256*4 = 1024)
						gpu_texture_tiles, //(float *)0);// float           * gpu_texture_tiles);  // (number of colors +1 + ?)*16*16 rgba texture tiles
						1, // int               linescan_order,     // if !=0 then output gpu_diff_rgb_combo in linescan order, else  - in gpu_texture_indices order
						(float *)0, //);//gpu_diff_rgb_combo);             // float           * gpu_diff_rgb_combo) // diff[num_cams], R[num_cams], B[num_cams],G[num_cams]
						width);
				cudaDeviceSynchronize(); // not needed yet, just for testing
				/* */
			}

	 }
	 __syncthreads();
}


/**
 * Helper kernel for generate_RBGA() - zeroes output array (next passes accumulate)
 * @param  texture_width            texture width in pixels, aligned to DTT_SIZE
 * @param  texture_slice_height     full number of output rows: texture height in pixels, multiplied by number of color slices
 * @param  texture_rbga_stride      texture line  stride in floats
 * @param  gpu_texture_tiles        pointer to the texture output
 */
// blockDim.x * gridDim.x >= width
__global__ void clear_texture_rbga(
		int               texture_width, // aligned to DTT_SIZE
		int               texture_slice_height,
		const size_t      texture_rbga_stride,     // in floats 8*stride
		float           * gpu_texture_tiles)  // (number of colors +1 + ?)*16*16 rgba texture tiles
{
	int col = (blockDim.x * blockIdx.x + threadIdx.x) << DTT_SIZE_LOG2;
	if (col > texture_width) {
		return;
	}
	int row = blockIdx.y;; // includes slices
	float * pix = gpu_texture_tiles + col + row * texture_rbga_stride;
#pragma unroll
	for (int n = 0; n < DTT_SIZE; n++) {
		*(pix++) = 0.0;
	}
}

/**
 * Helper kernel for generate_RBGA() -  prepare list of texture tiles, woi, and calculate orthogonal
 * neighbors for tiles (in 4 bits of the task field. Use 4x8=32 threads,
 *
 * @param num_cams             number of cameras
 * @param gpu_ftasks           flattened tasks, 27 floats for quad EO, 99 floats for LWIR16
 * @param num_tiles            number of tiles int gpu_tasks array prepared for processing
 * @param gpu_texture_indices  allocated array - 1 integer per tile to process
 * @param num_texture_tiles    number of texture tiles to process (allocated 8-element integer array)
 * @param woi                  4-element int array ( x,y,width,height of the woi, in tiles)
 * @param width                full image width in tiles <= TILES-X, use for faster processing of LWIR images (should be actual + 1)
 * @param height               full image height in tiles <= TILES-Y, use for faster processing of LWIR images
 */
__global__ void prepare_texture_list(
		int                num_cams,           // number of cameras used
		float            * gpu_ftasks,         // flattened tasks, 27 floats for quad EO, 99 floats for LWIR16
//		struct tp_task   * gpu_tasks,
		int                num_tiles,          // number of tiles in task list
		int              * gpu_texture_indices,// packed tile + bits (now only (1 << 7)
		// modified to have 8 length - split each subsequence into non-border/border tiles. Non-border will grow up,
		// border - down from the sam3\e 1/4 of the buffer
		int              * num_texture_tiles,  // number of texture tiles to process  (8 separate elements for accumulation)
		int              * woi,                // x,y,width,height of the woi
		int                width,  // <= TILES-X, use for faster processing of LWIR images (should be actual + 1)
		int                height) // <= TILES-Y, use for faster processing of LWIR images
{
//	int task_num = blockIdx.x;
//	int tid = threadIdx.x; // maybe it will be just <<<1,1>>>
    dim3 threads0((1 << THREADS_DYNAMIC_BITS), 1, 1);
//    int blocks_x = (width + 1) >> THREADS_DYNAMIC_BITS;
    int blocks_x = (width + ((1 << THREADS_DYNAMIC_BITS) - 1)) >> THREADS_DYNAMIC_BITS;

    dim3 blocks0 (blocks_x, height, 1);

	if (threadIdx.x == 0) {
		clear_texture_list<<<blocks0,threads0>>>(
				gpu_texture_indices,
				width,
				height);
		cudaDeviceSynchronize(); // not needed yet, just for testing
		dim3 threads((1 << THREADS_DYNAMIC_BITS), 1, 1);
		int blocks_t =   (num_tiles + ((1 << THREADS_DYNAMIC_BITS)) -1) >> THREADS_DYNAMIC_BITS;//
	    dim3 blocks(blocks_t, 1, 1);
	    // mark used tiles in gpu_texture_indices memory
		mark_texture_tiles <<<blocks,threads>>>(
				num_cams,
				gpu_ftasks,
//				gpu_tasks,
				num_tiles,          // number of tiles in task list
				width,
				gpu_texture_indices); // packed tile + bits (now only (1 << 7)
		cudaDeviceSynchronize();
	    // mark n/e/s/w used tiles from gpu_texture_indices memory to gpu_tasks lower 4 bits
		*(woi + 0) = width; // TILES-X;
		*(woi + 1) = height; // TILES-Y;
		*(woi + 2) = 0; // maximal x
		*(woi + 3) = 0; // maximal y
		mark_texture_neighbor_tiles <<<blocks,threads>>>(
				num_cams,
				gpu_ftasks,
//				gpu_tasks,
				num_tiles,           // number of tiles in task list
				width,               // number of tiles in a row
				height,              // number of tiles rows
				gpu_texture_indices, // packed tile + bits (now only (1 << 7)
				woi);                // min_x, min_y, max_x, max_y
		cudaDeviceSynchronize();
	    // Generate tile indices list, upper 24 bits - tile index, lower 4 bits: n/e/s/w neighbors, bit 7 - set to 1
		*(num_texture_tiles+0) = 0;
		*(num_texture_tiles+1) = 0;
		*(num_texture_tiles+2) = 0;
		*(num_texture_tiles+3) = 0;
		*(num_texture_tiles+4) = 0;
		*(num_texture_tiles+5) = 0;
		*(num_texture_tiles+6) = 0;
		*(num_texture_tiles+7) = 0;

		gen_texture_list <<<blocks,threads>>>(
				num_cams,
				gpu_ftasks,
//				gpu_tasks,
				num_tiles,           // number of tiles in task list
				width,               // number of tiles in a row
				height,              // int                height,               // number of tiles rows
				gpu_texture_indices, // packed tile + bits (now only (1 << 7)
				num_texture_tiles,   // number of texture tiles to process
				woi);                // x,y, here woi[2] = max_X, woi[3] - max-Y

		cudaDeviceSynchronize(); // not needed yet, just for testing
		*(woi + 2) += 1 - *(woi + 0); // width
		*(woi + 3) += 1 - *(woi + 1); // height
	}
	 __syncthreads();
}

/**
 * Helper kernel for prepare_texture_list() (for generate_RBGA) - clear texture list
 *
 * @param gpu_texture_indices  allocated array - 1 integer per tile to process
 * @param width                full image width in tiles <= TILES-X, use for faster processing of LWIR images (should be actual + 1)
 * @param height               full image height in tiles <= TILES-Y, use for faster processing of LWIR images
 */

// blockDim.x * gridDim.x >= width
__global__ void clear_texture_list(
		int              * gpu_texture_indices,// packed tile + bits (now only (1 << 7)
		int                width,  // <= TILES-X, use for faster processing of LWIR images
		int                height) // <= TILES-Y, use for faster processing of LWIR images
{
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int row = blockIdx.y;
	if (col > width) {
		return;
	}
	*(gpu_texture_indices + col + row * width) = 0; // TILES-X) = 0;
}
/**
 * Helper kernel for prepare_texture_list() (for generate_RBGA) - mark used tiles in
 * gpu_texture_indices memory
 *
 * @param num_cams             number of cameras <= NUM_CAMS
 * @param gpu_ftasks           flattened tasks, 27 floats for quad EO, 99 floats for LWIR16
 * @param num_tiles            number of tiles int gpu_tasks array prepared for processing
 * @param width                number of tiles in a row
 * @param gpu_texture_indices  allocated array - 1 integer per tile to process
 */

// treads (*,1,1), blocks = (*,1,1)
__global__ void mark_texture_tiles(
		int                num_cams,
		float            * gpu_ftasks,         // flattened tasks, 27 floats for quad EO, 99 floats for LWIR16
///		struct tp_task   * gpu_tasks,
		int                num_tiles,           // number of tiles in task list
		int                width,               // number of tiles in a row
		int              * gpu_texture_indices) // packed tile + bits (now only (1 << 7)
{
	int task_num = blockDim.x * blockIdx.x + threadIdx.x;
	if (task_num >= num_tiles) {
		return; // nothing to do
	}
///	int task = gpu_tasks[task_num].task;
	int task = get_task_task(task_num, gpu_ftasks, num_cams);

//	if (!(task & TASK_TEXTURE_BITS)){ // here any bit in TASK_TEXTURE_BITS is sufficient
	if (!(task & (1 << TASK_TEXT_EN))){ // here any bit in TASK_TEXTURE_BITS is sufficient
		if (!task) {// temporary disabling
			return; // NOP tile
		}
	}
///	int cxy = gpu_tasks[task_num].txy;
	int cxy = get_task_txy(task_num, gpu_ftasks, num_cams);

	*(gpu_texture_indices + (cxy & 0xffff) + (cxy >> 16) * width) = 1; // TILES-X) = 1;
}

/**
 * Helper kernel for prepare_texture_list() (for generate_RBGA) - calculate and save
 * bitmap of available neighbors in 4->8 directions (needed for alpha generation of
 * the result textures to fade along the border.
 *
 * @param num_cams             number of cameras
 * @param gpu_ftasks           flattened tasks, 29 floats for quad EO, 101 floats for LWIR16
 * @param num_tiles            number of tiles int gpu_tasks array prepared for processing
 * @param width                number of tiles in a row
 * @param height               number of tiles rows
 * @param gpu_texture_indices  allocated array - 1 integer per tile to process
 * @param woi                  4-element int array ( x,y,width,height of the woi, in tiles)
 */
// treads (*,1,1), blocks = (*,1,1)
__global__ void mark_texture_neighbor_tiles( // TODO: remove __global__?
		int                num_cams,
		float            * gpu_ftasks,          // flattened tasks, 27 floats for quad EO, 99 floats for LWIR16
		int                num_tiles,           // number of tiles in task list
		int                width,               // number of tiles in a row
		int                height,              // number of tiles rows
		int              * gpu_texture_indices, // packed tile + bits (now only (1 << 7)
		int              * woi)                  // x,y,width,height of the woi

{
	int task_num = blockDim.x * blockIdx.x + threadIdx.x;
	if (task_num >= num_tiles) {
		return; // nothing to do
	}

	int task = get_task_task(task_num, gpu_ftasks, num_cams);
//	if (!(task & TASK_TEXTURE_BITS)){ // here any bit in TASK_TEXTURE_BITS is sufficient
	if (!(task & (1 << TASK_TEXT_EN))){ // here any bit in TASK_TEXTURE_BITS is sufficient
		if (!task) {// temporary disabling
			return; // NOP tile
		}
	}
	int cxy = get_task_txy(task_num, gpu_ftasks, num_cams);

	int x = (cxy & 0xffff);
	int y = (cxy >> 16);
	atomicMin(woi+0, x);
	atomicMin(woi+1, y);
	atomicMax(woi+2, x);
	atomicMax(woi+3, y);
	int d = 0;
	if ((y > 0)                                 && *(gpu_texture_indices +  x +      (y - 1) * width)) d |= (1 << TASK_TEXT_N_BIT);
	if ((y > 0)            && (x < (width - 1)) && *(gpu_texture_indices + (x + 1) + (y - 1) * width)) d |= (1 << TASK_TEXT_NE_BIT);
	if (                      (x < (width - 1)) && *(gpu_texture_indices + (x + 1) +  y *      width)) d |= (1 << TASK_TEXT_E_BIT);
	if ((y < (height - 1)) && (x < (width - 1)) && *(gpu_texture_indices + (x + 1) + (y + 1) * width)) d |= (1 << TASK_TEXT_SE_BIT);
	if ((y < (height - 1))                      && *(gpu_texture_indices +  x +      (y + 1) * width)) d |= (1 << TASK_TEXT_S_BIT);
	if ((y < (height - 1)) && (x > 0)           && *(gpu_texture_indices + (x - 1) + (y + 1) * width)) d |= (1 << TASK_TEXT_SW_BIT);
	if (                      (x > 0)           && *(gpu_texture_indices + (x - 1) +  y *      width)) d |= (1 << TASK_TEXT_W_BIT);
	if ((y > 0)            && (x > 0)           && *(gpu_texture_indices + (x - 1) + (y - 1) * width)) d |= (1 << TASK_TEXT_NW_BIT);
	// Set task texture bits in global gpu_ftasks array (lower 4 bits)
///	gpu_tasks[task_num].task = ((task ^ d) & TASK_TEXTURE_BITS) ^ task;
	*(int *) (gpu_ftasks +  get_task_size(num_cams) * task_num) = ((task ^ d) & TASK_TEXTURE_BITS) ^ task; // updates task bits???
}

/**
 * Helper kernel for prepare_texture_list() (for generate_RBGA) - generate
 * list of tiles for texture calculation. As the tiles overlap, there are four lists
 * of non-overlapping tiles (odd/even rows/columns). At first made 8 lists, with pairs of
 * growing up and down for inner and border tiles, but now border attribute is not
 * used anymore.
 * @param num_cams             number of cameras
 * @param gpu_ftasks           flattened tasks, 27 floats for quad EO, 99 floats for LWIR16
// * @param gpu_tasks            array of per-tile tasks (struct tp_task)
 * @param num_tiles            number of tiles int gpu_tasks array prepared for processing
 * @param gpu_texture_indices  allocated array - 1 integer per tile to process
 * @param num_texture_tiles    number of texture tiles to process (allocated 8-element integer array)
 * @param woi                  4-element int array ( x,y,width,height of the woi, in tiles)
 */
__global__ void gen_texture_list(
		int                num_cams,
		float            * gpu_ftasks,           // flattened tasks, 27 floats for quad EO, 99 floats for LWIR16
		int                num_tiles,            // number of tiles in task list
		int                width,                // number of tiles in a row
		int                height,               // number of tiles rows
		int              * gpu_texture_indices,  // packed tile + bits (now only (1 << 7)
		int              * num_texture_tiles,    // number of texture tiles to process
		int              * woi)                  // min_x, min_y, max_x, max_y input

{
	int               tilesya =  ((height +3) & (~3)); //#define TILES-YA       ((TILES-Y +3) & (~3))
	int task_num = blockDim.x * blockIdx.x + threadIdx.x;
	if (task_num >= num_tiles) {
		return; // nothing to do
	}

///	int task = gpu_tasks[task_num].task & TASK_TEXTURE_BITS;
	int task = get_task_task(task_num, gpu_ftasks, num_cams);
	if (!task){ // here any bit in TASK_TEXTURE_BITS is sufficient
		return; // NOP tile - any non-zero bit is sufficient
	}
//	int cxy = gpu_tasks[task_num].txy;
	int cxy = get_task_txy(task_num, gpu_ftasks, num_cams);
	int x = (cxy & 0xffff);
	int y = (cxy >> 16);

#ifdef DEBUG12
		if ((x == DBG_TILE_X)  && (y == DBG_TILE_Y)){
			printf("\ngen_texture_list() x = %d, y= %d\n",x, y);
			printf("\ngen_texture_list() num_texture_tiles = %d(%d) %d(%d) %d(%d) %d(%d)\n",
					num_texture_tiles[0],num_texture_tiles[1],num_texture_tiles[2],num_texture_tiles[3],
					num_texture_tiles[4],num_texture_tiles[5],num_texture_tiles[6],num_texture_tiles[7]);
		}
		__syncthreads();// __syncwarp();
#endif // DEBUG12


	// don't care if calculate extra pixels that still fit into memory
	int is_border = (x == woi[0]) || (y == woi[1]) || (x == (width - 1)) || (y == woi[3]);
	int buff_head = 0;
	int num_offset = 0;
	if (x & 1) {
		buff_head += width * (tilesya >> 2); //TILES-YA - 2 LSB == 00
		num_offset += 2; // int *
	}
	if (y & 1) {
		buff_head += width * (tilesya >> 1);
		num_offset += 4; // int *
	}
	if (is_border){
		buff_head += (width * (tilesya >> 2) - 1); // end of the buffer
		num_offset += 1; // int *
	}
	gpu_texture_indices += buff_head;
	num_texture_tiles += num_offset;
	// using atomic operation in global memory - slow, but as operations here are per-tile, not per- pixel, it should be OK
	int buf_offset = atomicAdd(num_texture_tiles, 1);
	if (is_border){
		buf_offset = -buf_offset;
	}
#ifdef DEBUG12
	if ((x == DBG_TILE_X)  && (y == DBG_TILE_Y)){
		printf("\ngen_texture_list() buff_head=%d, buf_offset = %d, num_offset= %d, is_border=%d\n",
				buff_head, buf_offset, num_offset,is_border);
		printf("\ngen_texture_list() gpu_texture_indices = %p,  gpu_texture_indices + buf_offset = %p\n",
				(void *) gpu_texture_indices, (void *) (gpu_texture_indices + buf_offset));
	}
	__syncthreads();// __syncwarp();
#endif // DEBUG12
//	*(gpu_texture_indices + buf_offset) = task | ((x + y * width) << CORR_NTILE_SHIFT) | (1 << LIST_TEXTURE_BIT);
	// keep only 8 LSBs of task, use higher 24 for task number
//	*(gpu_texture_indices + buf_offset) = (task & ((1 << TEXT_NTILE_SHIFT) -1)) | ((x + y * width) << TEXT_NTILE_SHIFT) | (1 << LIST_TEXTURE_BIT);
	// keep only 4 lower task bits
	*(gpu_texture_indices + buf_offset) = (task & TASK_TEXTURE_BITS) | ((x + y * width) << TEXT_NTILE_SHIFT) | (1 << LIST_TEXTURE_BIT);
	//CORR_NTILE_SHIFT // TASK_TEXTURE_BITS
}
//inline __device__ int get_task_size(int num_cams){
//	return sizeof(struct tp_task)/sizeof(float) - 6 * (NUM_CAMS - num_cams);
//}

inline __device__ int get_task_task(int num_tile, float * gpu_ftasks, int num_cams) {
	return *(int *) (gpu_ftasks +  get_task_size(num_cams) * num_tile);
}
inline __device__ int get_task_txy(int num_tile, float * gpu_ftasks, int num_cams) {
	return *(int *) (gpu_ftasks +  get_task_size(num_cams) * num_tile + 1);
}
/**
 * Helper kernel for convert_direct() - generates dense list of tiles for direct MCLT.
 * Tile order from the original (sparse) list is not preserved
 *
 * @param task_size,           flattened task size in 4-byte floats
 * @param gpu_ftasks           flattened tasks, 27 floats for quad EO, 99 floats for LWIR16
// * @param gpu_tasks            array of per-tile tasks (struct tp_task)
 * @param num_tiles            number of tiles int gpu_tasks array prepared for processing
 * @param active_tiles         integer array to place the generated list
 * @param pnum_active_tiles    single-element integer array return generated list length
 */
__global__ void index_direct(
		int                task_size,        // flattened task size in 4-byte floats
		float            * gpu_ftasks,         // flattened tasks, 27 floats for quad EO, 99 floats for LWIR16
//		struct tp_task   * gpu_tasks,
		int                num_tiles,          // number of tiles in task
		int *              active_tiles,       // pointer to the calculated number of non-zero tiles
		int *              pnum_active_tiles)  //  indices to gpu_tasks  // should be initialized to zero
{
	int num_tile = blockIdx.x * blockDim.x + threadIdx.x;
	if (num_tile >= num_tiles){
		return;
	}
//	if (gpu_tasks[num_tile].task != 0) {
	if (gpu_ftasks[num_tile * task_size] != 0) {
		active_tiles[atomicAdd(pnum_active_tiles, 1)] = num_tile;
	}
}

/**
 * Helper kernel for textures_nonoverlap() - generates dense list of tiles for non-overlap
 * (i.e. colors x 16 x 16 per each tile in the list ) texture tile generation
 *
 * @param num_cams         number of cameras <= NUM_CAMS
 * @param gpu_ftasks           flattened tasks, 29 floats for quad EO, 101 floats for LWIR16
 * @param num_tiles            number of tiles int gpu_tasks array prepared for processing
 * @param width                number of tiles in a row
 * @param nonoverlap_list      integer array to place the generated list
 * @param pnonoverlap_length   single-element integer array return generated list length
 */
extern "C" __global__ void create_nonoverlap_list(
		int                num_cams,
		float            * gpu_ftasks,         // flattened tasks, 27 floats for quad EO, 99 floats for LWIR16
//		struct tp_task   * gpu_tasks,
		int                num_tiles,           // number of tiles in task
		int                width,               // number of tiles in a row
		int *              nonoverlap_list,     // pointer to the calculated number of non-zero tiles
		int *              pnonoverlap_length)  //  indices to gpu_tasks  // should be initialized to zero
{
	int num_tile = blockIdx.x * blockDim.x + threadIdx.x;
	if (num_tile >= num_tiles){
		return;
	}
	int task_task = get_task_task(num_tile, gpu_ftasks, num_cams);
///	if ((gpu_tasks[num_tile].task & TASK_TEXTURE_BITS) == 0){
//	if ((task_task & TASK_TEXTURE_BITS) == 0){
	if (!(task_task & (1 << TASK_TEXT_EN))){ // here any bit in TASK_TEXTURE_BITS is sufficient
		if (!task_task) {// temporary disabling
			return; // NOP tile
		}
	}
///	int cxy = gpu_tasks[num_tile].txy;
	int cxy = get_task_txy(num_tile, gpu_ftasks, num_cams);
	 // all texture direction bits as it is non-overlapped list (bits probably unused)
	int texture_task_code = (((cxy & 0xffff) + (cxy >> 16) * width) << TEXT_NTILE_SHIFT) | (1 << LIST_TEXTURE_BIT) | TASK_TEXTURE_BITS;
//	if (gpu_tasks[num_tile].task != 0) {
	if (task_task != 0) {
		nonoverlap_list[atomicAdd(pnonoverlap_length, 1)] = texture_task_code;
	}
}

/**
 * Helper kernel for correlate2D() - generates dense list of correlation tasks.
 * With the quad camera each tile may generate up to 6 pairs (int array elements)
 * Tiles are not ordered, but the correlation pairs for each tile are
 *
 * @param num_cams           number of cameras <= NUM_CAMS
 * @param sel_pairs          array of length to accommodate all pairs (4  for 16 cameras, 120 pairs).
 * @param gpu_ftasks         flattened tasks, 27 floats for quad EO, 99 floats for LWIR16
 * @param num_tiles          number of tiles int gpu_tasks array prepared for processing
 * @param gpu_corr_indices   integer array to place the generated list
 * @param pnum_corr_tiles    single-element integer array return generated list length
 */
__global__ void index_correlate(
		int               num_cams,
//		int *             sel_pairs,           // unused bits should be 0
		int               sel_pairs0,
		int               sel_pairs1,
		int               sel_pairs2,
		int               sel_pairs3,
		float            * gpu_ftasks,         // flattened tasks, 27 floats for quad EO, 99 floats for LWIR16
//		struct tp_task   * gpu_tasks,
		int                num_tiles,         // number of tiles in task
		int                width,                // number of tiles in a row
		int *              gpu_corr_indices,  // array of correlation tasks
		int *              pnum_corr_tiles)   // pointer to the length of correlation tasks array
{
	int num_tile = blockIdx.x * blockDim.x + threadIdx.x;
	if (num_tile >= num_tiles){
		return;
	}
    int sel_pairs[] = {sel_pairs0, sel_pairs1, sel_pairs2, sel_pairs3};
	//	int task_size = get_task_size(num_cams);
	int task_task =get_task_task(num_tile, gpu_ftasks, num_cams);
	if ((task_task & ((1 << TASK_CORR_EN) | (1 << TASK_INTER_EN))) == 0){ // needs correlation. Maybe just check task_task != 0? TASK_CORR_EN
		if (!task_task) { // temporary disabling
			return;
		}
	}
	int pair_list_start = pairs_offsets[num_cams];
	int pair_list_len =   pairs_offsets[num_cams+1] - pair_list_start;
	int num_mask_words = (pair_list_len + 31) >> 5; // ceil
	int nb = 0;
	for (int i = 0; i < num_mask_words; i++){
		if (sel_pairs[i]) {
			nb += __popc (sel_pairs[i]); // number of non-zero bits
		}
	}
	if (nb > 0){
		int indx = atomicAdd(pnum_corr_tiles, nb);
		int task_txy = get_task_txy(num_tile, gpu_ftasks, num_cams);
		int tx = task_txy & 0xffff;
		int ty = task_txy >> 16;
		int nt = ty * width + tx;
		//		for (int b = 0; b < pair_list_len; b++) if ((cm & (1 << b)) != 0) {
		for (int b = 0; b < pair_list_len; b++) if ((sel_pairs[ b >> 5] & (1 << (b & 31))) != 0) {
			gpu_corr_indices[indx++] = (nt << CORR_NTILE_SHIFT) | b;
		}
	}
}

/**
 * Helper kernel for correlateInter2D() - generates dense list of correlation tasks.
 * For interscene correlation. One correlation output for each selected sensor
 * plus a sum of them all. So for all 16 sensors selected ooutput will have 17
 * 2D correlations (with some being the l;ast one)
 * All pairs for the same tile will always be in the same order: increasing sensor numbers
 * with sum being the last. Sum will be marked by 0xff in the LSB.
 * With the quad camera each tile may generate up to 6 pairs (int array elements)
 * Tiles are not ordered, but the correlation pairs for each tile are.
 *
 * @param num_cams           number of cameras <= NUM_CAMS <32
 * @param sel_sensors        array of length to accommodate all pairs (4  for 16 cameras, 120 pairs).
 * @param gpu_ftasks         flattened tasks, 27 floats for quad EO, 99 floats for LWIR16
 * @param num_tiles          number of tiles int gpu_tasks array prepared for processing
 * @param gpu_corr_indices   integer array to place the generated list
 * @param pnum_corr_tiles    single-element integer array return generated list length
 */

__global__ void index_inter_correlate(
		int               num_cams,
		int               sel_sensors,
		float           * gpu_ftasks,         // flattened tasks, 27 floats for quad EO, 99 floats for LWIR16
		int               num_tiles,         // number of tiles in task
		int               width,                // number of tiles in a row
		int *             gpu_corr_indices,  // array of correlation tasks
		int *             pnum_corr_tiles)   // pointer to the length of correlation tasks array
{
	int num_tile = blockIdx.x * blockDim.x + threadIdx.x;
	if (num_tile >= num_tiles){
		return;
	}
	//	int task_size = get_task_size(num_cams);
	int task_task =get_task_task(num_tile, gpu_ftasks, num_cams);
	if (((task_task >> TASK_INTER_EN) & 1) == 0){ // needs correlation. Maybe just check task_task != 0?
		if (!task_task){ // temporary disabling
			return;
		}
	}
	int nb = __popc (sel_sensors); // number of non-zero bits
	if (nb > 0){
		int indx = atomicAdd(pnum_corr_tiles, nb+1);
		int task_txy = get_task_txy(num_tile, gpu_ftasks, num_cams);
		int tx = task_txy & 0xffff;
		int ty = task_txy >> 16;
		int nt = ty * width + tx;
		//		for (int b = 0; b < pair_list_len; b++) if ((cm & (1 << b)) != 0) {
		for (int b = 0; b < num_cams; b++) if ((sel_sensors & (1 << (b & 31))) != 0) {
			gpu_corr_indices[indx++] = (nt << CORR_NTILE_SHIFT) | b;
		}
		gpu_corr_indices[indx++] = (nt << CORR_NTILE_SHIFT) | 0xff; // will be used for sum
	}
}


/**
 * Direct MCLT transform and aberration correction with space-variant deconvolution
 * kernels. Results are used to output aberration-corrected images, textures and
 * 2D phase correlations.
 * This kernel is called from the CPU with <<<1,1>>>
 * @param num_cams             number of subcameras <= NUM_CAMS. 4 for RGB, 16 for lwir in LWIR16
 * @param num_colors           number of colors <= NUM_COLORS. 3 for RGB, 1 for lwir/mono
 * @param gpu_kernel_offsets   array of per-camera pointers to array of struct CltExtra (one element per kernel)
 * @param gpu_kernels          array of per-camera pointers to array of kernels (clt representation)
 * @param gpu_images           array of per-camera pointers to Bayer images
 * @param gpu_ftasks           flattened tasks, 27 floats per tile for quad EO, 99 floats -- for LWIR16
 * @param gpu_clt              output array of per-camera aberration-corrected transform-domain image representations
 *                             [num_cams][TILES-Y][TILES-X][num_colors][DTT_SIZE*DTT_SIZE]
 * @param dstride              stride (in floats) for the input Bayer images
 * @param num_tiles            number of tiles int gpu_tasks array prepared for processing
 * @param lpf_mask             apply lpf to colors : bit 0 - red, bit 1 - blue, bit2 - green. Now - always 0 !
 * @param woi_width            image width (was constant IMG-WIDTH, now variable to use with EO+LWIR
 * @param woi_height           image height (was constant IMG-HEIGHT, now variable to use with EO+LWIR
 * @param kernels_hor          number of deconvolution kernels per image width
 * @param kernels_vert         number of deconvolution kernels per image height
 * @param gpu_active_tiles     pointer to the calculated list of tiles
 * @param pnum_active_tiles    pointer to the number of active tiles
 */
extern "C" __global__ void convert_direct(  // called with a single block, single thread
		int                num_cams,           // actual number of cameras
		int                num_colors,         // actual number of colors: 3 for RGB, 1 for LWIR/mono
		float           ** gpu_kernel_offsets, // [num_cams],
		float           ** gpu_kernels,        // [num_cams],
		float           ** gpu_images,         // [num_cams],
		float            * gpu_ftasks,         // flattened tasks, 27 floats for quad EO, 99 floats for LWIR16
		float           ** gpu_clt,            // [num_cams][TILES-Y][TILES-X][num_colors][DTT_SIZE*DTT_SIZE]
		size_t             dstride,            // in floats (pixels)
		int                num_tiles,          // number of tiles in task
		int                lpf_mask,           // apply lpf to colors : bit 0 - red, bit 1 - blue, bit2 - green. Now - always 0 !
		int                woi_width,
		int                woi_height,
		int                kernels_hor,
		int                kernels_vert,
		int *              gpu_active_tiles,   // pointer to the calculated list of tiles
		int *              pnum_active_tiles,  // pointer to the number of active tiles
		int                tilesx)
{
	 dim3 threads0(CONVERT_DIRECT_INDEXING_THREADS, 1, 1);
	 dim3 blocks0 ((num_tiles + CONVERT_DIRECT_INDEXING_THREADS -1) >> CONVERT_DIRECT_INDEXING_THREADS_LOG2,1, 1);
	 if (threadIdx.x == 0) { // always 1
		 *pnum_active_tiles = 0;
		 int task_size = get_task_size(num_cams);
		 index_direct<<<blocks0,threads0>>>(
				 task_size,          // int                task_size,        // flattened task size in 4-byte floats
				 gpu_ftasks,         // float            * gpu_ftasks,         // flattened tasks, 27 floats for quad EO, 99 floats for LWIR16
				 num_tiles,           //int                num_tiles,          // number of tiles in task
				 gpu_active_tiles,    //int *              active_tiles,       // pointer to the calculated number of non-zero tiles
				 pnum_active_tiles);  //int *              pnum_active_tiles)  //  indices to gpu_tasks  // should be initialized to zero
		 cudaDeviceSynchronize();
		 // now call actual convert_correct_tiles
		 dim3 threads_tp(THREADSX, TILES_PER_BLOCK, 1);
		 dim3 grid_tp((*pnum_active_tiles + TILES_PER_BLOCK -1 )/TILES_PER_BLOCK, 1);
		 convert_correct_tiles<<<grid_tp,threads_tp>>>(
				 num_cams,           // int                num_cams,           // actual number of cameras
				 num_colors,         // int                num_colors,         // actual number of colors: 3 for RGB, 1 for LWIR/mono
				 gpu_kernel_offsets, // float           ** gpu_kernel_offsets, // [num_cams],
				 gpu_kernels,        // float           ** gpu_kernels,        // [num_cams],
				 gpu_images,         // float           ** gpu_images,         // [num_cams],
				 gpu_ftasks,         // float            * gpu_ftasks,         // flattened tasks, 27 floats for quad EO, 99 floats for LWIR16
				 gpu_active_tiles,   // int              * gpu_active_tiles,   // indices in gpu_tasks to non-zero tiles
				 *pnum_active_tiles, // int                num_active_tiles,   // number of tiles in task
				 gpu_clt,            // float           ** gpu_clt,            // [num_cams][TILES-Y][TILES-X][num_colors][DTT_SIZE*DTT_SIZE]
				 dstride,            // size_t             dstride,            // in floats (pixels)
				 lpf_mask,           // int                lpf_mask,           // apply lpf to colors : bit 0 - red, bit 1 - blue, bit2 - green. Now - always 0 !
				 woi_width,          // int                woi_width,          // varaible to swict between EO and LWIR
				 woi_height,         // int                woi_height,         // varaible to swict between EO and LWIR
				 kernels_hor,        // int                kernels_hor,        // varaible to swict between EO and LWIR
				 kernels_vert, // );      // int                kernels_vert);      // varaible to swict between EO and LWIR
				 tilesx); // int                tilesx)

	 }
}

/**
 * Erase CLT tiles before generating corrected images when not all tiles are converted. IMCLT for full images
 * processes all CLT tiles, so if some tiles are skipped, they preserve all TD data that appears in the output.
 * No erase is needed before correlations or texture generation.
 *
 * @param num_cams           number of subcameras <= NUM_CAMS. 4 for RGB, 16 for lwir in LWIR16
 * @param num_colors         number of colors <= NUM_COLORS. 3 for RGB, 1 for lwir/mono
 * @param tiles_x            number of tiles in a row
 * @param tiles_y            number of tile rows
 * @param gpu_clt            array of per-camera aberration-corrected transform-domain image representations
 * @param fill_data          data to write (normally 0.0f, may be NaN?)
 */
extern "C" __global__ void erase_clt_tiles(
		    int                num_cams,           // actual number of cameras
		    int                num_colors,         // actual number of colors: 3 for RGB, 1 for LWIR/mono
			int                tiles_x,
			int                tiles_y,
			float           ** gpu_clt,            // [num_cams][tiles_y][tiles_x][num_colors][4*DTT_SIZE*DTT_SIZE]
			float              fill_data)
{
	if (threadIdx.x == 0) { // anyway 1,1,1
		dim3 threads_erase(NUM_THREADS, 1, 1); // (32,1,1)
		dim3 grid_erase   (tiles_x, tiles_y, num_cams);
		erase_clt_tiles_inner<<<grid_erase,threads_erase>>>(
				num_colors,             // int                num_colors,
				tiles_x,                // int                tiles_x,
				gpu_clt,                // float           ** gpu_clt,
				fill_data);             // float              fill_data)
	}
}

extern "C" __global__ void erase_clt_tiles_inner(
		int                num_colors,         // actual number of colors: 3 for RGB, 1 for LWIR/mono
		int                tiles_x,
		float           ** gpu_clt,            // [num_cams][tiles_y][tiles_x][num_colors][4*DTT_SIZE*DTT_SIZE]
		float              fill_data)
{
	int tile_size = num_colors * (4*DTT_SIZE*DTT_SIZE);
	// can not use gridDim -> cuda.CudaException: CUDA_ERROR_INVALID_PTX
//	float * data = gpu_clt[blockIdx.z] + tile_size * (blockIdx.x + blockIdx.y * gridDim.x) + threadIdx.x;
	float * data = gpu_clt[blockIdx.z] + tile_size * (blockIdx.x + blockIdx.y * tiles_x) + threadIdx.x;
	for (int ncol = 0; ncol < num_colors; ncol++){
#pragma unroll
		for (int i = 0; i < (4*DTT_SIZE*DTT_SIZE/NUM_THREADS); i++){
			*data = fill_data;
			data += NUM_THREADS;
		}
	}
}



/**
 * Helper kernel for convert_direct() - perform actual conversion.
 *
 * @param num_cams             number of subcameras <= NUM_CAMS. 4 for RGB, 16 for lwir in LWIR16
 * @param num_colors           number of colors <= NUM_COLORS. 3 for RGB, 1 for lwir/mono
 * @param gpu_kernel_offsets   array of per-camera pointers to array of struct CltExtra (one element per kernel)
 * @param gpu_kernels          array of per-camera pointers to array of kernels (clt representation)
 * @param gpu_images           array of per-camera pointers to Bayer images
 * @param gpu_ftasks           flattened tasks, 27 floats per tile for quad EO, 99 floats -- for LWIR16
 * @param gpu_active_tiles     pointer to the calculated list of tiles
 * @param num_active_tiles     number of active tiles
 * @param gpu_clt              output array of per-camera aberration-corrected transform-domain image representations
 * @param dstride              stride (in floats) for the input Bayer images
 * @param lpf_mask             apply lpf to colors : bit 0 - red, bit 1 - blue, bit2 - green. Now - always 0 !
 * @param woi_width            image width (was constant IMG-WIDTH, now variable to use with EO+LWIR
 * @param woi_height           image height (was constant IMG-HEIGHT, now variable to use with EO+LWIR
 * @param kernels_hor          number of deconvolution kernels per image width
 * @param kernels_vert         number of deconvolution kernels per image height
 */
__global__ void convert_correct_tiles(
		    int                num_cams,           // actual number of cameras
		    int                num_colors,         // actual number of colors: 3 for RGB, 1 for LWIR/mono
			float           ** gpu_kernel_offsets, // [num_cams],
			float           ** gpu_kernels,        // [num_cams],
			float           ** gpu_images,         // [num_cams],
			float            * gpu_ftasks,         // flattened tasks, 27 floats for quad EO, 99 floats for LWIR16
			int              * gpu_active_tiles,   // indices in gpu_tasks to non-zero tiles
			int                num_active_tiles,   // number of tiles in task
			float           ** gpu_clt,            // [num_cams][TILES-Y][TILES-X][num_colors][DTT_SIZE*DTT_SIZE]
			size_t             dstride,            // in floats (pixels)
			int                lpf_mask,           // apply lpf to colors : bit 0 - red, bit 1 - blue, bit2 - green. Now - always 0 !
			int                woi_width,
			int                woi_height,
			int                kernels_hor,
			int                kernels_vert,
			int                tilesx)
{
//    int tilesx = TILES-X;
///	dim3 t = threadIdx;
	int tile_in_block = threadIdx.y;
	int task_indx = blockIdx.x * TILES_PER_BLOCK + tile_in_block;
	if (task_indx >=  num_active_tiles){
		return; // nothing to do
	}
	int task_num = gpu_active_tiles[task_indx];
	int task_size = get_task_size(num_cams);
	float * tp0 = gpu_ftasks + task_size * task_num;
	if (*(int *) tp0 == 0)     return; // NOP tile
	__shared__ struct tp_task tt [TILES_PER_BLOCK];
	// Copy task data to shared memory
	int thread0 =  threadIdx.x & 1; // 0,1
	int thread12 = threadIdx.x >>1; // now 0..3 (total number ==  (DTT_SIZE), will not change

	float * tp = tp0 + TP_TASK_XY_OFFSET + threadIdx.x;
	if (thread12 < num_cams) {
		tt[tile_in_block].xy[thread12][thread0] = *(tp);        // gpu_task -> xy[thread12][thread0];
	}
	if (num_cams > 4){ // was unlikely, now 16
		for (int nc0 = 4; nc0 < num_cams; nc0 += 4){
			int nc = nc0 + thread12;
			if (nc < num_cams) {
				tt[tile_in_block].xy[nc][thread0] = *(tp += 8); // gpu_task -> xy[nc][thread0];
			}
		}
	}


	if (threadIdx.x == 0){ // only one thread calculates, others - wait
		tt[tile_in_block].task = *(int *)     (tp0++);    // get first integer value
		tt[tile_in_block].txy =  *(int *)     (tp0++);    // get second integer value
		tt[tile_in_block].target_disparity = *(tp0);      //
		tp0 +=3; // skip centerXY and previous increment (was tt[tile_in_block].target_disparity = *(tp0++);
		tt[tile_in_block].scale =            *(tp0++);    // get scale to multiply before accumulating/saving
	}
	// float centerXY[2] is not used/copied here

     __syncthreads();// __syncwarp();
    __shared__ float clt_tile        [TILES_PER_BLOCK][4][DTT_SIZE][DTT_SIZE1];
    __shared__ float clt_kernels     [TILES_PER_BLOCK][4][DTT_SIZE][DTT_SIZE1]; // +1 to alternate column ports
    __shared__ int   int_topleft     [TILES_PER_BLOCK][2];
    __shared__ float residual_shift  [TILES_PER_BLOCK][2];
    __shared__ float window_hor_cos  [TILES_PER_BLOCK][2*DTT_SIZE];
    __shared__ float window_hor_sin  [TILES_PER_BLOCK][2*DTT_SIZE];
    __shared__ float window_vert_cos [TILES_PER_BLOCK][2*DTT_SIZE];
    __shared__ float window_vert_sin [TILES_PER_BLOCK][2*DTT_SIZE];

    // process each camera,l each color in series (to reduce shared memory)
    for (int ncam = 0; ncam <  num_cams; ncam++){
    	for (int color = 0; color <  num_colors; color++){
    		convertCorrectTile(
    				// TODO: remove debug when done
#ifdef DBG_TILE
    				num_colors  + (((task_num == DBG_TILE)&& (ncam == 0)) ? 16:0),                      // int                   num_colors, //*
#else
    				num_colors,                      // int                   num_colors, //*
#endif
    				(struct CltExtra*)(gpu_kernel_offsets[ncam]),        // struct CltExtra* gpu_kernel_offsets,
					gpu_kernels[ncam],               // float           * gpu_kernels,
					gpu_images[ncam],                // float           * gpu_images,
					gpu_clt[ncam],                   // float           * gpu_clt,
					color,                           // const int         color,
					lpf_mask,                        // const int         lpf_mask,
					tt[tile_in_block].xy[ncam][0],   // const float       centerX,
					tt[tile_in_block].xy[ncam][1],   // const float       centerY,
					tt[tile_in_block].txy,           // const int txy,
					tt[tile_in_block].scale,         // const float           tscale,
					dstride,                         // size_t            dstride, // in floats (pixels)
					(float * )(clt_tile [tile_in_block]),        // float clt_tile [TILES_PER_BLOCK][NUM_CAMS][num_colors][4][DTT_SIZE][DTT_SIZE])
					(float * )(clt_kernels[tile_in_block]),      // float clt_tile    [num_colors][4][DTT_SIZE][DTT_SIZE],
					int_topleft[tile_in_block],      // int   int_topleft  [num_colors][2],
					residual_shift[tile_in_block],   // float frac_topleft [num_colors][2],
					window_hor_cos[tile_in_block],   // float window_hor_cos  [num_colors][2*DTT_SIZE],
					window_hor_sin[tile_in_block],   // float window_hor_sin  [num_colors][2*DTT_SIZE],
					window_vert_cos[tile_in_block],  // float window_vert_cos [num_colors][2*DTT_SIZE]);
					window_vert_sin[tile_in_block],  // float window_vert_sin [num_colors][2*DTT_SIZE]);
					woi_width,                       // int                woi_width,
					woi_height,                      // int                woi_height,
					kernels_hor,                     // int                kernels_hor,
					kernels_vert, //int                kernels_vert)
					tilesx); // int                tilesx);
    		 __syncthreads();
    	}
    }
}

/**
 * Calculate texture tiles without combining in overlapping areas (16x16 for each 8x8 of the image)
 * from the in-memory frequency domain representation and the per-tile task array (may be sparse).
 * Determines WoI from min/max Y,X of the selected tiles, returns calculated WoI in woi parameter
 * color is the outer index of the result, the image is moved to the top-left corner
 * (woi.x -> 0, woi.y -> 0, packed texture_rbga_stride per line, number of output lines per slice
 * is woi.height.
 *
 * This kernel launches others with CDP, from CPU it is just <<<1,1>>>
 *
 * @param num_cams             number of cameras
 * @param gpu_ftasks           flattened tasks, 27 floats for quad EO, 99 floats for LWIR16
 * @param num_tiles            number of tiles int gpu_tasks array prepared for processing
 * @param gpu_texture_indices  allocated array - 1 integer per tile to process
 * @param num_texture_tiles    allocated array - 8 integers (may be reduced to 4 later)
 * @param gpu_clt              array of num_cams pointers to the CLT (frequency domain) data [TILES-Y][TILES-X][colors][DTT_SIZE*DTT_SIZE]
 * @param gpu_geometry_correction geometry correction structure, used for rXY to determine pairs weight
 * @param colors               number of colors used:  3 for RGB or 1 for monochrome
 * @param is_lwir              do not perform shot correction
 * @param params               array of 5 float parameters (mitigating CUDA_ERROR_INVALID_PTX):
 *     	  min_shot             shot noise minimal value (10.0)
 *   	  scale_shot           scale shot noise (3.0)
 *   	  diff_sigma           pixel value/pixel change (1.5)
 *   	  diff_threshold       pixel value/pixel change (10)
 *   	  min_agree            minimal number of channels to agree on a point (real number to work with fuzzy averages) (3.0)
 * @param weights              scales for R,B,G {0.294118, 0.117647, 0.588235}
 * @param dust_remove          do not reduce average weight when only one image differs much from the average (true)
 * @param keep_weights         Was not here before 10/12/2022. return channel weights after A in RGBA (was removed) (should be 0 if gpu_texture_rbg)?
 * @param texture_stride       output stride in floats (now 256*4 = 1024)
 * @param gpu_texture_tiles    output array (number of colors +1 + ?)*16*16 rgba texture tiles) float values. Will not be calculated if null
 * @param inescan_order        0 low-res tiles have the same order, as gpu_texture_indices, 1 - in linescan order
 * @param gpu_diff_rgb_combo   low-resolution output, with per-camera mismatch an each color average. Will not be calculated if null
 * @param num_tilesx           number of tiles in a row
 */
extern "C" __global__ void textures_nonoverlap(
		int               num_cams,           // number of cameras
		float            * gpu_ftasks,         // flattened tasks, 27 floats for quad EO, 99 floats
		int               num_tiles,          // number of tiles in task list
// declare arrays in device code?
		int             * gpu_texture_indices,// packed tile + bits (now only (1 << 7)
		int             * pnum_texture_tiles,  // returns total number of elements in gpu_texture_indices array
		float          ** gpu_clt,            // [num_cams] ->[TILES-Y][TILES-X][colors][DTT_SIZE*DTT_SIZE]
		// TODO: use geometry_correction rXY !
		struct gc       * gpu_geometry_correction,
		int               colors,             // number of colors (3/1)
		int               is_lwir,            // do not perform shot correction
		float             params[5],
		float             weights[3],         // scale for R,B,G
		int               dust_remove,        // Do not reduce average weight when only one image differs much from the average
		int               keep_weights,       // Was not here before 10/12/2022. return channel weights after A in RGBA (was removed) (should be 0 if gpu_texture_rbg)?
// combining both non-overlap and overlap (each calculated if pointer is not null )
		size_t            texture_stride,     // in floats (now 256*4 = 1024)
		float           * gpu_texture_tiles,  // (number of colors +1 + ?)*16*16 rgba texture tiles
		int               linescan_order,     // 0 low-res tiles have tghe same order, as gpu_texture_indices, 1 - in linescan order
		float           * gpu_diff_rgb_combo, // diff[num_cams], R[num_cams], B[num_cams],G[num_cams]
		int               num_tilesx)
// num_tilesx in the end - worked, after num_tiles - did not compile with JIT in Eclipse
{
//	int num_tilesx =  TILES-X;
	float             min_shot = params[0];           // 10.0
	float             scale_shot = params[1];         // 3.0
	float             diff_sigma = params[2];         // pixel value/pixel change
	float             diff_threshold = params[3];     // pixel value/pixel change
	float             min_agree = params[4];          // minimal number of channels to agree on a point (real number to work with fuzzy averages)

	 dim3 threads0(CONVERT_DIRECT_INDEXING_THREADS, 1, 1);
	 dim3 blocks0 ((num_tiles + CONVERT_DIRECT_INDEXING_THREADS -1) >> CONVERT_DIRECT_INDEXING_THREADS_LOG2,1, 1);

	 if (threadIdx.x == 0) { // only 1 thread, 1 block
		 *pnum_texture_tiles = 0;
		 create_nonoverlap_list<<<blocks0,threads0>>>(
				 num_cams,            // int                num_cams,
				 gpu_ftasks,          // float            * gpu_ftasks,         // flattened tasks, 27 floats for quad EO, 99 floats for LWIR16
//				 gpu_tasks,           // struct tp_task   * gpu_tasks,
				 num_tiles,           // int                num_tiles,           // number of tiles in task
				 num_tilesx,          // int                width,               // number of tiles in a row
				 gpu_texture_indices, // int *              nonoverlap_list,     // pointer to the calculated number of non-zero tiles
				 pnum_texture_tiles); // int *              pnonoverlap_length)  //  indices to gpu_tasks  // should be initialized to zero
		 cudaDeviceSynchronize();
		 int num_cams_per_thread = NUM_THREADS / TEXTURE_THREADS_PER_TILE; // 4 cameras parallel, then repeat
//		 dim3 threads_texture(TEXTURE_THREADS_PER_TILE, NUM_CAMS, 1); // TEXTURE_TILES_PER_BLOCK, 1);
		 dim3 threads_texture(TEXTURE_THREADS_PER_TILE, num_cams_per_thread, 1); // TEXTURE_TILES_PER_BLOCK, 1);
//	     dim3 threads_texture(TEXTURE_THREADS/num_cams, num_cams, 1); // TEXTURE_TILES_PER_BLOCK, 1);

		 dim3 grid_texture((*pnum_texture_tiles + TEXTURE_TILES_PER_BLOCK-1) / TEXTURE_TILES_PER_BLOCK,1,1);
		 int shared_size = get_textures_shared_size( // in bytes
				 num_cams,     // int                num_cams,     // actual number of cameras
				 colors,   // int                num_colors,   // actual number of colors: 3 for RGB, 1 for LWIR/mono
				 0);           // int *              offsets);     // in floats
#ifdef DEBUG7A
		printf("\n1. shared_size=%d, num_cams=%d, colors=%d\n",shared_size,num_cams, colors);

		__syncthreads();

#endif
		textures_accumulate <<<grid_texture,threads_texture,  shared_size>>>( // 65536>>>( //
				num_cams,                        // 	int               num_cams,           // number of cameras used
				(int *) 0,                       // int             * woi,                // x, y, width,height
				gpu_clt,                         // float          ** gpu_clt,            // [num_cams] ->[TILES-Y][TILES-X][colors][DTT_SIZE*DTT_SIZE]
				*pnum_texture_tiles,             // size_t            num_texture_tiles,  // number of texture tiles to process
				0,                               //                gpu_texture_indices_offset,// add to gpu_texture_indices
				gpu_texture_indices,             // int             * gpu_texture_indices,// packed tile + bits (now only (1 << 7)
				gpu_geometry_correction,         // struct gc       * gpu_geometry_correction,
				colors,                          // int               colors,             // number of colors (3/1)
				is_lwir,                         // int               is_lwir,            // do not perform shot correction
				min_shot,                        // float             min_shot,           // 10.0
				scale_shot,                      // float             scale_shot,         // 3.0
				diff_sigma,                      // float             diff_sigma,         // pixel value/pixel change
				diff_threshold,                  // float             diff_threshold,     // pixel value/pixel change
				min_agree,                       // float             min_agree,          // minimal number of channels to agree on a point (real number to work with fuzzy averages)
				weights,                         // float             weights[3],         // scale for R,B,G
				dust_remove,                     // int               dust_remove,        // Do not reduce average weight when only one image differs much from the average
				keep_weights,                    // int               keep_weights,       // return channel weights after A in RGBA (was removed) (should be 0 if gpu_texture_rbg)?
				// combining both non-overlap and overlap (each calculated if pointer is not null )
				0,                               // size_t      texture_rbg_stride, // in floats
				(float *) 0,                     // float           * gpu_texture_rbg,     // (number of colors +1 + ?)*16*16 rgba texture tiles
				texture_stride,                  // size_t      texture_stride,     // in floats (now 256*4 = 1024)
				gpu_texture_tiles,               // (float *)0);// float           * gpu_texture_tiles);  // (number of colors +1 + ?)*16*16 rgba texture tiles
				linescan_order,                  //	int               linescan_order,     // if !=0 then output gpu_diff_rgb_combo in linescan order, else  - in gpu_texture_indices order
				gpu_diff_rgb_combo, //);             // float           * gpu_diff_rgb_combo) // diff[num_cams], R[num_cams], B[num_cams],G[num_cams]
				num_tilesx);
	 }
}


/**
 * Helper for generate_RBGA() and textures_nonoverlap()
 *
 * Calculate texture as RGBA (or YA for mono) from the in-memory frequency domain representation
 * and from the int array of texture indices.
 * Output overlapped (if gpu_texture_rbg != 0 and texture_rbg_stride !=0),
 *        non-overlapped (if gpu_texture_tiles != 0 and texture_stride !=0),
 *        and low-resolution (1/8) gpu_diff_rgb_combo (if gpu_diff_rgb_combo !=0)
 * @param num_cams             Number of cameras used
 * @param woi                  WoI for the output texture (x,y,width,height of the woi), may be null if overlapped output is not used
 * @param gpu_clt              array of num_cams pointers to the CLT (frequency domain) data [TILES-Y][TILES-X][NUM_COLORS][DTT_SIZE*DTT_SIZE]
 * @param num_texture_tiles    number of texture tiles to process
 * @param gpu_texture_indices_offset add to gpu_texture_indices
 * @param gpu_texture_indices  array - 1 integer per tile to process
 * @param gpu_geometry_correction geometry correction structure, used for rXY to determine pairs weight
 * @param colors               number of colors used:  3 for RGB or 1 for monochrome
 * @param is_lwir              do not perform shot correction
 * @param min_shot             shot noise minimal value (10.0)
 * @param scale_shot           scale shot noise (3.0)
 * @param diff_sigma           pixel value/pixel change (1.5)
 * @param diff_threshold       pixel value/pixel change (10)
 * @param min_agree            minimal number of channels to agree on a point (real number to work with fuzzy averages) (3.0)
 * @param weights              scales for R,B,G {0.294118, 0.117647, 0.588235}
 * @param dust_remove          do not reduce average weight when only one image differs much from the average (true)
 * @param keep_weights         return channel weights after A in RGBA (was removed). Now (11/12/2022): +1 - old meaning, +2 - replace port_weights with channel imclt
 * @param texture_rbg_stride   output stride for overlapped texture in floats, or 0 to skip
 * @param gpu_texture_rbg      output array (number of colors +1 + ?) * woi.height * output stride(first woi.width valid) float values (or 0)
 * @param texture_stride       output stride for non-overlapping texture tile output in floats (or 0 to skip)
 * @param gpu_texture_tiles    output of the non-overlapping tiles (or 0 to skip)
 * @param linescan_order       if !=0 then output gpu_diff_rgb_combo in linescan order, else  - in gpu_texture_indices order
 * @param gpu_diff_rgb_combo   low-resolution output, with per-camera mismatch an each color average. Will not be calculated if null
 * @param tilesx               number of tiles in a row. If negative then output gpu_diff_rgb_combo in linescan order,
 *                             if positive - in gpu_texture_indices order
 */
extern "C" __global__ void textures_accumulate( // (8,4,1) (N,1,1)
		int               num_cams,           // number of cameras used
		int             * woi,                // x, y, width,height
		float          ** gpu_clt,            // [num_cams] ->[TILES-Y][TILES-X][colors][DTT_SIZE*DTT_SIZE]
		size_t            num_texture_tiles,  // number of texture tiles to process
		int               gpu_texture_indices_offset,// add to gpu_texture_indices
		int             * gpu_texture_indices,// packed tile + bits (now only (1 << 7)
		struct gc       * gpu_geometry_correction,
		int               colors,             // number of colors (3/1)
		int               is_lwir,            // do not perform shot correction
		float             min_shot,           // 10.0
		float             scale_shot,         // 3.0
		float             diff_sigma,         // pixel value/pixel change
		float             diff_threshold,     // pixel value/pixel change
		float             min_agree,          // minimal number of channels to agree on a point (real number to work with fuzzy averages)
		float             weights[3],         // scale for R,B,G
		int               dust_remove,        // Do not reduce average weight when only one image differs much from the average
		int               keep_weights,       // return channel weights after A in RGBA (was removed) (should be 0 if gpu_texture_rbg)? Now +2 - output raw channels
// combining both non-overlap and overlap (each calculated if pointer is not null )
		size_t            texture_rbg_stride, // in floats
		float           * gpu_texture_rbg,    // (number of colors +1 + ?)*16*16 rgba texture tiles
		size_t            texture_stride,     // in floats (now 256*4 = 1024)
		float           * gpu_texture_tiles,  // (number of colors +1 + ?)*16*16 rgba texture tiles
		int               linescan_order,     // if !=0 then output gpu_diff_rgb_combo in linescan order, else  - in gpu_texture_indices order
		float           * gpu_diff_rgb_combo, //) // diff[num_cams], R[num_cams], B[num_cams],G[num_cams]
		int               tilesx)
{
	// will process exactly 4 cameras at a time in one block,
	// so imclt is executed sequentially for each group of 4 cameras
///	if ((threadIdx.x == 0) && (threadIdx.y == 0)){
///		printf("DONE\n");
///	}
///	__syncthreads();
///	return;

	int offsets [9];
	int shared_size = get_textures_shared_size( // in bytes
			 num_cams,     // int                num_cams,     // actual number of cameras
			 colors,       // int                num_colors,   // actual number of colors: 3 for RGB, 1 for LWIR/mono
			 offsets);     // int *              offsets);     // in floats

//	int camera_num = threadIdx.y;
	int tile_indx = blockIdx.x; //  * TEXTURE_TILES_PER_BLOCK + tile_in_block;
	if (tile_indx >= num_texture_tiles){
		return; // nothing to do
	}
	// get number of tile
	int tile_code = gpu_texture_indices[tile_indx + gpu_texture_indices_offset]; // Added for Java, no DP
	if ((tile_code & (1 << LIST_TEXTURE_BIT)) == 0){
		return; // nothing to do
	}
	int tile_num = tile_code >> TEXT_NTILE_SHIFT;
#ifdef DEBUG7A
	__syncthreads();// __syncwarp();
	if ((tile_num == DBG_TILE)  && (threadIdx.x == 0) && (threadIdx.y == 0)){
		printf("textures_accumulate: diff_sigma =     %f\n",    diff_sigma);
		printf("textures_accumulate: diff_threshold = %f\n",diff_threshold);
		printf("textures_accumulate: min_agree =      %f\n",     min_agree);
		printf("textures_accumulate: weights[0] =     %f\n",weights[0]);
		printf("textures_accumulate: weights[1] =     %f\n",weights[1]);
		printf("textures_accumulate: weights[2] =     %f\n",weights[2]);
		printf("textures_accumulate: dust_remove =    %d\n",dust_remove);
		printf("textures_accumulate: keep_weights =   %d\n",keep_weights);
	}
#endif //DEBUG7A

#ifdef DEBUG7A // 22
	if ((tile_num == DBG_TILE) && (threadIdx.x == 0) && (threadIdx.y == 0)){
		for (int i = 0; i <9; i++){
			printf(" offsets[%d] = 0x%x\n",i,offsets[i]);
		}
	}
	__syncthreads();
#endif	// #ifdef DEBUG22
#ifdef DEBUG7AXX // 22
		if ((tile_num == DBG_TILE)) { //  && (threadIdx.x == 0) && (threadIdx.y == 0)){
			printf("\n1. tile_indx=%d, tile_num=%d threadIdx.x = %d threadIdx.y =%d\n",tile_indx,tile_num,threadIdx.x,threadIdx.y);
		}
		__syncthreads();
#endif	// #ifdef DEBUG22
	extern __shared__ float all_shared[];
	float * mclt_tiles =       &all_shared[offsets[0]] ; // [num_cams][colors][2*DTT_SIZE][DTT_SIZE21];  // 16*1*16*17=0x1100 | 4*3*16*17=0xcc0
	float * clt_tiles  =       &all_shared[offsets[1]] ; // [num_cams][colors][4][DTT_SIZE][DTT_SIZE1]; // 16 * 1 * 4 * 8 * 9  = 0x1200 | 4 * 3 * 4 * 8 * 9 = 0xd80
	float * mclt_debayer =     &all_shared[offsets[1]] ; // [num_cams][colors][MCLT_UNION_LEN]; //  16 * 1 * 16 * 18  = 0x1200 | 4 * 3 * 16 * 18 = 0xd80 | to align with clt_tiles
	float * mclt_tmps =        &all_shared[offsets[2]] ; // [num_cams][colors][DTT_SIZE2][DTT_SIZE21]; // 16*1*16*17=0x1100 | 4*3*16*17=0xcc0. Used only with Bayer, not with mono
	float * rgbaw =            &all_shared[offsets[2]] ; // [colors + 1 + num_cams + colors + 1][DTT_SIZE2][DTT_SIZE21];

	float * port_offsets =     &all_shared[offsets[3]] ; // [num_cams][2];          // 16 * 2 = 0x20 | 4*2 = 0x8
	float * ports_rgb_shared = &all_shared[offsets[4]] ; // [colors][num_cams]; // 16 * 1 = 0x10 | 4 * 3 = 0xc | return to system memory (optionally pass null to skip calculation)
	float * max_diff_shared =  &all_shared[offsets[5]] ; // [num_cams];             // 16 = 0x10     | 4 = 0x4     | return to system memory (optionally pass null to skip calculation)
	float * max_diff_tmp =     &all_shared[offsets[6]] ; // [num_cams][TEXTURE_THREADS_PER_TILE]; // 16 * 8 = 0x80 | 4 * 8 = 0x20 | [4][8]
	float * ports_rgb_tmp =    &all_shared[offsets[7]] ; // [colors][num_cams][TEXTURE_THREADS_PER_TILE]; // 16 * 1 * 8 = 0x80 | 4 * 3 * 8  = 0x60 |  [4*3][8]
	float * texture_averaging =  max_diff_tmp;           // [NUM_THREADS] reusing, needs 32 elements for texture averaging, shared

#ifdef DBG_TILE
#ifdef DEBUG7AXX
	if (tile_num == DBG_TILE){ // }  && (threadIdx.x == 0) && (threadIdx.y == 0)){
		printf("\ntextures_gen tile = %d\n",tile_num);
		//    	debug_print_clt1(clt_tile1, color,  0xf); //
		//        printf("\textures_gen tile = %d, pair=%d, color = %d   CAMERA22\n",tile_num, corr_pair,color);
		//    	debug_print_clt1(clt_tile2, color,  0xf); //
	}
	__syncthreads();// __syncwarp();
#endif
#endif
	// serially for each color, parallel for each camera
	// copy clt (frequency domain data)
	// have to do sequentially 4 cameras at a time to fit into 32 parallel threads
	for (int camera_num_offs = 0; camera_num_offs < num_cams; camera_num_offs+= blockDim.y) {// assuming num_cams is multiple blockDim.y
		int camera_num = threadIdx.y + camera_num_offs;
		if (threadIdx.x < 2){ // not more than 16 sensors, not less than
			port_offsets[camera_num * 2 + threadIdx.x] =   gpu_geometry_correction->rXY[camera_num][threadIdx.x];
		}
		__syncthreads();// __syncwarp(); // is it needed?

		for (int color = 0; color < colors; color++){
			// clt_tiles is union with mclt_debayer, so has to have same step
			float * clt_tile =  clt_tiles + (camera_num * colors + color) * MCLT_UNION_LEN;
			float * clt_tilei = clt_tile + threadIdx.x; // threadIdx.x = 0..7 here
			float * gpu_tile = ((float *) gpu_clt[camera_num]) +  (tile_num * colors + color) * (4 * DTT_SIZE * DTT_SIZE) + threadIdx.x;
			float * mclt_tile = mclt_tiles +    (camera_num * colors + color) * 2 * DTT_SIZE * DTT_SIZE21;
			float * mclt_dst =  mclt_debayer +  (camera_num * colors + color) * MCLT_UNION_LEN; // 16 * 18
			float * mclt_tmp =  mclt_tmps +     (camera_num * colors + color) * DTT_SIZE2 * DTT_SIZE21; // 16*17
			// no camera_num below
#pragma unroll
			for (int q = 0; q < 4; q++) {
				float *lpf = lpf_data[(colors > 1)? color : 3] + threadIdx.x; // lpf_data[3] - mono
#pragma unroll
				for (int i = 0; i < DTT_SIZE; i++){ // copy 32 rows (4 quadrants of 8 rows)
					//				*clt_tilei = *gpu_tile * (*lpf) * scale;
					*clt_tilei = *gpu_tile * (*lpf);
					clt_tilei +=  DTT_SIZE1;
					gpu_tile +=   DTT_SIZE;
					lpf +=        DTT_SIZE;
				}
			}
			__syncthreads();
#ifdef DEBUG7AXXX
			if ((tile_num == DBG_TILE)  && (threadIdx.x == 0) && (threadIdx.y == 0)){
				printf("\ntextures_gen LPF for color = %d\n",color);
				debug_print_lpf(lpf_data[(colors > 1)? color : 3]);

				printf("\ntextures_gen tile = %d, color = %d \n",tile_num, color);
				debug_print_clt_scaled(clt_tile, color,  0xf, 0.25); //
			}
			__syncthreads();// __syncwarp();
#endif

#ifdef DBG_TILEXXX		// perform idct
			imclt8threads(
					0,          // int     do_acc,     // 1 - add to previous value, 0 - overwrite
					clt_tile,   //        [4][DTT_SIZE][DTT_SIZE1], // +1 to alternate column ports [4][8][9]
					mclt_tile,  // float * mclt_tile )
					((tile_num == DBG_TILE)  && (threadIdx.x == 0)));
#else
			imclt8threads(
					0,          // int     do_acc,     // 1 - add to previous value, 0 - overwrite
					clt_tile,   //        [4][DTT_SIZE][DTT_SIZE1], // +1 to alternate column ports [4][8][9]
					mclt_tile,  // float * mclt_tile )
					0);
#endif
			__syncthreads();// __syncwarp();
#ifdef DEBUG7A
			if ((tile_num == DBG_TILE)  && (threadIdx.x == 0) && (threadIdx.y == 0)){
				for (int ncam = camera_num_offs; ncam < (camera_num_offs + 4); ncam++){
					printf("\n3104 textures_gen mclt camera = % d,  color = %d\n",ncam, color);
					debug_print_mclt(
							mclt_tiles + (ncam * colors + color) * 2 * DTT_SIZE * DTT_SIZE21, //         [4][DTT_SIZE][DTT_SIZE1], // +1 to alternate column ports)
							color);
				}
			}
			__syncthreads();// __syncwarp();
#endif
			if (colors > 1) {
#ifdef DBG_TILE_XXX
				debayer_shot(
						(color < 2), // const int rb_mode,    // 0 - green, 1 - r/b
						min_shot,    // float     min_shot,   // 10.0
						scale_shot,  // float     scale_shot, // 3.0 (0.0 for mono)
						mclt_tile,   // float   * mclt_src,   // [2* DTT_SIZE][DTT_SIZE1+ DTT_SIZE], // +1 to alternate column ports[16][17]
						mclt_dst,    // float   * mclt_dst,   // [2* DTT_SIZE][DTT_SIZE1+ DTT_SIZE], // +1 to alternate column ports[16][17]
						mclt_tmp,    // float   * mclt_tmp,
						((tile_num == DBG_TILE)  && (threadIdx.x == 0))); // int debug);
#else
				debayer_shot(
						(color < 2), // const int rb_mode,    // 0 - green, 1 - r/b
						min_shot,    // float     min_shot,   // 10.0
						scale_shot,  // float     scale_shot, // 3.0 (0.0 for mono)
						mclt_tile,   // float   * mclt_src,   // [2* DTT_SIZE][DTT_SIZE1+ DTT_SIZE], // +1 to alternate column ports[16][17]
						mclt_dst,    // float   * mclt_dst,   // [2* DTT_SIZE][DTT_SIZE1+ DTT_SIZE], // +1 to alternate column ports[16][17]
						mclt_tmp,    // float   * mclt_tmp,
						0); // int debug);
#endif
				__syncthreads();// __syncwarp();
			} else {
				// copy? - no, just remember to use mclt_tile, not mclt_dst
				// will have to copy mclt_tiles -> mclt_dst as they have different gaps
				// untested copy for mono mode

#ifdef DEBUG7AXXX
				if (tile_num == DBG_TILE) {
//					for (int n = 0; n <= DTT_SIZE; n += DTT_SIZE){
					int n = 0;
						printf("textures_gen mclt_tile camera_num_offs= %d threadIdx.y= %d, threadIdx.x= %d, n=%d, msp=0x%x, dst=0x%x\n",
								camera_num_offs,threadIdx.y, threadIdx.x, n,
								(int) (mclt_tile + threadIdx.x + n), (int)(mclt_dst +  threadIdx.x + n));
//					}

				}
				__syncthreads();// __syncwarp();
#endif

#ifdef DEBUG7AXX // Good here
				if (tile_num == DBG_TILE) {
					for (int ccam = 0; ccam < num_cams; ccam++) {
						if ((threadIdx.x == 0) && (camera_num == ccam)){
							printf("\n3155 textures_gen mclt_tile camera_num_offs= %d threadIdx.y= %d, color = %d\n",camera_num_offs,threadIdx.y, color);
							debug_print_mclt( // broken for camera 1
									mclt_tile, //         [4][DTT_SIZE][DTT_SIZE1], // +1 to alternate column ports)
									-1);
						}
						__syncthreads();// __syncwarp();
					}
					printf("3162 camera_num_offs= %d threadIdx.y= %d, color = %d mclt_tile=0x%x, mclt_dst=0x%x\n",
							camera_num_offs,threadIdx.y, color, (int) mclt_tile, (int) mclt_dst);
				}
				__syncthreads();// __syncwarp();
#endif

//#ifdef DEBUGXXXX // no copy at all

				//#pragma unroll
				for (int n = 0; n <= DTT_SIZE; n += DTT_SIZE){
					float * msp = mclt_tile + threadIdx.x + n;
					float * dst = mclt_dst +  threadIdx.x + n;
					//#pragma unroll
					for (int row = 0; row < DTT_SIZE2; row++){
						*dst = *msp;
						msp += DTT_SIZE21;
						dst += DTT_SIZE21;
					}
				}
//#endif
				__syncthreads();
			} //if (colors > 1)  else

#ifdef DEBUG7AXX // still good here
			if (tile_num == DBG_TILE) {
				for (int ccam = 0; ccam < num_cams; ccam++) {
					if ((threadIdx.x == 0) && ((camera_num & 0x3) == (ccam & 0x3))){
						printf("\n 3185 mclt_tile : textures_gen mclt_tile camera_num_offs= %d camera number= %d threadIdx.y= %d, color = %d\n", camera_num_offs, ccam,threadIdx.y, color);
						debug_print_mclt( // broken for camera 1
//								mclt_tile, //         [4][DTT_SIZE][DTT_SIZE1], // +1 to alternate column ports)
								mclt_tiles +  (ccam * colors + color) * 2 * DTT_SIZE * DTT_SIZE21,
								-1);

						printf("\n 3190 mclt_dst: textures_gen AFTER DEBAER camera_num_offs= %d  camera number= %d threadIdx.y= %d, color = %d\n", camera_num_offs, ccam, threadIdx.y, color);
						debug_print_mclt(
//								mclt_dst, //         [4][DTT_SIZE][DTT_SIZE1], // +1 to alternate column ports)
								mclt_debayer +(ccam * colors + color) * MCLT_UNION_LEN, // 16 * 18
								-1);
						/*
					printf("\ntextures_gen AFTER DEBAER0 cam= %d, color = %d\n",threadIdx.y, 0);
					debug_print_mclt(
							mclt_debayer + (ccam * colors * MCLT_UNION_LEN), //         [4][DTT_SIZE][DTT_SIZE1], // +1 to alternate column ports)
							-1);
						 */
					}
					__syncthreads();// __syncwarp();
				}
			}
			__syncthreads();// __syncwarp();
#endif
		} // for (int color = 0; color < colors; color++)

		__syncthreads(); // __syncwarp();
		///	return;

		//	__shared__ float mclt_tiles [num_cams][colors][2*DTT_SIZE][DTT_SIZE21];
	} // end of sequential camera group: for (int camera_num_offs = 0; camera_num_offs < num_cams; camera_num_offs+= blockDim.y)

#ifdef DEBUG7A
	//#ifdef DEBUG22
	for (int ccam = 0; ccam < num_cams; ccam++) {
		if ((tile_num == DBG_TILE)  && (threadIdx.x == 0) && (threadIdx.y == 0)){
			for (int nncol = 0; nncol < colors; nncol++){
				printf("\n3227: mclt_tiles +  (ccam * colors + nncol) * 2 * DTT_SIZE * DTT_SIZE21 cam= %d, color = %d\n",ccam, nncol);
				//				float * mclt_dst =  (float *) shr.mclt_debayer[camera_num][color];
				debug_print_mclt(
						mclt_tiles +  (ccam * colors + nncol) * 2 * DTT_SIZE * DTT_SIZE21,
						-1);
			}
		}
		__syncthreads();// __syncwarp();
	}
	__syncthreads();// __syncwarp();
#endif

#ifdef DEBUG7A
	//#ifdef DEBUG22
	for (int ccam = 0; ccam < num_cams; ccam++) {
		if ((tile_num == DBG_TILE)  && (threadIdx.x == 0) && (threadIdx.y == 0)){
			for (int nncol = 0; nncol < colors; nncol++){
				printf("\n 3244 mclt_dst: textures_gen AFTER DEBAER camera number= %d threadIdx.y= %d, color = %d\n", ccam, threadIdx.y, nncol);
				debug_print_mclt(
						mclt_debayer +(ccam * colors + nncol) * MCLT_UNION_LEN, // 16 * 18
						-1);
			}
		}
		__syncthreads();// __syncwarp();
	}
	__syncthreads();// __syncwarp();
#endif



#ifdef DBG_TILE
		int debug = (tile_num == DBG_TILE);
#else
		int debug = 0;
#endif

	int calc_extra = (gpu_diff_rgb_combo != 0);
	tile_combine_rgba(
			num_cams,                  // int num_cams,
			colors,                    // int     colors,        // number of colors
			mclt_debayer,              //  (float*) shr.mclt_debayer, // float * mclt_tile,     // debayer // has gaps to align with union !
			mclt_tiles,       // float * rbg_tile,      // if not null - original (not-debayered) rbg tile to use for the output
			rgbaw, // (float *) shr1.rgbaw,      // float * rgba,
			// if calc_extra, rbg_tile will be ignored and output generated with blurred (debayered) data. Done so as debayered data is needed
			// to calculate max_diff_shared
			calc_extra, //  | (keep_weights & 2),                // int     calc_extra,    // 1 - calcualate ports_rgb, max_diff
			ports_rgb_shared,// float ports_rgb_shared [colors][num_cams], // return to system memory (optionally pass null to skip calculation)
			max_diff_shared, // float max_diff_shared  [num_cams], // return to system memory (optionally pass null to skip calculation)
			max_diff_tmp,    //  float max_diff_tmp     [num_cams][TEXTURE_THREADS_PER_TILE],
			ports_rgb_tmp,   // float ports_rgb_tmp    [colors][num_cams][TEXTURE_THREADS_PER_TILE], // [4*3][8]
			port_offsets,    // float * port_offsets,  // [port]{x_off, y_off} - just to scale pixel value differences
			diff_sigma,                // float   diff_sigma,     // pixel value/pixel change
			diff_threshold,            // float   diff_threshold, // pixel value/pixel change
			min_agree,                 // float   min_agree,   NOT USED?   // minimal number of channels to agree on a point (real number to work with fuzzy averages)
			weights,                   // float * chn_weights,    // color channel weights, sum == 1.0
			dust_remove,               // int     dust_remove,    // Do not reduce average weight when only one image differs much from the average
			(keep_weights & 1), //  | (keep_weights & 2),        // int     keep_weights,   // return channel weights and rms after A in RGBA (weight are always calculated)
			debug );  // int     debug );

	__syncthreads(); // _syncthreads();1

// return either only 4 slices (RBGA) or all 12 (with weights and rms) if keep_weights
// float rgbaw              [colors + 1 + num_cams + colors + 1][DTT_SIZE2][DTT_SIZE21];
//	size_t texture_tile_offset = + tile_indx * texture_stride;
// multiply all 4(2) slices by a window (if not all directions)
	if (gpu_texture_tiles && (texture_stride != 0)){ // generate non-ovelapping tiles
		float * gpu_texture_tile = gpu_texture_tiles + tile_indx * texture_stride;

		for (int pass = 0; pass < 8; pass ++) {
			int row = pass * 2 + (threadIdx.y >> 1);
			int col = ((threadIdx.y & 1) << 3) + threadIdx.x;
			int i  = row * DTT_SIZE21 + col;
			int gi = row * DTT_SIZE2  + col;
			float * gpu_texture_tile_gi = gpu_texture_tile + gi;
			float * rgba_i = rgbaw + i;
			// always copy 3 (1) colors + alpha
			if (colors == 3){
				if (keep_weights & 1) {
					for (int ncol = 0; ncol < colors + 1 + num_cams + colors + 1 ; ncol++) { // 12
						*(gpu_texture_tile_gi + ncol * (DTT_SIZE2 * DTT_SIZE2)) = *(rgba_i + ncol * (DTT_SIZE2 * DTT_SIZE21));
					}
				} else {
					for (int ncol = 0; ncol < colors + 1; ncol++) { // 4
						*(gpu_texture_tile_gi + ncol * (DTT_SIZE2 * DTT_SIZE2)) = *(rgba_i + ncol * (DTT_SIZE2 * DTT_SIZE21));
					}
				}
				if (keep_weights & 2) {
					float * mclt_dst_i = mclt_debayer +  i;
					float * gpu_texture_tile_raw_gi = gpu_texture_tile_gi + (colors + 1) * (DTT_SIZE2 * DTT_SIZE2); // skip colors + alpha
					for (int ncam = 0; ncam < num_cams ; ncam++) { // 8
						*(gpu_texture_tile_raw_gi + ncam * (DTT_SIZE2 * DTT_SIZE2)) = *(mclt_dst_i + (ncam * 3 + 2) * (MCLT_UNION_LEN)); // green colors
					}
				}
			} else { // assuming colors = 1
				if (keep_weights & 1) {
					for (int ncol = 0; ncol < 1 + 1 + num_cams + 1 + 1 ; ncol++) { // 8
						*(gpu_texture_tile_gi + ncol * (DTT_SIZE2 * DTT_SIZE2)) = *(rgba_i + ncol * (DTT_SIZE2 * DTT_SIZE21));
					}
				} else {
					for (int ncol = 0; ncol < 1 + 1; ncol++) { // 2
						*(gpu_texture_tile_gi + ncol * (DTT_SIZE2 * DTT_SIZE2)) = *(rgba_i + ncol * (DTT_SIZE2 * DTT_SIZE21));
					}
				}
				if (keep_weights & 2) {
					float * mclt_dst_i = mclt_debayer +  i;
					float * gpu_texture_tile_raw_gi = gpu_texture_tile_gi + (colors + 1) * (DTT_SIZE2 * DTT_SIZE2); // skip colors + alpha
					for (int ncam = 0; ncam < num_cams ; ncam++) { // 8
						*(gpu_texture_tile_raw_gi + ncam * (DTT_SIZE2 * DTT_SIZE2)) = *(mclt_dst_i + (ncam * 1 + 0) * (MCLT_UNION_LEN));
					}
				}
			}
		}
#ifdef DEBUG7A
		if ((tile_num == DBG_TILE)  && (threadIdx.x == 0) && (threadIdx.y == 0)){
			printf("textures_accumulate tile done = %d, texture_stride= %d\n",tile_num, (int) texture_stride);
		}
		__syncthreads();// __syncwarp();
#endif
	} // if (gpu_texture_tiles){ // generate non-ovelapping tiles


	tile_code &= TASK_TEXTURE_BITS;
////	if (!tile_code){
////		return; // should not happen
////	}
	// if no extra and no overlap -> nothing remains, return
	if (gpu_texture_rbg && (texture_rbg_stride != 0)) { // generate RGBA (overlapped) // keep_weights
#ifdef DEBUG7A
		if ((tile_num == DBG_TILE)  && (threadIdx.x == 0) && (threadIdx.y == 0)){
//			printf("\ntextures_accumulate accumulating tile = %d, tile_code= %d, border_tile=%d\n",
//					tile_num, (int) tile_code, border_tile);
			printf("\ntextures_accumulate accumulating tile = %d, tile_code= %d\n", tile_num, (int) tile_code);

			for (int ncol = 0; ncol <= colors; ncol++) {
				printf("\ntile[%d]\n",ncol);
				debug_print_mclt(
//						(float *) (shr1.rgbaw[ncol]),
						rgbaw + (ncol + (DTT_SIZE2 * DTT_SIZE21)),
						-1);
			}
		}
		__syncthreads();// __syncwarp();
#endif // DEBUG12
		int alpha_mode = tile_code & 0xff; //  alphaIndex[tile_code]; // only 4 lowest bits
		if (alpha_mode != 0xff){ // only add if needed, alpha_mode == 0xff (neighbors from all 8 directions) - keep as is. FIXME: alpha_mode ???
			// Calculate average value per color, need 32 shared array
			for (int ncol = 0; ncol < colors; ncol++) {
				int sum_index = threadIdx.x + threadIdx.y * TEXTURE_THREADS_PER_TILE; // 0.. 31
				texture_averaging[sum_index] = 0;
				for (int pass = 0; pass < 8; pass ++) {
					int row = pass * 2 + (threadIdx.y >> 1);
					int col = ((threadIdx.y & 1) << 3) + threadIdx.x;
					int i  = row * DTT_SIZE21 + col;
					float * rgba_i = rgbaw + i;
					texture_averaging[sum_index] += *(rgba_i + ncol * (DTT_SIZE2 * DTT_SIZE21));
				}
				__syncthreads();
				if (threadIdx.y == 0){ // combine sums
#pragma unroll
					for (int i = 1; i < 4; i++) { // reduce sums to 8
						texture_averaging[threadIdx.x] += texture_averaging[threadIdx.x + TEXTURE_THREADS_PER_TILE * i];
					}
				}
				__syncthreads();
				if ((threadIdx.y == 0) && (threadIdx.x == 0)){ // combine sums
#pragma unroll
					for (int i = 1; i < TEXTURE_THREADS_PER_TILE; i++) { // reduce sums to 8
						texture_averaging[0] += texture_averaging[i];
					}
					texture_averaging[0] /= 64; // average value for uniform field
				}
				__syncthreads();
				float avg_val = texture_averaging[0];
				// now add scale average value for each missing direction
				for (int idir = 0; idir < 8; idir ++) if ((alpha_mode & (1 << idir)) == 0) { // no tile in this direction
					/* */
					int row, col;
					switch (idir >> 1) {
					case 0:
						row = 4 + threadIdx.y;
						col = 4 + threadIdx.x;
						break;
					case 1:
						row = 4 + (threadIdx.x >> 2) + (threadIdx.y << 1);
						col = 8 + (threadIdx.x & 3);
						break;
					case 2:
						row = 8 + threadIdx.y;
						col = 4 + threadIdx.x;
						break;
					case 3:
						row = 4 + (threadIdx.x >> 2) + (threadIdx.y << 1);
						col = 4 + (threadIdx.x & 3);
						break;
					}
					int i  = row * DTT_SIZE21 + col;
					float * rgba_i = rgbaw + i;
					// always copy 3 (1) colors + alpha
					*(rgba_i + ncol * (DTT_SIZE2 * DTT_SIZE21)) += textureBlend[idir][(threadIdx.y <<3) + threadIdx.x] * avg_val;
			/*
					for (int pass = 0; pass < 8; pass ++) {
						int row1 = pass * 2 + (threadIdx.y >> 1);
						int col1 = ((threadIdx.y & 1) << 3) + threadIdx.x;
						int i  = row1 * DTT_SIZE21 + col1;
						int gi = row1 * DTT_SIZE2  + col1;
						float * rgba_i = rgbaw + i;
						// always copy 3 (1) colors + alpha
								*(rgba_i + ncol * (DTT_SIZE2 * DTT_SIZE21)) += textureBlend[idir][gi] * avg_val;
					}
			*/
				}
			}
//			__syncthreads();
		}
		int slice_stride = texture_rbg_stride * (*(woi + 3) + 1) * DTT_SIZE; // offset to the next color
		int tileY = tile_num / tilesx; // TILES-X; // slow, but 1 per tile
		int tileX = tile_num - tileY * tilesx; // TILES-X;
		int tile_x0 = (tileX - *(woi + 0)) * DTT_SIZE; //  - (DTT_SIZE/2); // may be negative == -4
		int tile_y0 = (tileY - *(woi + 1)) * DTT_SIZE; //  - (DTT_SIZE/2); // may be negative == -4
///		int height = *(woi + 3) << DTT_SIZE_LOG2;

#ifdef DEBUG7A
		if ((tile_num == DBG_TILE)  && (threadIdx.x == 0) && (threadIdx.y == 0)){
			printf("\ntextures_accumulate () tileX=%d, tileY=%d, tile_x0=%d, tile_y0=%d, slice_stride=%d\n",
					tileX, tileY, tile_x0, tile_y0, slice_stride);

			for (int ncol = 0; ncol <= colors; ncol++) {
				printf("\ntile[%d]\n",ncol);
				debug_print_mclt(
//						(float *) (shr1.rgbaw[ncol]),
						rgbaw + (ncol + (DTT_SIZE2 * DTT_SIZE21)),
						-1);
			}
		}
		__syncthreads();// __syncwarp();
#endif // DEBUG12
		// copy textures to global memory
		for (int pass = 0; pass < 8; pass ++) {
			int row = pass * 2 + (threadIdx.y >> 1);          // row    inside a tile (0..15)
			int col = ((threadIdx.y & 1) << 3) + threadIdx.x; // column inside a tile (0..15)
			int g_row = row + tile_y0;
			int g_col = col + tile_x0;
			int i  = row * DTT_SIZE21 + col;
			int gi = g_row * texture_rbg_stride  + g_col; // offset to the top left corner
			float * gpu_texture_rbg_gi = gpu_texture_rbg + gi;
			float * rgba_i = rgbaw + i;
#ifdef DEBUG7A
			if ((tile_num == DBG_TILE)  && (threadIdx.x == 0) && (threadIdx.y == 0)){
				printf("\ntextures_accumulate () pass=%d, row=%d, col=%d, g_row=%d, g_col=%d, i=%d, gi=%d\n",
						pass, row, col, g_row, g_col, i, gi);

			}
			__syncthreads();// __syncwarp();
#endif // DEBUG12
			// always copy 3 (1) colors + alpha
			if (colors == 3){
#pragma unroll
				for (int ncol = 0; ncol < 3 + 1; ncol++) { // 4
					*(gpu_texture_rbg_gi + ncol * slice_stride) += *(rgba_i + ncol * (DTT_SIZE2 * DTT_SIZE21));
				}
			} else { // assuming colors = 1
#pragma unroll
				for (int ncol = 0; ncol < 1 + 1; ncol++) { // 2
					*(gpu_texture_rbg_gi + ncol * slice_stride) += *(rgba_i + ncol * (DTT_SIZE2 * DTT_SIZE21));
				}
			}
		}
		// generate and copy per-sensor texture
		if (keep_weights & 2){ // copy individual sensors output
			for (int ncam = 0; ncam < num_cams; ncam++) {
				float * mclt_dst_ncam = mclt_debayer +  (ncam * colors ) * (MCLT_UNION_LEN);
				//if (alpha_mode){ // only multiply if needed, alpha_mode == 0 - keep as is. FIXME: alpha_mode ???
				if (alpha_mode != 0xff){
					for (int ncol = 0; ncol < colors; ncol++) {
						// calculate average value for blending
						int sum_index = threadIdx.x + threadIdx.y * TEXTURE_THREADS_PER_TILE; // 0.. 31
						texture_averaging[sum_index] = 0;
						for (int pass = 0; pass < 8; pass ++) {
							int row = pass * 2 + (threadIdx.y >> 1);
							int col = ((threadIdx.y & 1) << 3) + threadIdx.x;
							int i  = row * DTT_SIZE21 + col;
							float * rgba_i = rgbaw + i;
							texture_averaging[sum_index] += *(rgba_i + ncol * (DTT_SIZE2 * DTT_SIZE21));
						}
						__syncthreads();
						if (threadIdx.y == 0){ // combine sums
		#pragma unroll
							for (int i = 1; i < 4; i++) { // reduce sums to 8
								texture_averaging[threadIdx.x] += texture_averaging[threadIdx.x + TEXTURE_THREADS_PER_TILE * i];
							}
						}
						__syncthreads();
						if ((threadIdx.y == 0) && (threadIdx.x == 0)){ // combine sums
		#pragma unroll
							for (int i = 1; i < TEXTURE_THREADS_PER_TILE; i++) { // reduce sums to 8
								texture_averaging[0] += texture_averaging[i];
							}
							texture_averaging[0] /= 64; // average value for uniform field
						}
						__syncthreads();
						float avg_val = texture_averaging[0];

						// Possible to re-use ports_rgb_shared[], if needed (change to (calc_extra | (keep_weights & 2) in tile_combine_rgba()).
						// Now using averaging here (less noise if averaging sensor outside).
//						float avg_val = ports_rgb_shared[ncol * num_cams + ncam]; // texture_averaging[0];
						for (int idir = 0; idir < 8; idir ++) if ((alpha_mode & (1 << idir)) == 0) { // no tile in this direction
							/* */
							int row, col;
							switch (idir >> 1) {
							case 0:
								row = 4 + threadIdx.y;
								col = 4 + threadIdx.x;
								break;
							case 1:
								row = 4 + (threadIdx.x >> 2) + (threadIdx.y << 1);
								col = 8 + (threadIdx.x & 3);
								break;
							case 2:
								row = 8 + threadIdx.y;
								col = 4 + threadIdx.x;
								break;
							case 3:
								row = 4 + (threadIdx.x >> 2) + (threadIdx.y << 1);
								col = 4 + (threadIdx.x & 3);
								break;
							}
							int i  = row * DTT_SIZE21 + col;
							float * mclt_dst_i = mclt_dst_ncam +  i;
							int gi = (threadIdx.y <<3) + threadIdx.x;
							*(mclt_dst_i + ncol * (MCLT_UNION_LEN)) += textureBlend[idir][gi] * avg_val;
						}
						__syncthreads(); // needed?
					}
				}

				for (int pass = 0; pass < 8; pass ++) {
					int row = pass * 2 + (threadIdx.y >> 1);          // row    inside a tile (0..15)
					int col = ((threadIdx.y & 1) << 3) + threadIdx.x; // column inside a tile (0..15)
					int g_row = row + tile_y0;
					int g_col = col + tile_x0;
					int i  = row * DTT_SIZE21 + col;
					int gi = g_row * texture_rbg_stride  + g_col; // offset to the top left corner
					float * gpu_texture_rbg_gi = gpu_texture_rbg + gi + (colors + 1 + colors * ncam) * slice_stride;
					float * mclt_dst_i = mclt_dst_ncam +  i;
					for (int ncol = 0; ncol < colors; ncol++) { // 4
						*(gpu_texture_rbg_gi + ncol * slice_stride) += *(mclt_dst_i + ncol * (MCLT_UNION_LEN));
					}
				}
			}
		}
	} // 	if (gpu_texture_rbg) { // generate RGBA
	if (calc_extra){ // gpu_diff_rgb_combo
		__syncthreads(); // needed?
#ifdef DEBUG7A
		if ((tile_num == DBG_TILE) && (threadIdx.x == 0) && (threadIdx.y == 0)){
			printf("\n3. tile_indx=%d, tile_num=%d\n",tile_indx,tile_num);
			printf    ("max_diff: ");for (int ccam = 0; ccam < num_cams; ccam++) {printf("%f, ",max_diff_shared[ccam]);} printf("\n");
			for (int ccol = 0; ccol < colors; ccol++){
				printf("color%d:   ",ccol);for (int ccam = 0; ccam < num_cams; ccam++) {printf("%f, ",ports_rgb_shared[ccol * num_cams +ccam]);} printf("\n");
			}
			for (int i = 0; i < TEXTURE_THREADS_PER_TILE; i++){
				printf("tmp[%d]:   ",i); for (int ccam = 0; ccam < num_cams; ccam++) {printf("%f, ",max_diff_tmp[ccam * TEXTURE_THREADS_PER_TILE + i]);} printf("\n");
			}
			for (int ncol = 0; ncol < colors; ncol++){
				printf("\n%d:total ",ncol);
				for (int ccam = 0; ccam < num_cams; ccam++) {printf("%f, ",ports_rgb_shared[ ncol *num_cams +ccam]);} printf("\n");
				for (int i = 0; i < TEXTURE_THREADS_PER_TILE; i++){
					printf("tmp[%d] ",i);
					for (int ccam = 0; ccam < num_cams; ccam++) {printf("%f, ",ports_rgb_tmp[(ncol*num_cams + ccam) * TEXTURE_THREADS_PER_TILE+ i]);} printf("\n");
				}
			}
		}
		__syncthreads();
//DBG_TILE
#endif// #ifdef DEBUG7A


#ifdef DEBUG7A
		if ((tile_num == DBG_TILE) && (threadIdx.x == 0) && (threadIdx.y == 0)){
			printf("\n4. tile_indx=%d, tile_num=%d, DBG_TILE = %d\n",tile_indx,tile_num, DBG_TILE);
		}
		__syncthreads();
//DBG_TILE
#endif// #ifdef DEBUG7A

		int tile_offset = (linescan_order ? tile_num : tile_indx) * num_cams* (colors + 1);
		for (int camera_num_offs = 0; camera_num_offs < num_cams; camera_num_offs+= blockDim.y) {// assuming num_cams is multiple blockDim.y
			int camera_num = threadIdx.y + camera_num_offs;

// Maybe needs to be changed back if output data should match tile index in task list, not the tile absolute position
//			float * pdiff_rgb_combo = gpu_diff_rgb_combo + tile_num  * num_cams* (colors + 1) + camera_num;//
			float * pdiff_rgb_combo = gpu_diff_rgb_combo + tile_offset + camera_num;//
			if (threadIdx.x == 0){
				*pdiff_rgb_combo = max_diff_shared[camera_num];
			}
			if (threadIdx.x < colors){
				*(pdiff_rgb_combo + (threadIdx.x + 1) * num_cams) = ports_rgb_shared[threadIdx.x * num_cams + camera_num];// [color][camera]
			}
		}
	} // if (calc_extra){ // gpu_diff_rgb_combo
} // textures_accumulate()

__device__ int get_textures_shared_size( // in bytes
//__device__ int get_textures_shared_size( // in bytes
	    int                num_cams,     // actual number of cameras
	    int                num_colors,   // actual number of colors: 3 for RGB, 1 for LWIR/mono
		int *              offsets){     // in floats
//	int shared_floats = 0;
	int offs = 0;
//	int texture_threads_per_tile = TEXTURE_THREADS/num_cams;
	if (offsets) offsets[0] = offs;
	offs += num_cams * num_colors * 2 * DTT_SIZE * DTT_SIZE21; //float mclt_tiles         [NUM_CAMS][NUM_COLORS][2*DTT_SIZE][DTT_SIZE21]
	if (offsets) offsets[1] = offs;
	offs += num_cams * num_colors * 4 * DTT_SIZE * DTT_SIZE1;  // float clt_tiles         [NUM_CAMS][NUM_COLORS][4][DTT_SIZE][DTT_SIZE1]
	if (offsets) offsets[2] = offs;
	int mclt_tmp_size = num_cams * num_colors * DTT_SIZE2 * DTT_SIZE21;                // [NUM_CAMS][NUM_COLORS][DTT_SIZE2][DTT_SIZE21]
	int rgbaw_size =    (2* (num_colors + 1) + num_cams) * DTT_SIZE2 * DTT_SIZE21;     // [NUM_COLORS + 1 + NUM_CAMS + NUM_COLORS + 1][DTT_SIZE2][DTT_SIZE21]
	offs += (rgbaw_size > mclt_tmp_size) ? rgbaw_size : mclt_tmp_size;
	if (offsets) offsets[3] = offs;
	offs += num_cams * 2;                                      // float port_offsets      [NUM_CAMS][2];
	if (offsets) offsets[4] = offs;
	offs += num_colors * num_cams;                             // float ports_rgb_shared  [NUM_COLORS][NUM_CAMS];
	if (offsets) offsets[5] = offs;
	offs += num_cams;                                          // float max_diff_shared   [NUM_CAMS];
	if (offsets) offsets[6] = offs;
	offs += num_cams * TEXTURE_THREADS_PER_TILE;               // float max_diff_tmp      [NUM_CAMS][TEXTURE_THREADS_PER_TILE]
	if (offsets) offsets[7] = offs;
	offs += num_colors * num_cams *  TEXTURE_THREADS_PER_TILE; //float ports_rgb_tmp     [NUM_COLORS][NUM_CAMS][TEXTURE_THREADS_PER_TILE];
	if (offsets) offsets[8] = offs;
	return sizeof(float) * offs; // shared_floats;

}

/**
 * Generate per-camera aberration-corrected images from the in-memory frequency domain representation.
 * This kernel launches others with CDP, from CPU it is just <<<1,1>>>
 * @param num_cams             actual number of cameras, <= NUM_CAMS
 * @param gpu_clt              array of num_cams (actual) pointers to the CLT (frequency domain) data [num_cams][TILES-Y][TILES-X][colors][DTT_SIZE*DTT_SIZE]
 * @param gpu_corr_images      array of num_cams (actual) pointers to the output images, [width, colors* height]. width height are from woi_twidth, woi_theight
 * @param apply_lpf            TODO: now it is not used - restore after testing
 * @param colors               number of colors used:  3 for RGB or 1 for monochrome
 * @param woi_twidth           full image width in tiles
 * @param woi_theight          full image height in tiles
 * @param dstride              output images stride in floats
 */
extern "C"
__global__ void imclt_rbg_all(
		int                num_cams,
		float           ** gpu_clt,            // [?][TILES-Y][TILES-X][colors][DTT_SIZE*DTT_SIZE]
		float           ** gpu_corr_images,    // [?][width, colors* height]
		int                apply_lpf,		   // TODO: now it is not used - restore?
		int                colors,
		int                woi_twidth,
		int                woi_theight,
		const size_t       dstride)            // in floats (pixels)
{

//	int num_cams = sizeof(gpu_clt)/sizeof(&gpu_clt[0]);
	dim3 threads_erase8x8(DTT_SIZE, NUM_THREADS/DTT_SIZE, 1);
	dim3 grid_erase8x8_right_col (1, woi_theight + 1, 1);
	dim3 grid_erase8x8_bottom_row(woi_twidth + 1, 1, 1);
	dim3 threads_imclt(IMCLT_THREADS_PER_TILE, IMCLT_TILES_PER_BLOCK, 1);
	if (threadIdx.x == 0) { // anyway 1,1,1
		for (int ncam = 0; ncam < num_cams; ncam++) { // was NUM_CAMS
			for (int color = 0; color < colors; color++) {
				// clear right and bottom 8-pixel column and row
				float *right_col = gpu_corr_images[ncam] + dstride * (woi_theight * DTT_SIZE + DTT_SIZE) * color + (woi_twidth * DTT_SIZE);
				erase8x8<<<grid_erase8x8_right_col,threads_erase8x8>>>(
						right_col,             // float           * gpu_top_left,
						dstride);              // const size_t      dstride);
				float *bottom_row = gpu_corr_images[ncam] + dstride * (woi_theight * DTT_SIZE + DTT_SIZE) * color +  dstride * (woi_theight * DTT_SIZE);
				erase8x8<<<grid_erase8x8_bottom_row,threads_erase8x8>>>(
						bottom_row,             // float           * gpu_top_left,
						dstride);              // const size_t      dstride);

				for (int v_offs = 0; v_offs < 2; v_offs++){
					for (int h_offs = 0; h_offs < 2; h_offs++){
						int tilesy_half = (woi_theight + (v_offs ^ 1)) >> 1;
						int tilesx_half = (woi_twidth + (h_offs ^ 1)) >> 1;
						int tiles_in_pass = tilesy_half * tilesx_half;
						dim3 grid_imclt((tiles_in_pass + IMCLT_TILES_PER_BLOCK-1) / IMCLT_TILES_PER_BLOCK,1,1);
						//    				printf("grid_imclt=   (%d, %d, %d)\n",grid_imclt.x,   grid_imclt.y,   grid_imclt.z);
						imclt_rbg<<<grid_imclt,threads_imclt>>>(
								gpu_clt[ncam],         // float           * gpu_clt,     // [TILES-Y][TILES-X][NUM_COLORS][DTT_SIZE*DTT_SIZE]
								gpu_corr_images[ncam], // float           * gpu_rbg,     // WIDTH, 3 * HEIGHT
								1,                     // int               apply_lpf,
								colors,                // int               colors,      // defines lpf filter
								color,                 // int               color,       // defines location of clt data
								v_offs,                // int               v_offset,
								h_offs,                // int               h_offset,
								woi_twidth,            // int               woi_twidth,  // will increase by DTT_SIZE (todo - cut away?)
								woi_theight,           // int               woi_theight, // will increase by DTT_SIZE (todo - cut away?)
								dstride);              // const size_t      dstride);    // in floats (pixels)
						cudaDeviceSynchronize();
					}
				}
			}
		}
	}
}

/**
 * Clear 8x8 tiles, used to erase right and bottom 8-pixel wide column/row before imclt_rbg
 * @param gpu_top_left - pointer to the top-left corner of the firsr tile to erase
 * @param dstride - offset for 1 pixel step down
 * block.x - horizontal tile  offset
 * block.y - vertical tile offset
 * 0<=thread.x < 8 - horizontal pixel offset
 * 0<=thread.y < 4 - vertical pixel offset
 */
extern "C"
__global__ void erase8x8(
		float           * gpu_top_left,
		const size_t      dstride)
{
	float * pixel = gpu_top_left + (((blockIdx.y * DTT_SIZE) + threadIdx.y) * dstride) + ((blockIdx.x * DTT_SIZE) + threadIdx.x);
	* pixel = 0.0f;
	pixel += dstride * blockDim.y; // add 4 pixel rows (assuming blockDim.x==4)
	* pixel = 0.0f;
}

/**
 * Helper kernel for imclt_rbg_all(), generate per-camera -per color image from the in-memory frequency domain representation.
 * @param gpu_clt              CLT (frequency domain) data [of][TILES-Y][TILES-X][colors][DTT_SIZE*DTT_SIZE]
 * @param gpu_corr_images      output images, [width, colors* height]. width height are from woi_twidth, woi_theight
 * @param apply_lpf            TODO: now it is not used - restore after testing
 * @param colors               number of colors used:  3 for RGB or 1 for monochrome
 * @param color                color to process
 * @param v_offset             vertical offset (0,1) for accumulating overlapping tiles
 * @param h_offset             horizontal  offset (0,1) for accumulating overlapping tiles
 * @param woi_twidth           full image width in tiles
 * @param woi_theight          full image height in tiles
 * @param dstride              output images stride in floats
 */
extern "C"
__global__ void imclt_rbg(
		float           * gpu_clt,            // [TILES-Y][TILES-X][colors][DTT_SIZE*DTT_SIZE]
		float           * gpu_rbg,            // WIDTH, 3 * HEIGHT
		int               apply_lpf,
		int               colors, // was mono
		int               color,
		int               v_offset,
		int               h_offset,
		int               woi_twidth,  // will increase by DTT_SIZE (todo - cut away?)
		int               woi_theight, // will increase by DTT_SIZE (todo - cut away?)
		const size_t      dstride)            // in floats (pixels)
{
	float *color_plane = gpu_rbg + dstride * (woi_theight * DTT_SIZE + DTT_SIZE) * color;
	int pass =           (v_offset << 1) + h_offset;     		// 0..3 to correctly accumulate 16x16 tiles stride 8
	int tile_in_block = threadIdx.y;
	int tile_num = blockIdx.x * IMCLT_TILES_PER_BLOCK + tile_in_block;
	int tilesx_half = (woi_twidth + (h_offset ^ 1)) >> 1;
	int tileY_half =  tile_num / tilesx_half;
	int tileX_half =  tile_num - tileY_half * tilesx_half;
	int tileY = (tileY_half << 1) + v_offset;
	int tileX = (tileX_half << 1) + h_offset;
	if (tileY >= woi_theight) {
		return; // just testing with a single tile
	}
#ifdef DEBUG4
	if (threadIdx.x == 0) {
		if (tileY == DBG_TILE_Y) {
			printf("tileX == %d, tileY = %d\n",tileX, tileY);
		}
		if (tileX == DBG_TILE_X) {
			printf("tileX == %d, tileY = %d\n",tileX, tileY);
		}
		if ((tileX == DBG_TILE_X) && (tileY == DBG_TILE_Y)) {
			printf("tileX == %d, tileY = %d\n",tileX, tileY);
		}
	}


#endif

	int thr3 =    threadIdx.x >> 3;
	int column =  threadIdx.x; // modify to use 2 * 8 threads, if needed.

    __shared__ float clt_tiles        [IMCLT_TILES_PER_BLOCK][4][DTT_SIZE][DTT_SIZE1];
    __shared__ float mclt_tiles       [IMCLT_TILES_PER_BLOCK][DTT_SIZE2][DTT_SIZE21];

    // copy clt (frequency domain data)
    float * clt_tile = ((float *) clt_tiles) +  tile_in_block * (4 * DTT_SIZE * DTT_SIZE1); // top left quadrant0
    float * gpu_tile = ((float *) gpu_clt) +  ((tileY * woi_twidth + tileX) * colors + color) * (4 * DTT_SIZE * DTT_SIZE); // top left quadrant0

    clt_tile += column + thr3; // first 2 rows
    gpu_tile += column;  // first 2 rows
    if (apply_lpf) {
    	// lpf - covers 2 rows, as there there are 16 threads
		float *lpf0 = lpf_data[(colors == 1)? 3 :color] + threadIdx.x; // lpf_data[3] - mono
#pragma unroll
		for (int q = 0; q < 4; q++){
			float *lpf = lpf0;
			for (int i = 0; i < DTT_SIZE/2; i++){
				*clt_tile= *gpu_tile * (*lpf);
				clt_tile += (2 * DTT_SIZE1);
				gpu_tile += (2 * DTT_SIZE);
				lpf +=      (2 * DTT_SIZE);
			}
		}
    } else {
#pragma unroll
    	for (int i = 0; i < DTT_SIZE2; i++){
    		*clt_tile= *gpu_tile;
    		clt_tile += (2 * DTT_SIZE1);
    		gpu_tile += (2 * DTT_SIZE);
    	}
    }
	float * mclt_top = ((float*) mclt_tiles) +  tile_in_block * (DTT_SIZE2 * DTT_SIZE21) + column;
	float * rbg_top = color_plane + (tileY * DTT_SIZE)* dstride + (tileX * DTT_SIZE) + column;
	float * mclt_tile = mclt_top;

	if (pass == 0){ // just set mclt tile to all 0
#pragma unroll
		for (int i = 0; i < DTT_SIZE2; i++){
			*mclt_tile= 0.0f;
			mclt_tile += DTT_SIZE21;
		}
	} else {
		float * rbg_p = rbg_top;
#pragma unroll
			for (int i = 0; i < DTT_SIZE2; i++){
				*mclt_tile= *rbg_p;
				mclt_tile += DTT_SIZE21;
				rbg_p +=     dstride; // DTT_SIZE2;
			}
	}
	__syncthreads();// __syncwarp();
	imclt(
			((float*) clt_tiles) +  tile_in_block * (4 * DTT_SIZE * DTT_SIZE1), // float * clt_tile,   //        [4][DTT_SIZE][DTT_SIZE1], // +1 to alternate column ports [4][8][9]
			((float*) mclt_tiles) +  tile_in_block * (DTT_SIZE2 * DTT_SIZE21)); // float * mclt_tile )
	__syncthreads();// __syncwarp();

#ifdef DEBUG5
    if (((threadIdx.x) == 0) &&(tileX == DBG_TILE_X)  && (tileY == DBG_TILE_Y)){
//        printf("\nMCLT Tiles after IMCLT\n");
		printf("tileX == %d, tileY = %d\n",tileX, tileY);
    	debug_print_mclt(mclt_tile, -1); // only 1 quadrant for R,B and 2 - for G
    }
    __syncthreads();// __syncwarp();
#endif



    //	save result (back)
    float * rbg_p = rbg_top;
    mclt_tile =     mclt_top;

#ifdef DBG_MARK_DBG_TILE
    if ((tileX == DBG_TILE_X)  && (tileY == DBG_TILE_Y)){
#pragma unroll
    	for (int i = 0; i < DTT_SIZE2; i++){
    		*rbg_p = (*mclt_tile) * 2.0; // just testing
    		mclt_tile += DTT_SIZE21;
    		rbg_p +=     dstride; // DTT_SIZE2; // FIXME
    	}
    } else {
#endif // #ifdef DBG_MARK_DBG_TILE
#pragma unroll
    	for (int i = 0; i < DTT_SIZE2; i++){
    		*rbg_p = *mclt_tile;
    		mclt_tile += DTT_SIZE21;
    		rbg_p +=     dstride; // DTT_SIZE2; // FIXME
    	}
#ifdef DBG_MARK_DBG_TILE
    }
#endif //#ifdef DBG_MARK_DBG_TILE
}

/**
 * Fractional pixel shift (phase rotation), horizontal. In-place. uses 8 threads (.x)
 * Used in convert_direct() -> convert_correct_tiles() -> convertCorrectTile
 *
 * @param clt_tile             transform domain representation of a tile: [4][8][8+1], // +1 to alternate column ports
 * @param residual_shift       fractional pixel shift [-0.5, +0.5)
 */
__device__ void shiftTileHor(
		float * clt_tile, //       [4][DTT_SIZE][DTT_SIZE1], // +1 to alternate column ports
		float residual_shift )
{
	int joffs = threadIdx.x; // * DTT_SIZE1;
	float * clt_tile_j0 = clt_tile +    joffs;                // ==&clt_tile[0][j][0]
	float * clt_tile_j1 = clt_tile_j0 + (DTT_SIZE1*DTT_SIZE); // ==&clt_tile[1][j][0]
	float * clt_tile_j2 = clt_tile_j1 + (DTT_SIZE1*DTT_SIZE); // ==&clt_tile[2][j][0]
	float * clt_tile_j3 = clt_tile_j2 + (DTT_SIZE1*DTT_SIZE); // ==&clt_tile[3][j][0]
	float x = residual_shift * ((threadIdx.x << 1 ) +1) * (0.5f/ DTT_SIZE);
	float ch =  cospif(x);
	float sh =  sinpif(x);
#pragma unroll
	for (int i = 0; i < DTT_SIZE; i++) {
		float clt_tile_a = *clt_tile_j0;
		float clt_tile_b = *clt_tile_j1;
		*clt_tile_j0 =  clt_tile_a * ch - clt_tile_b * sh;
		*clt_tile_j1 =  clt_tile_a * sh + clt_tile_b * ch;

		clt_tile_a = *clt_tile_j2;
		clt_tile_b = *clt_tile_j3;
		*clt_tile_j2 =  clt_tile_a * ch - clt_tile_b * sh;
		*clt_tile_j3 =  clt_tile_a * sh + clt_tile_b * ch;

		clt_tile_j0 +=DTT_SIZE1;
		clt_tile_j1 +=DTT_SIZE1;
		clt_tile_j2 +=DTT_SIZE1;
		clt_tile_j3 +=DTT_SIZE1;
	}
}

/**
 * Fractional pixel shift (phase rotation), vertical. In-place. uses 8 threads (.x)
 * Used in convert_direct() -> convert_correct_tiles() -> convertCorrectTile
 *
 * @param clt_tile             transform domain representation of a tile: [4][8][8+1], // +1 to alternate column ports
 * @param residual_shift       fractional pixel shift [-0.5, +0.5)
 */
__device__ void shiftTileVert(
		float * clt_tile, //       [4][DTT_SIZE][DTT_SIZE1], // +1 to alternate column ports
		float residual_shift)
{
	int joffs = threadIdx.x * DTT_SIZE1;
	float * clt_tile_j0 = clt_tile +    joffs;                // ==&clt_tile[0][j][0]
	float * clt_tile_j1 = clt_tile_j0 + (DTT_SIZE1*DTT_SIZE); // ==&clt_tile[1][j][0]
	float * clt_tile_j2 = clt_tile_j1 + (DTT_SIZE1*DTT_SIZE); // ==&clt_tile[2][j][0]
	float * clt_tile_j3 = clt_tile_j2 + (DTT_SIZE1*DTT_SIZE); // ==&clt_tile[3][j][0]
	float x = residual_shift * ((threadIdx.x << 1 ) +1) * (0.5f/ DTT_SIZE);
	float ch =  cospif(x);
	float sh =  sinpif(x);
#pragma unroll
	for (int i = 0; i < DTT_SIZE; i++) {
		float clt_tile_a = *clt_tile_j0;
		float clt_tile_b = *clt_tile_j2;
		*clt_tile_j0 =  clt_tile_a * ch - clt_tile_b * sh;
		*clt_tile_j2 =  clt_tile_a * sh + clt_tile_b * ch;

		clt_tile_a = *clt_tile_j1;
		clt_tile_b = *clt_tile_j3;
		*clt_tile_j1 =  clt_tile_a * ch - clt_tile_b * sh;
		*clt_tile_j3 =  clt_tile_a * sh + clt_tile_b * ch;

		clt_tile_j0 ++;
		clt_tile_j1 ++;
		clt_tile_j2 ++;
		clt_tile_j3 ++;
	}
}

/**
 * Convolve image tile with the kernel tile in transform domain
 * Used in convert_direct() -> convert_correct_tiles() -> convertCorrectTile
 *
 * @param clt_tile             transform domain representation of a tile [4][8][8+1], // +1 to alternate column ports
 * @param kernel               transform domain representation of a kernel [4][8][8+1], // +1 to alternate column ports
 */
__device__ void convolveTiles(
		float* clt_tile, //    [4][DTT_SIZE][DTT_SIZE1], // 4 quadrants of the clt data, rows extended to optimize shared ports
		float* kernel) //      [4][DTT_SIZE][DTT_SIZE1]) // 4 quadrants of the CLT kernel (DTT3 converted)
{
	int joffs = threadIdx.x * DTT_SIZE1;
	float * kernel_j; //  =   kernel +      joffs;                // ==&kernel[0][j][0]
	float * clt_tile_j0 = clt_tile +    joffs;                // ==&clt_tile[0][j][0]
	float * clt_tile_j1 = clt_tile_j0 + (DTT_SIZE1*DTT_SIZE); // ==&clt_tile[1][j][0]
	float * clt_tile_j2 = clt_tile_j1 + (DTT_SIZE1*DTT_SIZE); // ==&clt_tile[2][j][0]
	float * clt_tile_j3 = clt_tile_j2 + (DTT_SIZE1*DTT_SIZE); // ==&clt_tile[3][j][0]
//#pragma unroll
	for (int i = 0; i < DTT_SIZE; i++){
		// k=0
		kernel_j =   kernel + joffs + i;
		float krn = *(kernel_j);
		float r0 =  *(clt_tile_j0) * krn;
		float r1 =  *(clt_tile_j1) * krn;
		float r2 =  *(clt_tile_j2) * krn;
		float r3 =  *(clt_tile_j3) * krn;
		// k = 1
		kernel_j += (DTT_SIZE1*DTT_SIZE);
		krn = *(kernel_j);
		r0 -=  *(clt_tile_j1) * krn;
		r1 +=  *(clt_tile_j0) * krn;
		r2 -=  *(clt_tile_j3) * krn;
		r3 +=  *(clt_tile_j2) * krn;
		// k=2
		kernel_j += (DTT_SIZE1*DTT_SIZE);
		krn = *(kernel_j);
		r0 -=  *(clt_tile_j2) * krn;
		r1 -=  *(clt_tile_j3) * krn;
		r2 +=  *(clt_tile_j0) * krn;
		r3 +=  *(clt_tile_j1) * krn;
		// k=3
		kernel_j += (DTT_SIZE1*DTT_SIZE);
		krn = *(kernel_j);
		r0 +=  *(clt_tile_j3) * krn;
		r1 -=  *(clt_tile_j2) * krn;
		r2 -=  *(clt_tile_j1) * krn;
		r3 +=  *(clt_tile_j0) * krn;
		*(clt_tile_j0)= r0;
		*(clt_tile_j1)= r1;
		*(clt_tile_j2)= r2;
		*(clt_tile_j3)= r3;
		clt_tile_j0 ++;
		clt_tile_j1 ++;
		clt_tile_j2 ++;
		clt_tile_j3 ++;
	}
}



/**
 * Calculate 2D correlation of a pair from CLT representation and accumulate with a specified color weight
 * Called from correlate2D()->correlate2D_inner()
 *
 * @param scale                weight of the current component for accumulation.
 * @param clt_tile1            transform domain representation of a tile [4][8][8+1], 4 quadrants of the clt data 1,
 *                             rows extended to optimize shared ports
 * @param clt_tile2            transform domain representation of a tile [4][8][8+1]
 * @param corr_tile            result tile [4][8][8+1], should be initialized with resetCorrelation() before
 *                             the first color component.
 */
__device__ void correlateAccumulateTiles(
		float  scale,     // scale correlation
		float* clt_tile1, //    [4][DTT_SIZE][DTT_SIZE1], // 4 quadrants of the clt data 1, rows extended to optimize shared ports
		float* clt_tile2, //    [4][DTT_SIZE][DTT_SIZE1], // 4 quadrants of the clt data 2, rows extended to optimize shared ports
		float* corr_tile) //    [4][DTT_SIZE][DTT_SIZE1]) // 4 quadrants of the correlation result
{
	int joffs = threadIdx.x * DTT_SIZE1;
	float * clt_tile2_j; //  =   clt_tile2 +      joffs;                // ==&clt_tile2[0][j][0]
	float * clt_tile1_j0 = clt_tile1 +    joffs;                // ==&clt_tile[0][j][0]
	float * clt_tile1_j1 = clt_tile1_j0 + (DTT_SIZE1*DTT_SIZE); // ==&clt_tile[1][j][0]
	float * clt_tile1_j2 = clt_tile1_j1 + (DTT_SIZE1*DTT_SIZE); // ==&clt_tile[2][j][0]
	float * clt_tile1_j3 = clt_tile1_j2 + (DTT_SIZE1*DTT_SIZE); // ==&clt_tile[3][j][0]

	float * corr_tile_j0 = corr_tile +    joffs;                // ==&clt_tile[0][j][0]
	float * corr_tile_j1 = corr_tile_j0 + (DTT_SIZE1*DTT_SIZE); // ==&clt_tile[1][j][0]
	float * corr_tile_j2 = corr_tile_j1 + (DTT_SIZE1*DTT_SIZE); // ==&clt_tile[2][j][0]
	float * corr_tile_j3 = corr_tile_j2 + (DTT_SIZE1*DTT_SIZE); // ==&clt_tile[3][j][0]
//    __syncthreads(); // *** TESTING ***
//#pragma unroll
	for (int i = 0; i < DTT_SIZE; i++){
		// k=0
		clt_tile2_j =   clt_tile2 + joffs + i;
		float clt2 = *(clt_tile2_j);
		float r0 =  *(clt_tile1_j0) * clt2;
		float r1 = -*(clt_tile1_j1) * clt2;
		float r2 = -*(clt_tile1_j2) * clt2;
		float r3 =  *(clt_tile1_j3) * clt2;
		// k = 1
		clt_tile2_j += (DTT_SIZE1*DTT_SIZE);
		clt2 = *(clt_tile2_j);
		r0 +=  *(clt_tile1_j1) * clt2;
		r1 +=  *(clt_tile1_j0) * clt2;
		r2 -=  *(clt_tile1_j3) * clt2;
		r3 -=  *(clt_tile1_j2) * clt2;
		// k=2
		clt_tile2_j += (DTT_SIZE1*DTT_SIZE);
		clt2 = *(clt_tile2_j);
		r0 +=  *(clt_tile1_j2) * clt2;
		r1 -=  *(clt_tile1_j3) * clt2;
		r2 +=  *(clt_tile1_j0) * clt2;
		r3 -=  *(clt_tile1_j1) * clt2;
		// k=3
		clt_tile2_j += (DTT_SIZE1*DTT_SIZE);
		clt2 = *(clt_tile2_j);
		r0 +=  *(clt_tile1_j3) * clt2;
		r1 +=  *(clt_tile1_j2) * clt2;
		r2 +=  *(clt_tile1_j1) * clt2;
		r3 +=  *(clt_tile1_j0) * clt2;

		*(corr_tile_j0) += scale * r0;
		*(corr_tile_j1) += scale * r1;
		*(corr_tile_j2) += scale * r2;
		*(corr_tile_j3) += scale * r3;
		clt_tile1_j0 ++;
		clt_tile1_j1 ++;
		clt_tile1_j2 ++;
		clt_tile1_j3 ++;
		corr_tile_j0 ++;
		corr_tile_j1 ++;
		corr_tile_j2 ++;
		corr_tile_j3 ++;
//        __syncthreads(); // *** TESTING ***

	}
}

/**
 * Initailize 2D correlation (CLT representation) before accumulating colors.
 * Called from correlate2D()->correlate2D_inner()
 *
 * @param corr_tile            pointer to a tile [4][8][8+1] to be reset to all 0-s.
 */
__device__ void resetCorrelation(
		float* corr_tile) //    [4][DTT_SIZE][DTT_SIZE1]) // 4 quadrants of the correlation result
{
	int joffs = threadIdx.x * DTT_SIZE1;

	float * corr_tile_j0 = corr_tile +    joffs;                // k = 0
	float * corr_tile_j1 = corr_tile_j0 + (DTT_SIZE1*DTT_SIZE); // k = 1
	float * corr_tile_j2 = corr_tile_j1 + (DTT_SIZE1*DTT_SIZE); // k = 2
	float * corr_tile_j3 = corr_tile_j2 + (DTT_SIZE1*DTT_SIZE); // k = 3
//#pragma unroll
	for (int i = 0; i < DTT_SIZE; i++){

		*(corr_tile_j0) = 0;
		*(corr_tile_j1) = 0;
		*(corr_tile_j2) = 0;
		*(corr_tile_j3) = 0;
		corr_tile_j0 ++;
		corr_tile_j1 ++;
		corr_tile_j2 ++;
		corr_tile_j3 ++;
	}
}

/**
 * Normalize 2D correlation (CLT representation) to make it phase correlation.
 * Called from correlate2D()->correlate2D_inner()
 *
 * @param clt_tile             pointer to a correlation result tile [4][8][8+1] to be normalized
 * @param fat_zero2            value to add to amplitudes for regularization. Absolute value,
 *                             scale if needed outside.
 */
__device__ void normalizeTileAmplitude(
		float * clt_tile, //       [4][DTT_SIZE][DTT_SIZE1], // +1 to alternate column ports
		float fat_zero2 )  // fat zero is absolute, scale it outside
{
	int joffs = threadIdx.x * DTT_SIZE1;
	float * clt_tile_j0 = clt_tile +    joffs;                // ==&clt_tile[0][j][0]
	float * clt_tile_j1 = clt_tile_j0 + (DTT_SIZE1*DTT_SIZE); // ==&clt_tile[1][j][0]
	float * clt_tile_j2 = clt_tile_j1 + (DTT_SIZE1*DTT_SIZE); // ==&clt_tile[2][j][0]
	float * clt_tile_j3 = clt_tile_j2 + (DTT_SIZE1*DTT_SIZE); // ==&clt_tile[3][j][0]
#pragma unroll
	for (int i = 0; i < DTT_SIZE; i++) {
		float s2 = fat_zero2 +
				*(clt_tile_j0) * *(clt_tile_j0) +
				*(clt_tile_j1) * *(clt_tile_j1) +
				*(clt_tile_j2) * *(clt_tile_j2) +
				*(clt_tile_j3) * *(clt_tile_j3);
#ifdef FASTMATH
		float scale = __frsqrt_rn(s2); // 1.0/sqrt(s2)
#else
		float scale = rsqrtf(s2); // 1.0/sqrt(s2)
#endif
		*(clt_tile_j0) *= scale;
		*(clt_tile_j1) *= scale;
		*(clt_tile_j2) *= scale;
		*(clt_tile_j3) *= scale;

		clt_tile_j0 ++; // =DTT_SIZE1;
		clt_tile_j1 ++; // =DTT_SIZE1;
		clt_tile_j2 ++; // =DTT_SIZE1;
		clt_tile_j3 ++; // =DTT_SIZE1;
	}
}


/**
 * Used in convert_direct()->convert_correct_tiles() to convert/correct a single tile
 *
 * @param num_cams             actual number of cameras used (<=NUM_CAMS),
 * @param num_colors,          actual number of colors used  (<=NUM_COLORS)
 * @param gpu_kernel_offsets   array of per-camera pointers to array of struct CltExtra (one element per kernel)
 * @param gpu_kernels          array of per-camera pointers to array of kernels (clt representation)
 * @param gpu_images           array of per-camera pointers to Bayer images
 * @param gpu_clt              output array of per-camera aberration-corrected transform-domain image representations
 * @param color                color component
 * @param lpf_mask             apply lpf to colors : bit 0 - red, bit 1 - blue, bit2 - green. Now - always 0 !
 * @param centerX              full X-offset of the tile center, calculated from the geometry, distortions and disparity
 * @param centerY              full Y-offset of the tile center
 * @param txy                  integer value combining tile X (low 16 bits) and tile Y (high 16 bits)
 * @param tscale               float value to scale result. 0 - set. >0 scale and set, <0 subtract
 * @param dstride              stride (in floats) for the input Bayer images
 * @param clt_tile             image tile in shared memory [4][DTT_SIZE][DTT_SIZE1] (just allocated)
 * @param clt_kernels          kernel tile in shared memory [4][DTT_SIZE][DTT_SIZE1] (just allocated)
 * @param int_topleft          tile left and top, declared in shared memory (just allocated) [2]
 * @param residual_shift       tile fractional pixel shift (x,y) in shared memory (just allocated) [2]
 * @param window_hor_cos       array in shared memory for window horizontal cosine [2*DTT_SIZE]
 * @param window_hor_sin       array in shared memory for window horizontal sine   [2*DTT_SIZE]
 * @param window_vert_cos      array in shared memory for window vertical cosine   [2*DTT_SIZE]
 * @param window_vert_cos      array in shared memory for window vertical sine     [2*DTT_SIZE]
 * @param woi_width            image width (was constant IMG-WIDTH, now variable to use with EO+LWIR
 * @param woi_height           image height (was constant IMG-HEIGHT, now variable to use with EO+LWIR
 * @param kernels_hor          number of deconvolution kernels per image width
 * @param kernels_vert         number of deconvolution kernels per image height
 */
__device__ void convertCorrectTile(
		int                   num_colors, //*
		struct CltExtra     * gpu_kernel_offsets, // [tileY][tileX][color]
		float               * gpu_kernels,        // [tileY][tileX][color]
		float               * gpu_images,
		float               * gpu_clt,
		const int             color,
		const int             lpf_mask, // now 0
		const float           centerX,
		const float           centerY,
		const int             txy,
		const float           tscale,
		const size_t          dstride, // in floats (pixels)
		float               * clt_tile, //        [4][DTT_SIZE][DTT_SIZE1], // +1 to alternate column ports
		float               * clt_kernels, //      [4][DTT_SIZE][DTT_SIZE1], // +1 to alternate column ports
		int   int_topleft     [2],
		float residual_shift  [2],
	    float window_hor_cos  [2*DTT_SIZE],
	    float window_hor_sin  [2*DTT_SIZE],
	    float window_vert_cos [2*DTT_SIZE],
		float window_vert_sin [2*DTT_SIZE],
		int                woi_width,
		int                woi_height,
		int                kernels_hor,
		int                kernels_vert,
		int                tilesx)
{
#ifdef DEBUG30
	int dbg_tile = (num_colors & 16) != 0;
#endif
	num_colors &= 7;
//	int tilesx = TILES-X;
	int is_mono = num_colors == 1;
	// TODO: pass these values instead of constants to handle EO/LWIR
	int max_px =   woi_width - 1; // IMG-WIDTH  - 1; // odd
	int max_py =   woi_height - 1; // IMG-HEIGHT - 1; // odd
	int max_pxm1 = max_px - 1; // even
	int max_pym1 = max_py - 1; // even
	int max_kernel_hor = kernels_hor - 1; // KERNELS_HOR -1;
	int max_kernel_vert = kernels_vert - 1; // KERNELS_VERT-1;

	int   ktileX, ktileY;
	int   kernel_index; // common for all coors
	float kdx, kdy;
	if (threadIdx.x == 0){
//		ktileX = min(max_kernel_hor,  max(0, ((int) lrintf(centerX * (1.0/KERNELS_STEP)+1))));
//		ktileY = min(max_kernel_vert, max(0, ((int) lrintf(centerY * (1.0/KERNELS_STEP)+1))));
//		kdx =    centerX - (ktileX << KERNELS_LSTEP) + (1 << (KERNELS_LSTEP -1)); // difference in pixel
//		kdy =    centerY - (ktileY << KERNELS_LSTEP) + (1 << (KERNELS_LSTEP -1)); // difference in pixel

		// From ImageDttCPU.java: extract_correct_tile() (modified 2022/05/12):
//		int kernel_pitch = width/(clt_kernels[chn_kernel][0].length - 2);
			// 1. find closest kernel
//		ktileX = (int) Math.round(centerX/kernel_pitch) + 1;
//		ktileY = (int) Math.round(centerY/kernel_pitch) + 1;
//		if      (ktileY < 0)                                ktileY = 0;
//		else if (ktileY >= clt_kernels[chn_kernel].length)  ktileY = clt_kernels[chn_kernel].length-1;
//		if      (ktileX < 0)                                ktileX = 0;
//		else if (ktileX >= clt_kernels[chn_kernel][ktileY].length) ktileX = clt_kernels[chn_kernel][ktileY].length-1;
    	// extract center offset data stored with each kernel tile
//		CltExtra ce = new CltExtra (clt_kernels[chn_kernel][ktileY][ktileX][4]);
		// 2. calculate correction for center of the kernel offset
//		double kdx = centerX - (ktileX -1 +0.5) *  kernel_pitch; // difference in pixel
//		double kdy = centerY - (ktileY -1 +0.5) *  kernel_pitch;
		int kernel_pitch = woi_width / (kernels_hor - 2);
		ktileX = min(max_kernel_hor,  max(0, ((int) lrintf(centerX /kernel_pitch + 1))));
		ktileY = min(max_kernel_vert, max(0, ((int) lrintf(centerY /kernel_pitch + 1))));
		kdx =    centerX - (ktileX - 0.5) * kernel_pitch; // difference in pixel
		kdy =    centerY - (ktileY - 0.5) * kernel_pitch; //

		kernel_index = (ktileX + ktileY * kernels_hor) * num_colors;
	}
    // broadcast kernel_index
    kernel_index =  __shfl_sync(
    		0xffffffff,        // unsigned mask,
			kernel_index,      // T var,
			0,                 // int srcLane,
			THREADS_PER_TILE); // int width=warpSize);
    kdx =  __shfl_sync(
    		0xffffffff,        // unsigned mask,
			kdx,               // T var,
			0,                 // int srcLane,
			THREADS_PER_TILE); // int width=warpSize);
    kdy =  __shfl_sync(
    		0xffffffff,        // unsigned mask,
			kdy,               // T var,
			0,                 // int srcLane,
			THREADS_PER_TILE); // int width=warpSize);

     __syncthreads();// __syncwarp(); // is it needed?
    float px, py;
    // copy kernel
	int kernel_full_index = kernel_index + color;
    float * kernel_src = gpu_kernels + kernel_full_index* (DTT_SIZE * DTT_SIZE * 4);
    float * kernelp =  clt_kernels;

    kernel_src += threadIdx.x; // lsb;
    kernelp +=    threadIdx.x; // lsb;
#pragma unroll
    for (int j = 0; j < DTT_SIZE * 4; j++){ // all 4 components, 8 rows
    	// shared memory kernels use DTT_SIZE1 (same as image data)
    	*kernelp = *kernel_src;
    	kernelp+=DTT_SIZE1;
    	kernel_src+=THREADSX;
    }

    // Calculate offsets and prepare windows (all colors):
	struct CltExtra * clt_extra = &gpu_kernel_offsets[kernel_full_index];

	px = centerX - DTT_SIZE - (clt_extra->data_x + clt_extra->dxc_dx * kdx + clt_extra->dxc_dy * kdy) ; // fractional left corner
	int itlx = (int) floorf(px +0.5f);
	int_topleft [0] = itlx;
	float shift_hor =  itlx - px;
	residual_shift[0] = shift_hor;
	float x = shift_hor *(1.0f/16);
	float ahc = cospif(x);
	float ahs = sinpif(x);
	int i1 = DTT_SIZE;
	int i = 0;
	// embed sign for cosine and sine branches into window coefficients
#pragma unroll
	for (; i < (DTT_SIZE/2); i++ ){
		int ri = (DTT_SIZE-1) - i;
		window_hor_cos[i] =   HWINDOW[i]*ahc + HWINDOW[ri]*ahs;
		window_hor_cos[i1] =  HWINDOW[i]*ahs - HWINDOW[ri]*ahc;
		if (is_mono || (color == BAYER_GREEN)){
			window_hor_sin[i] =   HWINDOW[i]*ahc + HWINDOW[ri]*ahs; // bayer_color== 2
			window_hor_sin[i1] = -HWINDOW[i]*ahs + HWINDOW[ri]*ahc;
		}
		i1++;
	}
	// embed sign for cosine and sine branches into window coefficients
#pragma unroll
	for (; i < DTT_SIZE; i++ ){
		int ri = (DTT_SIZE-1) - i;
		window_hor_cos[i] =   -HWINDOW[i]*ahc - HWINDOW[ri]*ahs;
		window_hor_cos[i1] =   HWINDOW[i]*ahs - HWINDOW[ri]*ahc;
		if (is_mono || (color == BAYER_GREEN)){
			window_hor_sin[i] =    HWINDOW[i]*ahc + HWINDOW[ri]*ahs;
			window_hor_sin[i1] =   HWINDOW[i]*ahs - HWINDOW[ri]*ahc;
		}
		i1++;
	}

	py = centerY - DTT_SIZE - (clt_extra->data_y + clt_extra->dyc_dx * kdx + clt_extra->dyc_dy * kdy) ; // fractional top corner
	int itly = (int) floorf(py +0.5f);
	int_topleft[1] = itly;

#ifdef DEBUG_OOB11
#ifdef IMG_WIDTH
#ifdef IMG_HEIGHT
	if ((int_topleft[0] < 0) || (int_topleft[1] < 0) || (int_topleft[0] >= (IMG_WIDTH - DTT_SIZE)) || (int_topleft[1] >= IMG_HEIGHT - DTT_SIZE)){
		printf("Source data OOB, left=%d, top=%d\n",int_topleft[0],int_topleft[1]);
		printf("\n");
		printf("\n");
	    __syncthreads();// __syncwarp();
	}
#endif // IMG_HEIGHT
#endif // IMG_WIDTH
#endif // DEBUG_OOB1


	float shift_vert =  itly - py;
	residual_shift[1] = shift_vert;
	x = shift_vert *(1.0f/16);
	float avc = cospif(x);
	float avs = sinpif(x);
	i1 = DTT_SIZE; //**** Was commented out
	// embed sign for cosine branch only into window coefficients (for R,B only CC is needed, for G - CC and SC
	i = 0;
#pragma unroll
	for (; i < DTT_SIZE/2; i++ ){
		int ri = (DTT_SIZE-1) - i;
		window_vert_cos[i] =        HWINDOW[i]*avc + HWINDOW[ri]*avs;
		window_vert_cos[i1] =       HWINDOW[i]*avs - HWINDOW[ri]*avc;
		if (is_mono){
			window_vert_sin[i] =    HWINDOW[i]*avc + HWINDOW[ri]*avs;
			window_vert_sin[i1] =  -HWINDOW[i]*avs + HWINDOW[ri]*avc;
		}
		i1++;

	}
#pragma unroll
	for (; i < DTT_SIZE; i++ ){
		int ri = (DTT_SIZE-1) - i;
		window_vert_cos[i] =       -HWINDOW[i]*avc - HWINDOW[ri]*avs;
		window_vert_cos[i1] =       HWINDOW[i]*avs - HWINDOW[ri]*avc;
		if (is_mono){
			window_vert_sin[i] =    HWINDOW[i]*avc + HWINDOW[ri]*avs;
			window_vert_sin[i1] =   HWINDOW[i]*avs - HWINDOW[ri]*avc;
		}
		i1++;

	}
     __syncthreads();// __syncwarp();
#ifdef DEBUG30
    if (dbg_tile && (threadIdx.x) == 0){
		printf("COLOR=%d\n",color);
		printf("centerX=%f,    centerY=%f\n",centerX, centerY);
		printf("ktileX=%d,     ktileY=%d\n", ktileX,  ktileY);
		printf("kdx=%f,        kdy=%f\n",    kdx, kdy);
		printf("int_topleft[%d][0]=%d,    int_topleft[%d][1]=%d\n",i,int_topleft[0],i,int_topleft[1]);
		printf("residual_shift[%d][0]=%f, residual_shift[%d][1]=%f\n",i,residual_shift[0],i,residual_shift[1]);
		printf("\nwindow_hor_cos\n");
		for (int ii = 0; ii < 2*DTT_SIZE; ii++){
			printf("%6f, ",window_hor_cos[ii]);
		}
		printf("\nwindow_hor_sin\n");
		for (int ii = 0; ii < 2*DTT_SIZE; ii++){
			printf("%6f, ",window_hor_sin[ii]);
		}
		printf("\nwindow_vert_cos\n");
		for (int ii = 0; ii < 2*DTT_SIZE; ii++){
			printf("%6f, ",window_vert_cos[ii]);
		}
		printf("\nwindow_vert_sin\n");
		for (int ii = 0; ii < 2*DTT_SIZE; ii++){
			printf("%6f, ",window_vert_sin[ii]);
		}
    }
     __syncthreads();// __syncwarp();
#endif



    // prepare, fold and write data to DTT buffers
    int dstride2 = dstride << 1; // in floats (pixels)
    if (is_mono) {
    	// clear 4 buffer of the CLT tile
    	float *dtt_buf =  clt_tile + threadIdx.x;
#pragma unroll
    	for (int i = 0; i < 4* DTT_SIZE; i++) {
    		(*dtt_buf)  = 0.0f;
    		dtt_buf +=    DTT_SIZE1;
    	}
    	__syncthreads();// __syncwarp();

        for (int npass = 0; npass < 4; npass++) { // 4 passes to reuse Bayer tables
//          for (int npass = 1; npass < 2; npass++) { // 4 passes to reuse Bayer tables
        	int col_tl = int_topleft[0]; //  + (threadIdx.x << 1);
        	int row_tl = int_topleft[1];
        	// for red, blue and green, pass 0
        	int local_col = (npass & 1)  + (threadIdx.x << 1);
        	int local_row = (npass >> 1) & 1;
        	float hwind_cos = window_hor_cos [local_col];
        	float hwind_sin = window_hor_sin [local_col];

        	int dtt_offset =     fold_indx2[local_row][local_col];
        	int dtt_offset_inc = fold_inc[local_row];

        	float *dcct_buf = clt_tile + (0 * (DTT_SIZE * DTT_SIZE1)); // CC buffer
        	float *dsct_buf = clt_tile + (1 * (DTT_SIZE * DTT_SIZE1)); // SC buffer
        	float *dcst_buf = clt_tile + (2 * (DTT_SIZE * DTT_SIZE1)); // CS buffer
        	float *dsst_buf = clt_tile + (3 * (DTT_SIZE * DTT_SIZE1)); // SS buffer

        	// replace pixels outside input window
        	int col_src = col_tl + local_col;
        	if (col_src < 0) {
        		col_src = 0;
        	} else if (col_src > max_px){
        		col_src = max_px;
        	}
        	int row_src = row_tl + local_row;
        	int row_use = row_src;
        	if (row_use < 0) {
        		row_use = 0;
        	} else if (row_use > max_py){
        		row_use = max_py;
        	}
//            __syncthreads();// ?

    		float *image_p = gpu_images + dstride * row_use + col_src;
    #pragma unroll
    		for (int i = 0; i < 8; i++) {
    			float vwind_cos = window_vert_cos[local_row];
    			float vwind_sin = window_vert_sin[local_row];
    			float d = (*image_p);
    			int dtt_offset1 = dtt_offset + (dtt_offset >> 3); // converting for 9-long rows (DTT_SIZE1)
    			dcct_buf[dtt_offset1] += d * hwind_cos * vwind_cos;
    			dsct_buf[dtt_offset1] += d * hwind_sin * vwind_cos;
    			dcst_buf[dtt_offset1] += d * hwind_cos * vwind_sin;
    			dsst_buf[dtt_offset1] += d * hwind_sin * vwind_sin;
    			dtt_offset = ( dtt_offset + ((dtt_offset_inc & 0xf) << 3)) & 0x3f;
    			dtt_offset_inc >>= 4;
    			local_row += 2;
    			row_src +=2;
    			if ((row_src >= 0) && (row_src <= max_pym1)){
    				image_p += dstride2;
    			}
    		}
        	__syncthreads();// __syncwarp();
#ifdef DEBUG30
        if (dbg_tile && (threadIdx.x) == 0){
        	printf("\nFOLDing DTT Tiles,mono, npass=%d\n",npass);
        	debug_print_clt1(clt_tile, color, 0xf);
        }
        __syncthreads();// __syncwarp();
#endif

        }
//        __syncthreads();// __syncwarp();
        // no need to clone calculated tile so each will be processed to CC, SC, CS, and SS in-place, will be done in dttiv_mono_2d(clt_tile);
#ifdef DEBUG30
        if (dbg_tile && (threadIdx.x) == 0){
        	printf("\nFOLDED DTT Tiles,mono\n");
        	debug_print_clt1(clt_tile, color, 0xf);
        }
        __syncthreads();// __syncwarp();
#endif

#ifdef DEBUG2
        if ((threadIdx.x) == 0){
        	printf("\nFOLDED DTT Tiles,mono\n");
        	debug_print_clt1(clt_tile, color, (color== BAYER_GREEN)?3:1); // only 1 quadrant for R,B and 2 - for G
        }
        __syncthreads();// __syncwarp();
#endif

        dttiv_mono_2d(clt_tile);

#ifdef DEBUG30
        if (dbg_tile && (threadIdx.x) == 0){
        	printf("\nDTT Tiles after vertical pass (both passes), mono\n");
        	debug_print_clt1(clt_tile, color, 0xf); // only 1 quadrant for R,B and 2 - for G
        }
        __syncthreads();// __syncwarp();
#endif

    } else { // if (is_mono) {
    	float *dtt_buf =  clt_tile + threadIdx.x;
#pragma unroll
    	for (int i = 0; i < 4*DTT_SIZE; i++) {
    		(*dtt_buf)  = 0.0f;
    		dtt_buf +=    DTT_SIZE1;
    	}
    	__syncthreads();// __syncwarp();
    	int color0 = color & 1;
    	int color1 = (color >>1) & 1;

    	for (int gpass = 0; gpass < (color1 + 1); gpass++) { // Only once for R, B, twice - for G
    		int col_tl = int_topleft[0]; //  + (threadIdx.x << 1);
    		int row_tl = int_topleft[1];
    		// for red, blue and green, pass 0
    		int local_col = ((col_tl & 1) ^ (BAYER_RED_COL ^ color0 ^ color1 ^ gpass)) + (threadIdx.x << 1); // green red row: invert column from red
    		int local_row = ((row_tl & 1) ^ BAYER_RED_ROW ^ color0          ^ gpass);                            // use red row
#ifdef DEBUG30
        if (dbg_tile && (threadIdx.x) == 0){
        	printf("\ngpass= %d, local_row=%d, local_col=%d\n",gpass, local_row, local_col);
        }
#endif
        __syncthreads();// __syncwarp();

    		float hwind_cos = window_hor_cos[local_col];
    		float hwind_sin = window_hor_sin[local_col]; // **** only used for green

    		int dtt_offset =     fold_indx2[local_row][local_col];
    		int dtt_offset_inc = fold_inc[local_row];
    		float *dct_buf = clt_tile + ((gpass << 1) * (DTT_SIZE * DTT_SIZE1));
    		float *dst_buf = clt_tile + (((gpass << 1) + 1) * (DTT_SIZE * DTT_SIZE1));   // **** only used for green

    		// replace pixels outside input window
    		int col_src = col_tl + local_col;
    		if (col_src < 0) {
    			col_src &= 1; // same Bayer
    		} else if (col_src > max_px){
    			col_src = (col_src & 1) + max_pxm1;
    		}
    		int row_src = row_tl + local_row;
    		int row_use = row_src;
    		if (row_use < 0) {
    			row_use &= 1; // same Bayer
    		} else if (row_use > max_py){
    			row_use = (row_use & 1) + max_pym1;
    		}
//           __syncthreads();// worse

    		//		float *image_p = gpu_images + dstride * (row_tl + local_row)+ col_tl + local_col;
    		float *image_p = gpu_images + dstride * row_use + col_src;
#pragma unroll
    		for (int i = 0; i < 8; i++) {
    			float d = (*image_p);
    			d *= window_vert_cos[local_row]; //warp illegal address (0,2,1)

    			int dtt_offset1 = dtt_offset + (dtt_offset >> 3); // converting for 9-long rows (DTT_SIZE1)
    			dct_buf[dtt_offset1] = d * hwind_cos;
    			dst_buf[dtt_offset1] = d * hwind_sin; // **** only used for green
    			dtt_offset = ( dtt_offset + ((dtt_offset_inc & 0xf) << 3)) & 0x3f;
    			dtt_offset_inc >>= 4;
    			local_row += 2;
    			row_src +=2;
    			if ((row_src >= 0) && (row_src <= max_pym1)){
    				image_p += dstride2;
    			}
    		}
//        	__syncthreads();// __syncwarp();
    	}
    	__syncthreads();// __syncwarp();
#ifdef DEBUG30
        if (dbg_tile && (threadIdx.x) == 0){
    		printf("\nFOLDED DTT Tiles Green before reduction\n");
    		debug_print_clt1(clt_tile, color, 0xf); // all quadrants for green only
    	}
    	__syncthreads();// __syncwarp();
#endif
    	if (color == BAYER_GREEN) {
    		// reduce 4 green DTT buffers into 2 (so free future rotated green that were borrowed)
    		float *dtt_buf =  clt_tile + threadIdx.x;
    		float *dtt_buf1 = dtt_buf+ (2 * DTT_SIZE1 * DTT_SIZE); // ((float *) clt_tile[2]) + threadIdx.x;
#pragma unroll
    		for (int i = 0; i < 2*DTT_SIZE; i++) {
    			(*dtt_buf) += (*dtt_buf1);
    			dtt_buf +=    DTT_SIZE1;
    			dtt_buf1 +=   DTT_SIZE1;
    		}
    		__syncthreads();// __syncwarp();
    	}

#ifdef DEBUG2
    	if ((threadIdx.x) == 0){
    		printf("\nFOLDED DTT Tiles,color=%d\n", color);
    		debug_print_clt1(clt_tile, color, (color== BAYER_GREEN)?3:1); // only 1 quadrant for R,B and 2 - for G
    	}
    	__syncthreads();// __syncwarp();
#endif
    	dttiv_color_2d(
    			clt_tile,
				color);
#ifdef DEBUG30
        if (dbg_tile && (threadIdx.x) == 0){
    		printf("\nDTT Tiles after vertical pass (both passes), color = %d\n",color);
        	debug_print_clt1(clt_tile, color, 0xf); // only 1 quadrant for R,B and 2 - for G
        }
        __syncthreads();// __syncwarp();
#endif

    	// Replicate DTT, so non-bayer can still use same in-place rotation code
    	float *src, *dst;
    	int   negate; // , dst_inc;

    	// Replicate horizontally (for R and B only):
    	if (color != BAYER_GREEN) {
    		negate = 1-(((int_topleft[0] & 1) ^ (BAYER_RED_COL ^ color)) << 1); // +1/-1
    		src = clt_tile + threadIdx.x; // &clt_tile[0][0][threadIdx.x    ];
    		dst = clt_tile + (DTT_SIZE1 * DTT_SIZE) + (threadIdx.x ^ 7); // &clt_tile[1][0][threadIdx.x ^ 7];
#pragma unroll
    		for (int i = 0; i < DTT_SIZE; i++){
    			*dst = negate*(*src);
    			src += DTT_SIZE1;
    			dst += DTT_SIZE1;
    		}
    		__syncthreads();// __syncwarp();
#ifdef DEBUG2
    		if ((threadIdx.x) == 0){
    			printf("\nDTT Tiles after first replicating, color = %d\n",color);
    			debug_print_clt1(clt_tile, color, 0x3);
    		}
    		__syncthreads();// __syncwarp();
#endif

    	}
    	// replicate all colors down diagonal
    	negate = 1-(((int_topleft[0] & 1) ^ (int_topleft[1] & 1) ^ (BAYER_RED_COL ^ BAYER_RED_ROW ^ (color >> 1))) << 1); // +1/-1 // 1 -
    	// CC -> SS
    	src = clt_tile + threadIdx.x; // &clt_tile[0][0][threadIdx.x    ];
    	dst = clt_tile + (DTT_SIZE1 * (DTT_SIZE * 3 + 7)) +  (threadIdx.x ^ 7); // &clt_tile[3][7][threadIdx.x ^ 7];
#pragma unroll
    	for (int i = 0; i < DTT_SIZE; i++){
    		*dst = negate*(*src);
    		src += DTT_SIZE1;
    		dst -= DTT_SIZE1;
    	}
    	//SC -> CS
    	src = clt_tile + (DTT_SIZE1 * DTT_SIZE) + threadIdx.x; // &clt_tile[1][0][threadIdx.x    ];
    	dst = clt_tile + (DTT_SIZE1 * (DTT_SIZE * 2 + 7)) +  (threadIdx.x ^ 7); // &clt_tile[2][7][threadIdx.x    ];
    	__syncthreads();// did not help
#pragma unroll
    	for (int i = 0; i < DTT_SIZE; i++){
    		*dst = negate*(*src);
    		src += DTT_SIZE1;
    		dst -= DTT_SIZE1;
    	}
#ifdef DEBUG30
        if (dbg_tile && (threadIdx.x) == 0){
    		printf("\nDTT Tiles after all replicating, color = %d\n",color);
    		debug_print_clt1(clt_tile, color, 0xf);
    	}
    	__syncthreads();// __syncwarp();
#endif



#ifdef DEBUG30
        if (dbg_tile && (threadIdx.x) == 0){
    		printf("\nKernel tiles to convolve, color = %d\n",color);
    		debug_print_clt1(clt_kernels, color, 0xf); // all colors, all quadrants
    	}
    	__syncthreads();// __syncwarp();
#endif

    	__syncthreads();// did not help

    } //  else { // if (is_mono) {

#ifdef DEBUG30
        if (dbg_tile && (threadIdx.x) == 0){
    		printf("\nDTT Tiles after before convolving, color = %d\n",color);
    		debug_print_clt1(clt_tile, color, 0xf);
    	}
    	__syncthreads();// __syncwarp();
#endif



    // convolve first, then rotate to match Java and make it easier to verify
    convolveTiles(
    		clt_tile,      // float clt_tile   [4][DTT_SIZE][DTT_SIZE1], // 4 quadrants of the clt data, rows extended to optimize shared ports
			clt_kernels); // float kernel     [4][DTT_SIZE][DTT_SIZE1]); // 4 quadrants of the CLT kernel (DTT3 converted)
     __syncthreads();// __syncwarp();
#ifdef DEBUG2
    if ((threadIdx.x) == 0){
        printf("\nDTT Tiles after convolution, color = %d\n",color);
    	debug_print_clt1(clt_tile, color, 0xf); // all colors, all quadrants
    }
     __syncthreads();// __syncwarp();
#endif

    // rotate phases: first horizontal, then vertical
    shiftTileHor(
    		clt_tile,           // float clt_tile       [4][DTT_SIZE][DTT_SIZE1], // +1 to alternate column ports
			residual_shift[0]); // float residual_shift);
     __syncthreads();// __syncwarp();
#ifdef DEBUG2
    if ((threadIdx.x) == 0){
        printf("\nDTT Tiles after horizontal shift, color = %d\n",color);
    	debug_print_clt1(clt_tile, color,  0xf); // only 1 quadrant for R,B and 2 - for G
    }
     __syncthreads();// __syncwarp();
#endif


    shiftTileVert(
    		clt_tile,           // float clt_tile       [4][DTT_SIZE][DTT_SIZE1], // +1 to alternate column ports
			residual_shift[1]); // float residual_shift);
     __syncthreads();// __syncwarp();
#ifdef DEBUG1
    if ((threadIdx.x) == 0){
        printf("\nDTT Tiles after vertical shift, color = %d\n",color);
    	debug_print_clt1(clt_tile, color,  0xf); // only 1 quadrant for R,B and 2 - for G
        printf("\nDTT All done\n");
    }
     __syncthreads();// __syncwarp();
#endif


#ifdef DBG_TILE
#ifdef DEBUG3
    if ((threadIdx.x) == 0){
        printf("\nDTT Tiles after vertical shift, color = %d\n",color);
    	debug_print_clt1(clt_tile, color,  0xf); // only 1 quadrant for R,B and 2 - for G
        printf("\nDTT All done\n");
    }
     __syncthreads();// __syncwarp();
#endif
#endif



     // optionally apply LF
     if ((lpf_mask >> color) & 1){
    	 float * clt = clt_tile + threadIdx.x;
#pragma unroll
    	 for (int q = 0; q < 4; q++) {
    		 float *lpf = lpf_data[color] + threadIdx.x;
    		 //(colors > 1)? color : 3 for mono - not yet implemented
#pragma unroll
    		 for (int i = 0; i <8; i++){
    			 (*clt) *= (*lpf);
    			 clt   += DTT_SIZE1;
    			 lpf   += DTT_SIZE;
    		 }
    	 }
    	 __syncthreads();// __syncwarp();
#ifdef DEBUG3
    	 if ((threadIdx.x) == 0){
    		 printf("\nDTT Tiles after LPF, color = %d\n",color);
    		 debug_print_clt1(clt_tile, color,  0xf); // only 1 quadrant for R,B and 2 - for G
    		 printf("\nDTT All done\n");
    	 }
    	 __syncthreads();// __syncwarp();
#endif
     }

     int offset_src = threadIdx.x;
//     int offset_dst = (((txy >> 16) * TILES-X + (txy & 0xfff))*NUM_COLORS + color)* ( 4 * DTT_SIZE * DTT_SIZE) + threadIdx.x;
     int offset_dst = (((txy >> 16) * tilesx + (txy & 0xfff))*num_colors + color)* ( 4 * DTT_SIZE * DTT_SIZE) + threadIdx.x;

     float * clt_src = clt_tile + offset_src; // threadIdx.x;
     float * clt_dst = gpu_clt +  offset_dst; // ((ty * TILES-X + tx)*NUM_COLORS + color)* ( 4 * DTT_SIZE * DTT_SIZE1) + threadIdx.x; // gpu_kernels + kernel_full_index* (DTT_SIZE * DTT_SIZE * 4);
//#ifndef NOICLT

#ifdef DEBUG3
    if ((threadIdx.x) == 0){
        printf("clt_src = 0x%lx\n",clt_src);
        printf("clt_dst = 0x%lx\n",clt_dst);
    }
#endif


    if (tscale == 0) { // just set w/o scaling
#pragma unroll
    	for (int j = 0; j < DTT_SIZE * 4; j++){ // all 4 components, 8 rows
    		// shared memory tiles use DTT_SIZE1
    		*clt_dst =  *clt_src;
    		clt_src   += DTT_SIZE1;
    		clt_dst   += DTT_SIZE;
    	}
    } else if (tscale > 0) { // positive - scale and set. For motion blur positive should be first
#pragma unroll
    	for (int j = 0; j < DTT_SIZE * 4; j++){ // all 4 components, 8 rows
    		// shared memory tiles use DTT_SIZE1
    		*clt_dst =  *clt_src * tscale;
    		clt_src   += DTT_SIZE1;
    		clt_dst   += DTT_SIZE;
    	}
    } else { // negative - scale and subtract from existing. For motion blur positive should be first
#pragma unroll
    	for (int j = 0; j < DTT_SIZE * 4; j++){ // all 4 components, 8 rows
    		// shared memory tiles use DTT_SIZE1
    		*clt_dst +=  *clt_src * tscale;
    		clt_src   += DTT_SIZE1;
    		clt_dst   += DTT_SIZE;
    	}
    }
    __syncthreads();// __syncwarp();
}



/**
 * Prepare for matching images to generate textures - measure difference in a noise-equivalent way,
 * relative to the shot noise at that intensity value. Do not use it for the images that are not shot-noise limited
 * Used in {generate_RBGA(), textures_nonoverlap()} -> textures_accumulate()
 *
 * @param rb_mode              color type: 0 - green, 1 - r/b
 * @param min_shot             shot noise minimal value (10.0)
 * @param scale_shot           scale shot noise (3.0)
 * @param mclt_src             mclt source tile (from inverse transform) [2* DTT_SIZE][DTT_SIZE1+ DTT_SIZE]
 * @param mclt_dst             mclt destination tile (from inverse transform) [2* DTT_SIZE][DTT_SIZE1+ DTT_SIZE]
 * @param mclt_tmp             mclt tmp tile [2* DTT_SIZE][DTT_SIZE1+ DTT_SIZE]
 * @param debug                debug if != 0
 */
__device__ void debayer_shot(
		const int rb_mode,    // 0 - green, 1 - r/b
		float     min_shot,   // 10.0
		float     scale_shot, // 3.0 (0.0 for mono)
		float   * mclt_src,   // [2* DTT_SIZE][DTT_SIZE1+ DTT_SIZE], // +1 to alternate column ports[16][17]
		float   * mclt_dst,   // [2* DTT_SIZE][DTT_SIZE1+ DTT_SIZE], // +1 to alternate column ports[16][17]
		float   * mclt_tmp,
		int       debug)
{
	// unapply squared window
#pragma unroll
	for (int n = 0; n < 2; n++){
		int col = threadIdx.x;
		if (n) col ^= 0xf;
		float wx = HWINDOW_SQi[threadIdx.x];
		float * msp = mclt_src + col;
		float * mtp = mclt_tmp + col;

#pragma unroll
		for (int row = 0; row < DTT_SIZE2; row++){
			int row0 = row;
			if 	(row >= DTT_SIZE) row0 ^= 0xf;
			*mtp = *msp * wx * HWINDOW_SQi[row0];
			mtp += DTT_SIZE21;
			msp += DTT_SIZE21;
		}
	}
	__syncthreads();

#ifdef DEBUG7
	for (int ccam = 0; ccam < NUM_CAMS; ccam++) {
		if (debug  && (threadIdx.x == 0) && (threadIdx.y == ccam)){
			printf("\ndebayer_shot HWINDOW_SQi applied, camera = %d\n",threadIdx.y);
			debug_print_mclt(
					mclt_tmp, //         [4][DTT_SIZE][DTT_SIZE1], // +1 to alternate column ports)
					-1);
		}
		__syncthreads();// __syncwarp();
	}
	__syncthreads();// __syncwarp();
#endif

	// debayer
	debayer(rb_mode,   // 0 - green, 1 - r/b
			mclt_tmp,  // [2* DTT_SIZE][DTT_SIZE1+ DTT_SIZE], // +1 to alternate column ports[16][17]
			mclt_dst,  // [2* DTT_SIZE][DTT_SIZE1+ DTT_SIZE], // +1 to alternate column ports[16][17]
			debug);
#ifdef DEBUG7
	for (int ccam = 0; ccam < NUM_CAMS; ccam++) {
		if (debug  && (threadIdx.x == 0) && (threadIdx.y == ccam)){
			printf("\ndebayer_shot debayer() applied, camera = %d\n",threadIdx.y);
			debug_print_mclt(
					mclt_dst, //         [4][DTT_SIZE][DTT_SIZE1], // +1 to alternate column ports)
					-1);
		}
		__syncthreads();// __syncwarp();
	}
	__syncthreads();// __syncwarp();
#endif


	if (scale_shot > 0.0) {

#ifdef FASTMATH
		float k = __frsqrt_rn(min_shot);
#else
		float k = rsqrtf(min_shot);
#endif

		// double k = 1.0/Math.sqrt(min_shot); //sqrtf
		//for (int i = 0; i < tile.length; i++) tile_db[i] = scale_shot* ((tile_db[i] > min_shot)? Math.sqrt(tile_db[i]) : (k*tile_db[i]));
		float *mcltp = mclt_dst + threadIdx.x;
#pragma unroll
		for (int row = 0; row < DTT_SIZE2; row++){
#pragma unroll
			for (int col = 0; col < DTT_SIZE2; col += DTT_SIZE){
				float d = *mcltp;
#ifdef FASTMATH
				*mcltp = scale_shot * (( d > min_shot)? __fsqrt_rn(d) : (k * d));
#else
				*mcltp = scale_shot * (( d > min_shot)? sqrtf(d) : (k * d));
#endif



				mcltp += DTT_SIZE;
			}
			mcltp += (DTT_SIZE21-DTT_SIZE2);
		}
#ifdef DEBUG7
		for (int ccam = 0; ccam < NUM_CAMS; ccam++) {
			if (debug  && (threadIdx.x == 0) && (threadIdx.y == ccam)){
				printf("\ndebayer_shot sqrt applied, camera = %d, scale_shot = %f, min_shot = %f, k= %f\n",threadIdx.y, scale_shot, min_shot, k);
				debug_print_mclt(
						mclt_dst, //         [4][DTT_SIZE][DTT_SIZE1], // +1 to alternate column ports)
						-1);
			}
			__syncthreads();// __syncwarp();
		}
		__syncthreads();// __syncwarp();
#endif

	}
	// apply squared window back
#pragma unroll
	for (int n = 0; n < 2; n++){
		int col = threadIdx.x;
		if (n) col ^= 0xf;
		float wx = HWINDOW_SQ[threadIdx.x];
		float * mdp = mclt_dst + col;

#pragma unroll
		for (int row = 0; row < DTT_SIZE2; row++){
			int row0 = row;
			if 	(row >= DTT_SIZE) row0 ^= 0xf;
			*mdp *= wx * HWINDOW_SQ[row0];
			mdp += DTT_SIZE21;
		}
	}
	__syncthreads();

#ifdef DEBUG7
	for (int ccam = 0; ccam < NUM_CAMS; ccam++) {
		if (debug  && (threadIdx.x == 0) && (threadIdx.y == ccam)){
			printf("\ndebayer_shot HWINDOW2 applied, camera = %d \n",threadIdx.y);
			debug_print_mclt(
					mclt_dst, //         [4][DTT_SIZE][DTT_SIZE1], // +1 to alternate column ports)
					-1);
		}
		__syncthreads();// __syncwarp();
	}
	__syncthreads();// __syncwarp();
#endif



}

/**
 * Simple de-Bayer LPF - convolution with color-variant 3x3 kernels. Input is RGB, not Bayer
 * relative to the shot noise at that intensity value. Do not use it for the images that are not shot-noise limited
 * Used in {generate_RBGA(), textures_nonoverlap()} -> textures_accumulate() -> debayer_shot()
 *
 * @param rb_mode              color type: 0 - green, 1 - r/b
 * @param mclt_src             mclt source tile (from inverse transform) [2* DTT_SIZE][DTT_SIZE1+ DTT_SIZE]
 * @param mclt_dst             mclt destination tile (from inverse transform) [2* DTT_SIZE][DTT_SIZE1+ DTT_SIZE]
 * @param debug                debug if != 0
 */
__device__ void debayer( // 8 threads
		const int rb_mode,   // 0 - green, 1 - r/b
		float * mclt_src,  // [2* DTT_SIZE][DTT_SIZE1+ DTT_SIZE], // +1 to alternate column ports[16][17]
		float * mclt_dst,  // [2* DTT_SIZE][DTT_SIZE1+ DTT_SIZE], // +1 to alternate column ports[16][17]
		int debug)
{

#pragma unroll
	for (int n = 0; n < 25; n++){
		int row, col;
		if (n < 14) {
			row = (n+1);
			col = 1 + threadIdx.x;
		} else if (n < 21){
			row = 2 * (n-14) + 1 + (threadIdx.x >> 2);
			col = (1 + DTT_SIZE) + (threadIdx.x & 3);
		} else {
			row = 4 * (n - 21) + 1 + (threadIdx.x >> 1);
			col = (1 + DTT_SIZE + DTT_SIZE/2) + (threadIdx.x & 1);
		}
		if (row >= DTT_SIZE2M1) { // 17*15 - last (unused row
			continue;
		}
		int indx = DTT_SIZE21 * row + col;
		float * msp = mclt_src + indx;
		float * mdp = mclt_dst + indx;

		if (rb_mode){ // red and blue, all threads simultaneously
			*mdp = 0.0625 * (*(msp - (DTT_SIZE21 + 1)) + *(msp - (DTT_SIZE21 - 1)) + *(msp + (DTT_SIZE21 - 1)) + *(msp + (DTT_SIZE21 + 1)))+
					 0.125 *(*(msp - DTT_SIZE21) + *(msp + DTT_SIZE21) + *(msp - 1) + *(msp + 1))+
					 0.25 * *msp;
		} else { // green, all threads simultaneously
			*mdp = 0.125 *(*(msp - DTT_SIZE21) + *(msp + DTT_SIZE21) + *(msp - 1) + *(msp + 1))+
					 0.5 * *msp;
		}
	}
	// fill the remaining with non-dbayered
	int offs = threadIdx.x;
	*(mclt_dst + offs) = *(mclt_src + offs);
	offs     += DTT_SIZE;
	*(mclt_dst + offs) = *(mclt_src + offs);
	offs     = (DTT_SIZE21 * DTT_SIZE2M1) + threadIdx.x;
	*(mclt_dst + offs) = *(mclt_src + offs);
	offs     += DTT_SIZE;
	*(mclt_dst + offs) = *(mclt_src + offs);
	offs = threadIdx.x * DTT_SIZE21; // 2 corners will repeat
	*(mclt_dst + offs) = *(mclt_src + offs);
	offs     += DTT_SIZE * DTT_SIZE21;
	*(mclt_dst + offs) = *(mclt_src + offs);
	offs = threadIdx.x * DTT_SIZE21 + DTT_SIZE2M1; // 2 corners will repeat
	*(mclt_dst + offs) = *(mclt_src + offs);
	offs     += DTT_SIZE * DTT_SIZE21;
	*(mclt_dst + offs) = *(mclt_src + offs);
}


/**
 * Combines multi-camera rgba tiles
 * Used in {generate_RBGA(), textures_nonoverlap()} -> textures_accumulate()
 *
 * @param num_cams             number of cameras used <= NUM_CAMS
 * @param colors               number of colors used:  3 for RGB or 1 for monochrome
 * @param mclt_tile            tile after debayer (shared memory, has gaps to align with union !)
 * @param rbg_tile             if not null (usually) - original (not-debayered) rbg tile to use for the output
 * @param rgba                 result
 * @param calc_extra           calculate ports_rgb, max_diff. If not null - will ignore rbg_tile, so this mode
 *                             should not be combined with texture generation. It is intended to generate a
 *                             lo-res (1/8) images for macro correlation
 * @param ports_rgb_shared     shared memory data to be used to return lo-res images tile average color [NUM_COLORS * NUM_CAMS]
 * @param max_diff_shared      shared memory data to be used to return lo-res images tile mismatch from average [NUM_CAMS]
 * @param max_diff_tmp         shared memory to be used here for temporary storage [NUM_CAMS * TEXTURE_THREADS_PER_TILE]
 * @param ports_rgb_tmp        shared memory to be used here for temporary storage [NUM_COLORS *NUM_CAMS * TEXTURE_THREADS_PER_TILE], [4*3][8]
 * @param port_offsets         [port]{x_off, y_off} - just to scale pixel value differences (quad - {{-0.5, -0.5},{0.5,-0.5},{-0.5,0.5},{0.5,0.5}}
 * @param diff_sigma           pixel value/pixel change (1.5)
 * @param diff_threshold       pixel value/pixel change (10)
 * @param min_agree            minimal number of channels to agree on a point (real number to work with fuzzy averages) (3.0)
 * @param weights              scales for R,B,G {0.294118, 0.117647, 0.588235}
 * @param dust_remove          do not reduce average weight when only one image differs much from the average (true)
 * @param keep_weights         return channel weights after A in RGBA (weight are always calculated, not so for the crms)
 * @param debug                debug if != 0
 */
//DTT_SIZE21
__device__ void tile_combine_rgba(
		int     num_cams,      // actual number of cameras
		int     colors,        // number of colors
		float * mclt_tile,     // debayer // has gaps to align with union !
		float * rbg_tile,      // if not null - original (not-debayered) rbg tile to use for the output
		float * rgba,          // result
		int     calc_extra,    // 1 - calculate ports_rgb, max_diff (if not null - will ignore rbg_tile !)

///		float ports_rgb_shared [NUM_COLORS][NUM_CAMS], // return to system memory (optionally pass null to skip calculation)
///		float max_diff_shared  [NUM_CAMS], // return to system memory (optionally pass null to skip calculation)
///		float max_diff_tmp     [NUM_CAMS][TEXTURE_THREADS_PER_TILE],
///		float ports_rgb_tmp    [NUM_COLORS][NUM_CAMS][TEXTURE_THREADS_PER_TILE], // [4*3][8]

		float * ports_rgb_shared, //  [NUM_COLORS][NUM_CAMS], // return to system memory (optionally pass null to skip calculation)
		float * max_diff_shared,  //   [NUM_CAMS], // return to system memory (optionally pass null to skip calculation)
		float * max_diff_tmp,     //      [NUM_CAMS][TEXTURE_THREADS_PER_TILE],
		float * ports_rgb_tmp,  //     [NUM_COLORS][NUM_CAMS][TEXTURE_THREADS_PER_TILE], // [4*3][8]

		float * port_offsets,  // [port]{x_off, y_off} - just to scale pixel value differences
		//		int           port_mask,      // which port to use, 0xf - all 4 (will modify as local variable)
		float   diff_sigma,    // pixel value/pixel change
		float   diff_threshold,// pixel value/pixel change - never used
		// next not used
		//		boolean       diff_gauss,     // when averaging images, use gaussian around average as weight (false - sharp all/nothing)
		float   min_agree,     // minimal number of channels to agree on a point (real number to work with fuzzy averages)
		float * chn_weights,   // color channel weights, sum == 1.0
		int     dust_remove,   // Do not reduce average weight when only one image differs much from the average
		int     keep_weights,  // return channel weights and rms after A in RGBA (weight are always calculated, not so for the crms)
		int     debug)
{
	float * alpha =        rgba + (colors * (DTT_SIZE2*DTT_SIZE21));
	float * port_weights = alpha + (DTT_SIZE2*DTT_SIZE21);
	float * crms =         port_weights + num_cams*(DTT_SIZE2*DTT_SIZE21); // calculated only if keep_weights
	float  threshold2 = diff_sigma * diff_threshold; // never used?
	threshold2 *= threshold2; // squared to compare with diff^2
	// Using NUM_CAMS as it is >= needed num_cams
	float  pair_dist2r [NUM_CAMS*(NUM_CAMS-1)/2]; // new double [ports*(ports-1)/2]; // reversed squared distance between images - to be used with gaussian. Can be calculated once !
	int    pair_ports[NUM_CAMS*(NUM_CAMS-1)/2][2];  // int [][]  pair_ports = new int [ports*(ports-1)/2][2];
	int    indx = 0;
	float  ksigma = 1.0/(2.0*diff_sigma*diff_sigma); // multiply by a weighted sum of squares of the differences
#ifdef DEBUG9
	__shared__ int dbg_bestPort1 [DTT_SIZE2*DTT_SIZE21];
	__shared__ int dbg_bestPort2 [DTT_SIZE2*DTT_SIZE21];
#endif // #ifdef DEBUG9

	for (int i = 0; i < num_cams; i++) { // if ((port_mask & ( 1 << i)) != 0){
		for (int j = i+1; j < num_cams; j++) { //   if ((port_mask & ( 1 << j)) != 0){
			//				double dx = port_offsets[j][0] - port_offsets[i][0];
			//				double dy = port_offsets[j][1] - port_offsets[i][1];
			float dx = *(port_offsets + 2 * j) -     *(port_offsets + 2 * i);
			float dy = *(port_offsets + 2 * j + 1) - *(port_offsets + 2 * i + 1);
			pair_ports[indx][0] = i;
			pair_ports[indx][1] = j;
			pair_dist2r[indx++] = ksigma / (dx*dx+dy*dy); // 2*sigma^2 * r^2
		}
	}
	int colors_offset = colors * MCLT_UNION_LEN; // padded in union !

#ifdef DEBUG7A
	__syncthreads();// __syncwarp();
	if (debug  && (threadIdx.x == 0) && (threadIdx.y == 0)){
		printf("diff_sigma = %f\n",    diff_sigma);
		printf("diff_threshold = %f\n",diff_threshold);
		printf("min_agree = %f\n",     min_agree);
		printf("chn_weights[0] = %f\n",chn_weights[0]);
		printf("chn_weights[1] = %f\n",chn_weights[1]);
		printf("chn_weights[2] = %f\n",chn_weights[2]);
		printf("dust_remove =    %d\n",dust_remove);
		printf("keep_weights =   %d\n",keep_weights);

		printf("\ntile_combine_rgba ksigma = %f\n",ksigma);
		for (int i = 0; i < indx; i++) {
			printf("%02d: %d :%d %f\n",i,pair_ports[i][0], pair_ports[i][1], pair_dist2r[i]);
		}
	}
	__syncthreads();// __syncwarp();
	if (debug  && (threadIdx.x == 0) && (threadIdx.y == 0)){
		for (int ccam = 0; ccam < num_cams; ccam++) { // if ((port_mask & ( 1 << i)) != 0){
			for (int nncol = 0; nncol < colors; nncol++){
				printf("\ntile_combine_rgba cam = %d, color = %d\n",ccam, nncol);
				debug_print_mclt(
						mclt_tile + ((nncol + colors * ccam) * MCLT_UNION_LEN),
						-1);
			}
		}
		printf("\ntile_combine_rgba break 1\n");
	}
	__syncthreads();// __syncwarp();

#endif

	for (int pass = 0; pass < 8; pass ++) {
		// below non-parametrized !
		int row = pass * 2 + (threadIdx.y >> 1);
		int col = ((threadIdx.y & 1) << 3) + threadIdx.x;
		int i = row * DTT_SIZE21 + col;
		float * crms_i = crms+i;
		float * port_weights_i = port_weights + i;
		float * mclt_tile_i = mclt_tile +  i; // has gaps to align in a union !
		float * alpha_i = alpha + i;
		if (keep_weights){
			float sw = 0.0;
			for (int ncol = 0; ncol < colors; ncol ++ ) { //  if (iclt_tile[0][ncol] != null){
				float s0 = 0, s1 = 0, s2 = 0;
				float * crms_col_i = crms_i + (DTT_SIZE2*DTT_SIZE21) * ncol;
				float * mclt_col_i = mclt_tile_i + MCLT_UNION_LEN * ncol;
				for (int cam = 0; cam < num_cams; cam++) { // if ((port_mask & ( 1 << ip)) != 0){
					s0 += 1.0;
					float d = * (mclt_col_i + colors_offset * cam);
					s1 += d;
					s2 += d * d;
				}
				float mse = (s0*s2 - s1*s1) / (s0 * s0);
#ifdef FASTMATH
				* crms_col_i = __fsqrt_rn(mse);
#else
				* crms_col_i = sqrtf(mse);
#endif

				sw += *(chn_weights +ncol) * mse;
			}
#ifdef FASTMATH
			*(crms_i + (DTT_SIZE2*DTT_SIZE21) * colors) = __fsqrt_rn(sw); // will fade as window
#else
			*(crms_i + (DTT_SIZE2*DTT_SIZE21) * colors) = sqrtf(sw); // will fade as window
#endif
		}
#ifdef DEBUG9
	}

#ifdef DEBUG8
	if (debug  && (threadIdx.x == 0) && (threadIdx.y == 0)){
		for (int ncol = 0; ncol < colors; ncol++) {
			printf("\n+++++ crms[%d] +++++\n",ncol);
			debug_print_mclt(
					crms + (ncol * (DTT_SIZE2*DTT_SIZE21)),
					-1);
		}
		printf("\n+++++ cmrs_combo +++++\n");
		debug_print_mclt(
				crms + (colors * (DTT_SIZE2*DTT_SIZE21)), //
				-1);
	}
	__syncthreads();// __syncwarp();
#endif
	__syncthreads();// __syncwarp();
	for (int pass = 0; pass < 8; pass ++) {
		// below non-parametrized !
		int row = pass * 2 + (threadIdx.y >> 1);
		int col = ((threadIdx.y & 1) << 3) + threadIdx.x;
		int i = row * DTT_SIZE21 + col;
		float * crms_i = crms+i;
		float * port_weights_i = port_weights + i;
		float * mclt_tile_i = mclt_tile +  i; // has gaps to align in a union !
		float * alpha_i = alpha + i;
#endif // #ifdef DEBUG9

		for (int cam = 0; cam < num_cams; cam++) {
			*(port_weights_i + cam*(DTT_SIZE2*DTT_SIZE21)) = 0.0;
		}
		int row_sym = row ^ ((row & 8)? 0xf : 0);
		int col_sym = col ^ ((col & 8)? 0xf : 0);
		float wnd2 = HWINDOW_SQ[row_sym] * HWINDOW_SQ[col_sym];
		float wnd2_inv = 1.0/wnd2;

//#pragma unroll
		for (int ipair = 0; ipair < (num_cams*(num_cams-1)/2); ipair++){
			float d = 0;
//#pragma unroll // non-constant
			for (int ncol = 0; ncol < colors; ncol++) { // if (iclt_tile[0][ncol] != null){
				//					double dc = iclt_tile[pair_ports[ip][0]][ncol][i] - iclt_tile[pair_ports[ip][1]][ncol][i];
				float * mclt_col_i = mclt_tile_i + MCLT_UNION_LEN * ncol;
				float dc =
						*(mclt_col_i + colors_offset * pair_ports[ipair][0]) -
						*(mclt_col_i + colors_offset * pair_ports[ipair][1]);
				dc *= wnd2_inv; // to compensate fading near the edges
				d+= *(chn_weights + ncol) * dc * dc;
			}
#ifdef FASTMATH
			d = __expf(-pair_dist2r[ipair] * d) + (FAT_ZERO_WEIGHT); // 0.5 for exact match, lower for mismatch. Add this weight to both ports involved
#else
			d = expf(-pair_dist2r[ipair] * d) + (FAT_ZERO_WEIGHT); // 0.5 for exact match, lower for mismatch. Add this weight to both ports involved
#endif

			// Add weight to both channels in a pair
			*(port_weights_i + (DTT_SIZE2*DTT_SIZE21) * pair_ports[ipair][0]) +=d;
			*(port_weights_i + (DTT_SIZE2*DTT_SIZE21) * pair_ports[ipair][1]) +=d;
		}
		// find 2 best ports (resolving 2 pairs of close values)
		int bestPort1 = 0;
		float best_val= *port_weights_i;
		for (int cam = 1; cam < num_cams; cam++) {
			float val = *(port_weights_i + cam * (DTT_SIZE2*DTT_SIZE21));
			if (val > best_val){
				bestPort1 = cam;
				best_val =  val;
			}
		}
		int bestPort2 = (bestPort1 == 0) ? 1 : 0;
		best_val= *(port_weights_i + bestPort2 * (DTT_SIZE2*DTT_SIZE21));
		for (int cam = bestPort2 + 1; cam < num_cams; cam++){
			float val = *(port_weights_i + cam * (DTT_SIZE2*DTT_SIZE21));
			if ((cam != bestPort1) && (val > best_val)){
				bestPort2 = cam;
				best_val =  val;
			}
		}
#ifdef DEBUG9
		dbg_bestPort1[i] = bestPort1;
		dbg_bestPort2[i] = bestPort2;
#endif // #ifdef DEBUG9

		// find weighted average between these 2 ports
		float pw1 = *(port_weights_i + bestPort1 * (DTT_SIZE2*DTT_SIZE21));
		float w1 = pw1/(pw1 + *(port_weights_i + bestPort2 * (DTT_SIZE2*DTT_SIZE21)));
		float w2 = 1.0 - w1;
		float * rgba_i = rgba + i;
//#pragma unroll  // non-constant
		for (int ncol = 0; ncol < colors; ncol++) { // if (iclt_tile[0][ncol] != null) {
			float * mclt_col_i = mclt_tile_i + MCLT_UNION_LEN * ncol;
			* (rgba_i + ncol * (DTT_SIZE2*DTT_SIZE21))=
					w1 *  *(mclt_col_i + colors_offset * bestPort1) +
					w2 *  *(mclt_col_i + colors_offset * bestPort2);
		}


#ifdef DEBUG9
	}

#ifdef DEBUG8
	if (debug  && (threadIdx.x == 0) && (threadIdx.y == 0)){
		for (int ccam = 0; ccam < num_cams; ccam++) {
			printf("\n===== port_weight[%d] ====\n",ccam);
			debug_print_mclt(
					port_weights + (ccam * (DTT_SIZE2*DTT_SIZE21)),
					-1);
		}
		printf("\n+++++ best ports +++++\n");
		for (int dbg_row = 0; dbg_row < DTT_SIZE2; dbg_row++){
			for (int dbg_col = 0; dbg_col < DTT_SIZE2; dbg_col++){
				printf ("%1d[%1d] ",dbg_bestPort1[dbg_row *DTT_SIZE21 + dbg_col],dbg_bestPort2[dbg_row *DTT_SIZE21 + dbg_col]);
			}
			printf("\n");
		}
		printf("\n");
		for (int ncol = 0; ncol < colors; ncol++) {
			printf("\n+++++ rgba[%d] +++++\n",ncol);
			debug_print_mclt(
					rgba + (ncol * (DTT_SIZE2*DTT_SIZE21)),
					-1);
		}
	}
	__syncthreads();// __syncwarp();
#endif

	__syncthreads();// __syncwarp();
	for (int pass = 0; pass < 8; pass ++) {
		// below non-parametrized !
		int row = pass * 2 + (threadIdx.y >> 1);
		int col = ((threadIdx.y & 1) << 3) + threadIdx.x;
		int i = row * DTT_SIZE21 + col;
		float * crms_i = crms+i;
		float * port_weights_i = port_weights + i;
		float * mclt_tile_i = mclt_tile +  i; // has gaps to align in a union !
		float * alpha_i = alpha + i;
		float * rgba_i = rgba + i;
		int row_sym = row ^ ((row & 8)? 0xf : 0);
		int col_sym = col ^ ((col & 8)? 0xf : 0);
		float wnd2 = HWINDOW_SQ[row_sym] * HWINDOW_SQ[col_sym];
		float wnd2_inv = 1.0/wnd2;

#endif // #ifdef DEBUG9

		// recalculate all weights using difference from this average of the best pair
		for (int cam = 0; cam < num_cams; cam++) { // if ((port_mask & ( 1 << ip)) != 0){
			float * mclt_cam_i = mclt_tile_i + cam * colors_offset;
			float d2_ip = 0;
			for (int ncol = 0; ncol < colors; ncol++) { //  if (iclt_tile[0][ncol] != null){
				float * mclt_cam_col_i = mclt_cam_i + MCLT_UNION_LEN * ncol; // DTT_SIZE2*DTT_SIZE21 * ncol;
				float dc = *(mclt_cam_col_i) - * (rgba_i + ncol * (DTT_SIZE2*DTT_SIZE21));
				dc    *= wnd2_inv; // /= lt_window[i]; // to compensate fading near the edges
				d2_ip += *(chn_weights + ncol) * dc * dc;
			}
			// TODO: Should it use pair_dist2r ? no as it is relative?
			//				port_weights[ip][i] = Math.exp(-ksigma * d2[ip]);

			*(port_weights_i + (DTT_SIZE2*DTT_SIZE21) * cam) = expf(-ksigma * d2_ip) + (FAT_ZERO_WEIGHT);

		}
		// and now make a new average with those weights
		// Inserting dust remove here
		if (dust_remove) {
			int worstPort = 0;
			float worst_val= *port_weights_i;
			for (int cam = 1; cam < num_cams; cam++) {
				float val = *(port_weights_i + cam * (DTT_SIZE2*DTT_SIZE21));
				if (val < worst_val){
					worstPort = cam;
					worst_val = val;
				}
			}
			float avg = -worst_val; // avoid conditional
#pragma unroll
			for (int cam = 0; cam < num_cams; cam++){
					avg += *(port_weights_i + cam * (DTT_SIZE2*DTT_SIZE21));
			}
			avg /= (num_cams -1);
			float scale = 1.0 + worst_val * (avg - worst_val)/(avg * avg * (num_cams-1));
			for (int cam = 0; cam < num_cams; cam++){
				if (cam != worstPort){
					*(port_weights_i + cam * (DTT_SIZE2*DTT_SIZE21)) *= scale;
				}
			}
			*(port_weights_i + worstPort * (DTT_SIZE2*DTT_SIZE21)) *= worst_val/avg;
		}


#ifdef DEBUG9
	}

#ifdef DEBUG8
	if (debug  && (threadIdx.x == 0) && (threadIdx.y == 0)){
		for (int ccam = 0; ccam < num_cams; ccam++) {
			printf("\n===== UPDATED port_weight[%d] after dust_remove ====\n",ccam);
			debug_print_mclt(
					port_weights + (ccam * (DTT_SIZE2*DTT_SIZE21)),
					-1);
		}
	}
	__syncthreads();// __syncwarp();
#endif

	__syncthreads();// __syncwarp();
	for (int pass = 0; pass < 8; pass ++) {
		// below non-parametrized !
		int row = pass * 2 + (threadIdx.y >> 1);
		int col = ((threadIdx.y & 1) << 3) + threadIdx.x;
		int i = row * DTT_SIZE21 + col;
		float * crms_i = crms+i;
		float * port_weights_i = port_weights + i;
		float * mclt_tile_i = mclt_tile +  i; // has gaps to align in a union !
		float * alpha_i = alpha + i;
		float * rgba_i = rgba + i;
		int row_sym = row ^ ((row & 8)? 0xf : 0);
		int col_sym = col ^ ((col & 8)? 0xf : 0);
		float wnd2 = HWINDOW_SQ[row_sym] * HWINDOW_SQ[col_sym];
		float wnd2_inv = 1.0/wnd2;

#endif // #ifdef DEBUG9
		///
		if (rbg_tile && (calc_extra == 0)) { // will keep debayered if (calc_extra == 0)
			float k = 0.0;
			int rbga_offset = colors * (DTT_SIZE2*DTT_SIZE21); // padded in union !
#pragma unroll
			for (int cam = 0; cam < num_cams; cam++){
				k += *(port_weights_i + (DTT_SIZE2*DTT_SIZE21) * cam); // port_weights[ip][i];
			}
			k = 1.0/k;
			float * rbg_tile_i = rbg_tile + i;
#pragma unroll  // non-constant
			for (int ncol = 0; ncol < colors; ncol++) { // if (iclt_tile[0][ncol] != null) {
				float * rgba_col_i = rgba_i + ncol * (DTT_SIZE2*DTT_SIZE21);
//				float * mclt_col_i = mclt_tile_i + MCLT_UNION_LEN * ncol;
				float * rbg_col_i = rbg_tile_i + ncol * (DTT_SIZE2*DTT_SIZE21); // different gap between tiles than MCLT_UNION_LEN
				*rgba_col_i = 0.0; // color_avg[ncol][i] = 0;
#pragma unroll
				for (int cam = 0; cam < num_cams; cam++) {
//					*rgba_col_i += k * *(port_weights_i + (DTT_SIZE2*DTT_SIZE21) * cam) * *(mclt_col_i + cam * colors_offset);
					*rgba_col_i += k * *(port_weights_i + (DTT_SIZE2*DTT_SIZE21) * cam) * *(rbg_col_i +  cam * rbga_offset);
				}
			}
		}
//	int colors_offset = colors * MCLT_UNION_LEN; // padded in union !

		// calculate alpha from channel weights. Start with just a sum of weights?
//		int used_ports = num_cams;
//		if (dust_remove){
//			used_ports--;
//		}
		float a = 0;

//#pragma unroll
		for (int cam = 0; cam < num_cams; cam++) {
			a +=  *(port_weights_i + (DTT_SIZE2*DTT_SIZE21) * cam);
		}
		*alpha_i = wnd2 * a / num_cams; // used_ports;
	}// for (int pass = 0; pass < 8; pass ++)
	__syncthreads();

#ifdef DEBUG7A // 8
	if (debug  && (threadIdx.x == 0) && (threadIdx.y == 0)){
		printf("\ntile_combine_rgba() final\n");
		for (int ncol = 0; ncol < colors; ncol++) {
			printf("\ntile_combine_rgba() rgba[%d]\n",ncol);
			debug_print_mclt(
					rgba + (ncol * (DTT_SIZE2*DTT_SIZE21)),
					-1);
		}
		printf("\ntile_combine_rgba() alpha\n");
		debug_print_mclt(
				alpha, //
				-1);
		for (int cam = 0; cam < num_cams; cam++) {
			printf("\ntile_combine_rgba() port_weights[%d]\n",cam);
			debug_print_mclt(
					port_weights + (cam * (DTT_SIZE2*DTT_SIZE21)),
					-1);
		}
		for (int ncol = 0; ncol < (colors + 1); ncol++) {
			printf("\ntile_combine_rgba() crms[%d]\n",ncol);
			debug_print_mclt(
					crms + (ncol * (DTT_SIZE2*DTT_SIZE21)),
					-1);
		}
	}
	__syncthreads();// __syncwarp();
#endif


	if (calc_extra){
		for (int camera_num_offs = 0; camera_num_offs < num_cams; camera_num_offs+= blockDim.y) {// assuming num_cams is multiple blockDim.y
			int cam = camera_num_offs + threadIdx.y;
			int indx0 = cam * TEXTURE_THREADS_PER_TILE;
			int indx =  indx0 + threadIdx.x;
			//		max_diff_tmp[cam][threadIdx.x] = 0.0;
			max_diff_tmp[indx] = 0.0;
#pragma unroll
			for (int pass = 0; pass < 32; pass++){
				int row = (pass >> 1);
				int col = ((pass & 1) << 3) + threadIdx.x;
				int i = row * DTT_SIZE21 + col;
				float * mclt_cam_i = mclt_tile +  colors_offset * cam + i;
				float d2 = 0.0;
//#pragma unroll  // non-constant
				for (int ncol = 0; ncol < colors; ncol++){
					float dc = *(mclt_cam_i + (DTT_SIZE2*(DTT_SIZE21 + 1)) * ncol) - *(rgba + (DTT_SIZE2*DTT_SIZE21) * ncol + i);
					d2 += *(chn_weights + ncol) * dc * dc;
				}
				//max_diff_tmp[cam][threadIdx.x] = fmaxf(max_diff_tmp[cam][threadIdx.x], d2);
				max_diff_tmp[indx] = fmaxf(max_diff_tmp[indx], d2);
			}
			__syncthreads();
			if (threadIdx.x == 0){ // combine results
				float mx = 0.0;
#pragma unroll
				for (int i = 0; i < TEXTURE_THREADS_PER_TILE; i++){
					//				mx = fmaxf(mx, max_diff_tmp[cam][i]);
					mx = fmaxf(mx, max_diff_tmp[indx0 + i]);
				}
				max_diff_shared[cam] = sqrtf(mx);
			}
#ifdef DEBUG22
			if (debug  && (threadIdx.x == 0) && (threadIdx.y == 0)){
				printf("\n 1. max_diff\n");
				printf("total %f %f %f %f\n",max_diff_shared[0],max_diff_shared[1],max_diff_shared[2],max_diff_shared[3]);
				for (int i = 0; i < TEXTURE_THREADS_PER_TILE; i++){
					//				printf("tmp[%d] %f %f %f %f\n",i, max_diff_tmp[0][i],max_diff_tmp[1][i],max_diff_tmp[2][i],max_diff_tmp[3][i]);
					printf("tmp[%d] %f %f %f %f\n",i,
							max_diff_tmp[0 * TEXTURE_THREADS_PER_TILE + i],
							max_diff_tmp[1 * TEXTURE_THREADS_PER_TILE + i],
							max_diff_tmp[2 * TEXTURE_THREADS_PER_TILE + i],
							max_diff_tmp[3 * TEXTURE_THREADS_PER_TILE + i]);
				}
				for (int ncol = 0; ncol < colors; ncol++){
					printf("\n average for color %d\n",ncol);
					debug_print_mclt(
							rgba + (DTT_SIZE2*DTT_SIZE21) * ncol,
							-1);
					for (int ncam = 0; ncam < num_cams;ncam ++){
						printf("\n mclt for color %d, camera %d\n",ncol,ncam);
						debug_print_mclt(
								mclt_tile +  (DTT_SIZE2*(DTT_SIZE21 + 1)) * ncol +  colors_offset * ncam,
								-1);
#if 0
						printf("\n rgb_tile for color %d, camera %d\n",ncol,ncam);
						if (rgb_tile) {
							debug_print_mclt(
									rbg_tile +  (DTT_SIZE2*(DTT_SIZE21 + 1)) * ncol +  colors_offset * ncam,
									-1);
						}
#endif
					}
				}
			}
			__syncthreads();// __syncwarp();

#endif // #ifdef DEBUG22
		} // for (int camera_num_offs = 0; camera_num_offs < num_cams; camera_num_offs+= blockDim.y)

#ifdef DEBUG7A
		if (debug  && (threadIdx.x == 0) && (threadIdx.y == 0)){
			printf("\n X2. max_diff\n");
			printf("total ");for (int ccam = 0; ccam < num_cams; ccam++) {printf("%f, ",max_diff_shared[ccam]);} printf("\n");
			for (int i = 0; i < TEXTURE_THREADS_PER_TILE; i++){
				printf("tmp[%d]:   ",i); for (int ccam = 0; ccam < num_cams; ccam++) {printf("%f, ",max_diff_tmp[ccam * TEXTURE_THREADS_PER_TILE + i]);} printf("\n");
			}
			for (int ncol = 0; ncol < colors; ncol++){
				printf("\n average for color %d\n",ncol);
				debug_print_mclt(
						rgba + (DTT_SIZE2*DTT_SIZE21) * ncol,
						-1);
				for (int ncam = 0; ncam < num_cams;ncam ++){
					printf("\n mclt for color %d, camera %d\n",ncol,ncam);
					debug_print_mclt(
							mclt_tile +  (DTT_SIZE2*(DTT_SIZE21 + 1)) * ncol +  colors_offset * ncam,
							-1);
				}
			}
		}
		__syncthreads();// __syncwarp();
#endif // #ifdef DEBUG7A
	}



	if (calc_extra) {
		int incr = num_cams * TEXTURE_THREADS_PER_TILE;
		for (int camera_num_offs = 0; camera_num_offs < num_cams; camera_num_offs+= blockDim.y) {// assuming num_cams is multiple blockDim.y
			int cam = camera_num_offs + threadIdx.y;
			//		int cam =  threadIdx.y;  // BUG!
			int indx = cam * TEXTURE_THREADS_PER_TILE + threadIdx.x;
			int indx1 = indx;
			for (int ncol = 0; ncol < colors; ncol++){
				//			ports_rgb_tmp[ncol][cam][threadIdx.x] = 0.0;
				ports_rgb_tmp[indx1] = 0.0; // no difference in wrong zeros when removed
				indx1 += incr;
			}
#ifdef DEBUG7AXX
			if (debug  && (threadIdx.x == 0) && (threadIdx.y == 0)){
				printf("\nAA: indx = %d, camera_num_offs=%d, indx1=%d, cam = %d\n",indx, camera_num_offs, indx1, cam);
				__syncthreads();// __syncwarp();
			}
#endif // #ifdef DEBUG7A

#pragma unroll
			for (int pass = 0; pass < 32; pass++){
				int row = (pass >> 1);
				int col = ((pass & 1) << 3) + threadIdx.x;
				int i = row * DTT_SIZE21 + col;
				float * mclt_cam_i = mclt_tile +  colors_offset * cam + i;
				indx1 = indx;
				for (int ncol = 0; ncol < colors; ncol++){
					//				ports_rgb_tmp[ncol][cam][threadIdx.x] += *(mclt_cam_i + (DTT_SIZE2*(DTT_SIZE21 +1)) * ncol);
//					ports_rgb_tmp[indx1 += incr] += 1.0; /// *(mclt_cam_i + (DTT_SIZE2*(DTT_SIZE21 +1)) * ncol);
					ports_rgb_tmp[indx1] += *(mclt_cam_i + (DTT_SIZE2*(DTT_SIZE21 +1)) * ncol);
					indx1 += incr;

				}
			}
			__syncthreads();
#ifdef DEBUG7AXX
			if (debug  && (threadIdx.x == 0) && (threadIdx.y == 0)){
				printf("\nBB: indx = %d, camera_num_offs=%d, indx1=%d, cam = %d\n",indx, camera_num_offs, indx1, cam);
				__syncthreads();// __syncwarp();
			}
#endif // #ifdef DEBUG7A

			if (threadIdx.x == 0){ // combine results
				for (int ncol = 0; ncol < colors; ncol++){
					int indx2 = ncol * num_cams + cam;
					//				ports_rgb_shared[ncol][cam] = 0;
					ports_rgb_shared[indx2] = 0;
					int indx3 = indx2 * TEXTURE_THREADS_PER_TILE;
#pragma unroll
					for (int i = 0; i < TEXTURE_THREADS_PER_TILE; i++){
						//					ports_rgb_shared[ncol][cam] += ports_rgb_tmp[ncol][cam][i];
						ports_rgb_shared[indx2] += ports_rgb_tmp[indx3++];
					}
					ports_rgb_shared[indx2] /= DTT_SIZE2*DTT_SIZE2; // correct for window?
				}
			}
		} // for (int camera_num_offs = 0; camera_num_offs < num_cams; camera_num_offs+= blockDim.y) {

		__syncthreads();
#ifdef DEBUG7A
		if (debug  && (threadIdx.x == 0) && (threadIdx.y == 0)){
			printf("\n 2. max_diff, ports_rgb_shared, DBG_TILE = %d\n",DBG_TILE);
			//				printf("total %f %f %f %f\n",max_diff_shared[0],max_diff_shared[1],max_diff_shared[2],max_diff_shared[3]);
			printf("max_diff_shared ");for (int ccam = 0; ccam < num_cams; ccam++) {printf("%f, ",max_diff_shared[ccam]);} printf("\n");
			for (int i = 0; i < TEXTURE_THREADS_PER_TILE; i++){
				printf("tmp[%d]:   ",i); for (int ccam = 0; ccam < num_cams; ccam++) {printf("%f, ",max_diff_tmp[ccam * TEXTURE_THREADS_PER_TILE + i]);} printf("\n");
			}

			for (int ncol = 0; ncol < colors; ncol++){
				printf("\n%d:ports_rgb_shared ",ncol);
				for (int ccam = 0; ccam < num_cams; ccam++) {printf("%f, ",ports_rgb_shared[ ncol *num_cams + ccam]);} printf("\n");

				for (int i = 0; i < TEXTURE_THREADS_PER_TILE; i++){
					printf("ports_rgb_tmp[%d] ",i);
					for (int ccam = 0; ccam < num_cams; ccam++) {printf("%f, ",ports_rgb_tmp[(ncol*num_cams + ccam) * TEXTURE_THREADS_PER_TILE+ i]);} printf("\n");
				}
			}

		}
		__syncthreads();// __syncwarp();
#endif // #ifdef DEBUG7A

	}
}

// ------------- Debugging functions, output compared against tested CPU/Java implementation ---

/**
 * Print LPF data (8x8)
 * @param lpf_tile             LPF data to print
 */
__device__ void debug_print_lpf(
		float * lpf_tile)
{
#ifdef	HAS_PRINTF
	for (int dbg_row = 0; dbg_row < DTT_SIZE; dbg_row++){
		for (int dbg_col = 0; dbg_col < DTT_SIZE; dbg_col++){
			printf ("%10.5f ", lpf_tile[dbg_row * DTT_SIZE + dbg_col]);
		}
		printf("\n");
	}
#endif
}

/**
 * Print CLT tile (4x8x8)
 * @param clt_tile             CLT data to print [4][DTT_SIZE][DTT_SIZE + 1], // +1 to alternate column ports)
 * @param color                print color if >=0, skip if negative
 */
__device__ void debug_print_clt1(
		float * clt_tile, //         [4][DTT_SIZE][DTT_SIZE1], // +1 to alternate column ports)
		const int color,
		int mask)
{
#ifdef	HAS_PRINTF
	if (color >= 0) printf("----------- Color = %d -----------\n",color);
	for (int dbg_quadrant = 0; dbg_quadrant < 4; dbg_quadrant++){
		printf("----------- Quadrant (c(h)-c(v), s-c, c-s, s-s) = %d -----------\n",dbg_quadrant);
		if ((mask >> dbg_quadrant) & 1) {
			for (int dbg_row = 0; dbg_row < DTT_SIZE; dbg_row++){
				for (int dbg_col = 0; dbg_col < DTT_SIZE; dbg_col++){
					printf ("%10.5f ", clt_tile[(dbg_quadrant*DTT_SIZE + dbg_row)*DTT_SIZE1 + dbg_col]);
				}
				printf("\n");
			}
		}
		printf("\n");
	}
#endif
}

/**
 * Print selected quadrants of CLT tile (4x8x8)
 * @param clt_tile             CLT data to print [4][DTT_SIZE][DTT_SIZE + 1], // +1 to alternate column ports)
 * @param color                print color if >=0, skip if negative
 * @param mask                 bitmask of the quadrants to include in the output
 * @param scale                scale all results by this value
 */
__device__ void debug_print_clt_scaled(
		float * clt_tile, //         [4][DTT_SIZE][DTT_SIZE1], // +1 to alternate column ports)
		const int color,
		int mask,
		float scale)
{
#ifdef	HAS_PRINTF
	if (color >= 0) printf("----------- Color = %d -----------\n",color);
	for (int dbg_quadrant = 0; dbg_quadrant < 4; dbg_quadrant++){
		printf("----------- Quadrant (c(h)-c(v), s-c, c-s, s-s) = %d -----------\n",dbg_quadrant);
		if ((mask >> dbg_quadrant) & 1) {
			for (int dbg_row = 0; dbg_row < DTT_SIZE; dbg_row++){
				for (int dbg_col = 0; dbg_col < DTT_SIZE; dbg_col++){
					printf ("%10.5f ", scale * clt_tile[(dbg_quadrant*DTT_SIZE + dbg_row)*DTT_SIZE1 + dbg_col]);
				}
				printf("\n");
			}
		}
		printf("\n");
	}
#endif
}

/**
 * Print MCLT tile (16x16)
 * @param mclt_tile            MCLT data to print [4][2*DTT_SIZE][2*DTT_SIZE + 1], // +1 to alternate column ports)
 * @param color                print color if >=0, skip if negative
 */
__device__ void debug_print_mclt(
		float * mclt_tile, //         [4][DTT_SIZE][DTT_SIZE1], // +1 to alternate column ports)
		const int color)
{
#ifdef	HAS_PRINTF
	if (color >= 0) printf("----------- Color = %d -----------\n",color);
	for (int dbg_row = 0; dbg_row < DTT_SIZE2; dbg_row++){
		for (int dbg_col = 0; dbg_col < DTT_SIZE2; dbg_col++){
			printf ("%10.4f ", mclt_tile[dbg_row *DTT_SIZE21 + dbg_col]);
		}
		printf("\n");
	}
	printf("\n");
#endif
}

/**
 * Print 2D correlation tile (maximal 15x15 , ((2 * corr_radius + 1) * (2 * corr_radius + 1)) )
 * @param corr_radius          correlation radius - reduces amount of correlation data by trimming outer elements
 * @param mclt_tile            2D correlation tile in a line-scan order [(2 * corr_radius + 1) * (2 * corr_radius + 1)]
 * @param color                print color if >=0, skip if negative
 */
__device__ void debug_print_corr_15x15(
		int     corr_radius,
		float * mclt_tile, //DTT_SIZE2M1 x DTT_SIZE2M1
		const int color)
{
#ifdef	HAS_PRINTF
	int size2r1 = 2 * corr_radius + 1;
	if (color >= 0) printf("----------- Color = %d -----------\n",color);
	for (int dbg_row = 0; dbg_row < size2r1; dbg_row++){
		for (int dbg_col = 0; dbg_col < size2r1; dbg_col++){
			printf ("%10.5f ", mclt_tile[dbg_row * size2r1 + dbg_col]);
		}
		printf("\n");
	}
	printf("\n");
#endif
}



