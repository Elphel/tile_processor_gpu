/**
 **
 ** tp_defines.h
 **
 ** Copyright (C) 2020 Elphel, Inc.
 **
 ** -----------------------------------------------------------------------------**
 **
 **  tp_defines.h is free software: you can redistribute it and/or modify
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
* \file tp_defines.h
* \brief Defines for running in C++ environment, replaced when called from Java

*/
// Avoiding includes in jcuda, all source files will be merged
#pragma once
#ifndef JCUDA
#include <stdio.h>
#define THREADSX              (DTT_SIZE)
#define TEST_LWIR                      1
#define NUM_CAMS                      16 // now maximal number of cameras
//#define NUM_PAIRS                      6
//#define NUM_COLORS                     1 //3
// kernels [num_cams][num_colors][KERNELS_HOR][KERNELS_VERT][4][64]
#if TEST_LWIR
	#define IMG_WIDTH                   640
	#define IMG_HEIGHT                  512
	#define KERNELS_HOR                 82 // 80+2
	#define KERNELS_VERT                66 // 64+2
#else
	#define IMG_WIDTH                  2592
	#define IMG_HEIGHT                 1936
	#define KERNELS_HOR                 164  // 2592 / 16 + 2
	#define KERNELS_VERT                123  // 1936 / 16 + 2
#endif
#define KERNELS_LSTEP                  4
#define THREADS_PER_TILE               8
#define TILES_PER_BLOCK                4
#define CORR_THREADS_PER_TILE          8
#define CORR_TILES_PER_BLOCK           4
#define CORR_TILES_PER_BLOCK_NORMALIZE 4 // increase to 8?
#define CORR_TILES_PER_BLOCK_COMBINE   4 // increase to 16?
//#define TEXTURE_THREADS               32 //
#define NUM_THREADS                   32
#define TEXTURE_THREADS_PER_TILE       8
#define TEXTURE_TILES_PER_BLOCK        1
#define IMCLT_THREADS_PER_TILE        16
#define IMCLT_TILES_PER_BLOCK          4
#define CORR_NTILE_SHIFT               8 // higher bits - number of a pair, other bits tile number
// only lower bit will be used to request correlations, correlation mask will be common for all the scene
//#define CORR_PAIRS_MASK             0x3f// lower bits used to address correlation pair for the selected tile
#define CORR_TEXTURE_BIT               7 // bit 7 used to request texture for the tile
#define TASK_CORR_BITS                 4
#define TASK_TEXTURE_N_BIT             0 // Texture with North neighbor
#define TASK_TEXTURE_E_BIT             1 // Texture with East  neighbor
#define TASK_TEXTURE_S_BIT             2 // Texture with South neighbor
#define TASK_TEXTURE_W_BIT             3 // Texture with West  neighbor
#define TASK_TEXTURE_BIT               3 // bit to request texture calculation int task field of struct tp_task
#define LIST_TEXTURE_BIT               7 // bit to request texture calculation
#define CORR_OUT_RAD                   7 // full tile (15x15), was 4 (9x9)
#define FAT_ZERO_WEIGHT                0.0001 // add to port weights to avoid nan

#define THREADS_DYNAMIC_BITS           5 // treads in block for CDP creation of the texture list

#define DBG_DISPARITY                  0.0 // 56.0//   0.0 // 56.0 // disparity for which to calculate offsets (not needed in Java)
#define RBYRDIST_LEN                5001   // for doubles 10001 - floats   // length of rByRDist to allocate shared memory
#define RBYRDIST_STEP                  0.0004 // for doubles, 0.0002 - floats // to fit into GPU shared memory (was 0.001);
#define TILES_PER_BLOCK_GEOM          (32/NUM_CAMS)   // each tile has NUM_CAMS threads

// only used in C++ test
#define TILESX        (IMG_WIDTH / DTT_SIZE)
#define TILESY        (IMG_HEIGHT / DTT_SIZE)
#define TILESYA       ((TILESY +3) & (~3))




#define DEBUG_OOB1 1

// Use CORR_OUT_RAD for the correlation output

//#define DBG_TILE_X     40
//#define DBG_TILE_Y     80
#if TEST_LWIR
	#define DBG_TILE_X    50 // 52 // 32 // 162 // 151 // 161 // 49
	#define DBG_TILE_Y    19 //  5 // 36 // 88 // 121 // 69  // 111 // 66
	#define DBG_TILE    (DBG_TILE_Y * 80 + DBG_TILE_X)
#else
	#define DBG_TILE_X     114 // 32 // 162 // 151 // 161 // 49
	#define DBG_TILE_Y     51  // 52  // 88 // 121 // 69  // 111 // 66
	#define DBG_TILE    (DBG_TILE_Y * 324 + DBG_TILE_X)
#endif
#undef DBG_MARK_DBG_TILE
//#undef DBG_TILE

//#undef HAS_PRINTF
#define HAS_PRINTF
//7
//#define DEBUG1 1
//#define DEBUG2 1
//#define DEBUG3 1
//#define DEBUG4 1
//#define DEBUG5 1
//#define DEBUG6 1

// #define DEBUG7 1
//// #define DEBUG7A 1
/*
#define DEBUG7 1
#define DEBUG8 1
#define DEBUG9 1
*/
#define DEBUG8A 1
//textures
//#define DEBUG10 1
//#define DEBUG11 1
//#define DEBUG12 1
//#define USE_textures_gen
//#define DEBUG_OOB1 1
// geom
//#define DEBUG20 1


#if (DBG_TILE_X >= 0) && (DBG_TILE_Y >= 0)
#define DEBUG20 1 // Geometry Correction
#define DEBUG21 1 // Geometry Correction
//#define DEBUG210 1
////#define DEBUG30 1
//#define DEBUG22 1
//#define DEBUG23 1

#endif //#if (DBG_TILE_X >= 0) && (DBG_TILE_Y >= 0)

#endif //#ifndef JCUDA

