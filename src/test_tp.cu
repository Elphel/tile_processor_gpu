/**
 **
 ** dtt8x8.cu - CPU test code to run GPU tile processor
 **
 ** Copyright (C) 2018 Elphel, Inc.
 **
 ** -----------------------------------------------------------------------------**
 **
 **  dtt8x8.cu is free software: you can redistribute it and/or modify
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

// #define NOCORR
// #define NOCORR_TD
// #define NOTEXTURES
// #define NOTEXTURE_RGBA
#define SAVE_CLT
//#define NO_DP
#define CORR_INTER_SELF 1


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

// for reading binary files
#include <fstream>
#include <iterator>
#include <vector>

#include "dtt8x8.h"
#include "geometry_correction.h"
#include "TileProcessor.cuh"

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
#define CORR_OUT_RAD                   7 // full tile (15x15), was 4 (9x9)
#define DBG_DISPARITY                  0.0 // 56.0//   0.0 // 56.0 // disparity for which to calculate offsets (not needed in Java)
// only used in C++ test
#define TILESX        (IMG_WIDTH / DTT_SIZE)
#define TILESY        (IMG_HEIGHT / DTT_SIZE)
#define TILESYA       ((TILESY +3) & (~3))



float * copyalloc_kernel_gpu(float * kernel_host,
		                int size, // size in floats
						int full_size)
{
	float *kernel_gpu;
    checkCudaErrors(cudaMalloc((void **)&kernel_gpu, full_size * sizeof(float)));
    checkCudaErrors(cudaMemcpy( // segfault
    		kernel_gpu,
    		kernel_host,
			size * sizeof(float),
            cudaMemcpyHostToDevice));
    return kernel_gpu;
}

float * copyalloc_kernel_gpu(float * kernel_host,
		                int size)
{
	return copyalloc_kernel_gpu(kernel_host,
            size, // size in floats
			size);
}



float * alloccopy_from_gpu(
		float * gpu_data,
		float * cpu_data, // if null, will allocate
		int size)
{
	if (!cpu_data) {
		cpu_data = (float *)malloc(size*sizeof(float));
	}
	checkCudaErrors(cudaMemcpy( // segfault
			cpu_data,
			gpu_data,
			size * sizeof(float),
			cudaMemcpyDeviceToHost));

	return cpu_data;
}


float * alloc_kernel_gpu(int size) // size in floats
{
	float *kernel_gpu;
    checkCudaErrors(cudaMalloc((void **)&kernel_gpu, size * sizeof(float)));
    return kernel_gpu;
}


float ** copyalloc_pointers_gpu(float ** gpu_pointer,
		                int size) // number of entries (cameras)
{
	float ** gpu_pointer_to_gpu_pointers;
    checkCudaErrors(cudaMalloc((void **)&gpu_pointer_to_gpu_pointers, size * sizeof(float*)));
    checkCudaErrors(cudaMemcpy(
    		gpu_pointer_to_gpu_pointers,
			gpu_pointer,
			size * sizeof(float*),
            cudaMemcpyHostToDevice));
    return gpu_pointer_to_gpu_pointers;
}


float * copyalloc_image_gpu(float * image_host,
						size_t* dstride, // in floats !
		                int width,
						int height)
{
	float *image_gpu;
    checkCudaErrors(cudaMallocPitch((void **)&image_gpu, dstride, width * sizeof(float), height));
    checkCudaErrors(cudaMemcpy2D(
    		image_gpu,
            *dstride, //  * sizeof(float),
			image_host,
			width * sizeof(float), // make in 16*n?
            width * sizeof(float),
			height,
			cudaMemcpyHostToDevice));
    return image_gpu;
}

float * alloc_image_gpu(size_t* dstride, // in bytes!!
		                int width,
						int height)
{
	float *image_gpu;
    checkCudaErrors(cudaMallocPitch((void **)&image_gpu, dstride, width * sizeof(float), height));
    return image_gpu;
}

int get_file_size(std::string filename) // path to file
{
    FILE *p_file = NULL;
    p_file = fopen(filename.c_str(),"rb");
    fseek(p_file,0,SEEK_END);
    int size = ftell(p_file);
    fclose(p_file);
    return size;
}
int readFloatsFromFile(float *       data, // allocated array
					   const char *  path) // file path
{
    printf("readFloatsFromFile(%s)\n", path);

    int fsize = get_file_size(path);
    std::ifstream input(path, std::ios::binary );
    // copies all data into buffer
    std::vector<char> buffer((
            std::istreambuf_iterator<char>(input)),
            (std::istreambuf_iterator<char>()));
    std::copy( buffer.begin(), buffer.end(), (char *) data);
    printf("---- Bytes read: %d from %s\n", fsize, path);
	return 0;
}

float * readAllFloatsFromFile(const char *  path,
		int * len_in_floats) //
{
    int fsize = get_file_size(path);
    float * data = (float *) malloc(fsize);
    std::ifstream input(path, std::ios::binary );
    std::vector<char> buffer((
            std::istreambuf_iterator<char>(input)),
            (std::istreambuf_iterator<char>()));
    std::copy( buffer.begin(), buffer.end(), (char *) data);
    printf("---- Bytes read: %d from %s\n", fsize, path);
    * len_in_floats = fsize/sizeof(float);
    return data;

}



int writeFloatsToFile(float *       data, // allocated array
		               int           size, // length in elements
					   const char *  path) // file path
{

	std::ofstream ofile(path, std::ios::binary);
	ofile.write((char *) data, size * sizeof(float));
	return 0;
}

// Prepare low pass filter (64 long) to be applied to each quadrant of the CLT data
void set_clt_lpf(
		float * lpf,    // size*size array to be filled out
		float   sigma,
		const int     dct_size)
{
	int dct_len = dct_size * dct_size;
	if (sigma == 0.0f) {
		lpf[0] = 1.0f;
		for (int i = 1; i < dct_len; i++){
			lpf[i] = 0.0;
		}
	} else {
		for (int i = 0; i < dct_size; i++){
			for (int j = 0; j < dct_size; j++){
				lpf[i*dct_size+j] = exp(-(i*i+j*j)/(2*sigma));
			}
		}
		// normalize
		double sum = 0;
		for (int i = 0; i < dct_size; i++){
			for (int j = 0; j < dct_size; j++){
				double d = 	lpf[i*dct_size+j];
				d*=cos(M_PI*i/(2*dct_size))*cos(M_PI*j/(2*dct_size));
				if (i > 0) d*= 2.0;
				if (j > 0) d*= 2.0;
				sum +=d;
			}
		}
		for (int i = 0; i< dct_len; i++){
			lpf[i] /= sum;
		}
	}
}

int host_get_textures_shared_size( // in bytes
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
//	offs += num_cams * num_colors * DTT_SIZE2 * DTT_SIZE21;    //float mclt_tmp           [NUM_CAMS][NUM_COLORS][DTT_SIZE2][DTT_SIZE21];
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


void generate_RBGA_host(
		int                num_cams,           // number of cameras used
		// Parameters to generate texture tasks
		float            * gpu_ftasks,         // flattened tasks, 27 floats for quad EO, 99 floats for LWIR16
//		struct tp_task   * gpu_tasks,
		int                num_tiles,          // number of tiles in task list
		// declare arrays in device code?
		int              * gpu_texture_indices,// packed tile + bits (now only (1 << 7)
		int              * gpu_num_texture_tiles,  // number of texture tiles to process  (8 separate elements for accumulation)
		int              * gpu_woi,                // x,y,width,height of the woi
		int                width,  // <= TILES-X, use for faster processing of LWIR images (should be actual + 1)
		int                height, // <= TILES-Y, use for faster processing of LWIR images
		// Parameters for the texture generation
		float          ** gpu_clt,            // [num_cams] ->[TILES-Y][TILES-X][colors][DTT_SIZE*DTT_SIZE]
		// TODO: use geometry_correction rXY !
		struct gc       * gpu_geometry_correction,
		int               colors,             // number of colors (3/1)
		int               is_lwir,            // do not perform shot correction
		float             cpu_params[5],      // mitigating CUDA_ERROR_INVALID_PTX
		float             weights[3],         // scale for R,B,G should be host_array, not gpu
		int               dust_remove,        // Do not reduce average weight when only one image differs much from the average
		int               keep_weights,       // return channel weights after A in RGBA (was removed)
		const size_t      texture_rbga_stride,     // in floats
		float           * gpu_texture_tiles)  // (number of colors +1 + ?)*16*16 rgba texture tiles
{
	int               cpu_woi[4];
	int               cpu_num_texture_tiles[8];
	float             min_shot = cpu_params[0];           // 10.0
	float             scale_shot = cpu_params[1];         // 3.0
	float             diff_sigma = cpu_params[2];         // pixel value/pixel change
	float             diff_threshold = cpu_params[3];     // pixel value/pixel change
	float             min_agree = cpu_params[4];          // minimal number of channels to agree on a point (real number to work with fuzzy averages)
	int               tilesya =  ((height +3) & (~3)); //#define TILES-YA       ((TILES-Y +3) & (~3))
	dim3 threads0((1 << THREADS_DYNAMIC_BITS), 1, 1);
    int blocks_x = (width + ((1 << THREADS_DYNAMIC_BITS) - 1)) >> THREADS_DYNAMIC_BITS;
    dim3 blocks0 (blocks_x, height, 1);
    clear_texture_list<<<blocks0,threads0>>>(
    		gpu_texture_indices,
			width,
			height);
	checkCudaErrors(cudaDeviceSynchronize()); // not needed yet, just for testing
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
	checkCudaErrors(cudaDeviceSynchronize());
    // mark n/e/s/w used tiles from gpu_texture_indices memory to gpu_tasks lower 4 bits
    checkCudaErrors(cudaMemcpy(
    		(float * ) cpu_woi,
			gpu_woi,
			4  * sizeof(float),
			cudaMemcpyDeviceToHost));
    cpu_woi[0] = width;
    cpu_woi[1] = height;
    cpu_woi[2] = 0;
    cpu_woi[3] = 0;
    checkCudaErrors(cudaMemcpy(
    		gpu_woi,
			cpu_woi,
			4 * sizeof(float),
			cudaMemcpyHostToDevice));
    /*
    *(woi + 0) = width;  // TILES-X;
    *(woi + 1) = height; // TILES-Y;
    *(woi + 2) = 0; // maximal x
    *(woi + 3) = 0; // maximal y
    int * gpu_woi = (int  *) copyalloc_kernel_gpu(
    		(float * ) woi,
			4); // number of elements
   */
    // TODO: create gpu_woi to pass	(copy from woi)
    // set lower 4 bits in each gpu_ftasks task
    mark_texture_neighbor_tiles <<<blocks,threads>>>(
    		num_cams,            // int                num_cams,
			gpu_ftasks,          // float            * gpu_ftasks,         // flattened tasks, 27 floats for quad EO, 99 floats for LWIR16
			num_tiles,           // number of tiles in task list
			width,               // number of tiles in a row
			height,              // number of tiles rows
			gpu_texture_indices, // packed tile + bits (now only (1 << 7)
			gpu_woi);                // min_x, min_y, max_x, max_y

	checkCudaErrors(cudaDeviceSynchronize());
	/*
    checkCudaErrors(cudaMemcpy( //
    		(float * ) cpu_num_texture_tiles,
			gpu_num_texture_tiles,
			8 * sizeof(float),
			cudaMemcpyDeviceToHost));
	*/
    // Generate tile indices list, upper 24 bits - tile index, lower 4 bits: n/e/s/w neighbors, bit 7 - set to 1
    for (int i = 0; i <8; i++){
    	cpu_num_texture_tiles[i] = 0;
    }
    /*
    *(num_texture_tiles+0) = 0;
    *(num_texture_tiles+1) = 0;
    *(num_texture_tiles+2) = 0;
    *(num_texture_tiles+3) = 0;
    *(num_texture_tiles+4) = 0;
    *(num_texture_tiles+5) = 0;
    *(num_texture_tiles+6) = 0;
    *(num_texture_tiles+7) = 0;
    */
    // copy zeroed  num_texture_tiles
//    int * gpu_num_texture_tiles = (int  *) copyalloc_kernel_gpu(
//   		(float * ) num_texture_tiles,
//			8); // number of elements
    checkCudaErrors(cudaMemcpy(
    		gpu_num_texture_tiles,
			cpu_num_texture_tiles,
			8 * sizeof(float),
			cudaMemcpyHostToDevice));

    gen_texture_list <<<blocks,threads>>>(
    		num_cams,            // int                num_cams,
			gpu_ftasks,          // float            * gpu_ftasks,         // flattened tasks, 27 floats for quad EO, 99 floats for LWIR16
			num_tiles,           // number of tiles in task list
			width,               // number of tiles in a row
			height,              // int                height,               // number of tiles rows
			gpu_texture_indices, // packed tile + bits (now only (1 << 7)
			gpu_num_texture_tiles,   // number of texture tiles to process
			gpu_woi);                // x,y, here woi[2] = max_X, woi[3] - max-Y
	checkCudaErrors(cudaDeviceSynchronize()); // not needed yet, just for testing

    // copy gpu_woi back to host woi
    checkCudaErrors(cudaMemcpy(
    		(float * ) cpu_woi,
			gpu_woi,
			4  * sizeof(float),
			cudaMemcpyDeviceToHost));
//    *(cpu_woi + 2) += 1 - *(cpu_woi + 0); // width  (was min and max)
//    *(cpu_woi + 3) += 1 - *(cpu_woi + 1); // height (was min and max)
      cpu_woi[2] += 1 - cpu_woi[0]; // width  (was min and max)
      cpu_woi[3] += 1 - cpu_woi[1]; // height (was min and max)
    // copy host-modified data back to GPU
    checkCudaErrors(cudaMemcpy(
    		gpu_woi,
			cpu_woi,
			4 * sizeof(float),
			cudaMemcpyHostToDevice));

    // copy gpu_num_texture_tiles back to host num_texture_tiles
    checkCudaErrors(cudaMemcpy(
    		(float * ) cpu_num_texture_tiles,
			gpu_num_texture_tiles,
			8 * sizeof(float),
			cudaMemcpyDeviceToHost));

// Zero output textures. Trim
// texture_rbga_stride
//	 int texture_width =        (*(cpu_woi + 2) + 1) * DTT_SIZE;
//	 int texture_tiles_height = (*(cpu_woi + 3) + 1) * DTT_SIZE;
	 int texture_width =        (cpu_woi[2] + 1) * DTT_SIZE;
	 int texture_tiles_height = (cpu_woi[3] + 1) * DTT_SIZE;
	 int texture_slices =       colors + 1;

	 dim3 threads2((1 << THREADS_DYNAMIC_BITS), 1, 1);
	 int blocks_x2 = (texture_width + ((1 << (THREADS_DYNAMIC_BITS + DTT_SIZE_LOG2 )) - 1)) >> (THREADS_DYNAMIC_BITS + DTT_SIZE_LOG2);
	 dim3 blocks2 (blocks_x2, texture_tiles_height * texture_slices, 1); // each thread - 8 vertical


#ifdef DEBUG8A
	 int                cpu_texture_indices      [TILESX*TILESYA];
	 checkCudaErrors(cudaMemcpy(
			 (float * ) cpu_texture_indices,
			 gpu_texture_indices,
			 TILESX*TILESYA * sizeof(float),
			 cudaMemcpyDeviceToHost));
	 for (int i = 0; i < 256; i++){
		 int indx = cpu_texture_indices[i];
		 printf("%02d %04x %03d %03d %x\n",i,indx, (indx>>8) / 80, (indx >> 8) % 80, indx&0xff);
	 }
#endif // #ifdef DEBUG8A


	 clear_texture_rbga<<<blocks2,threads2>>>( // illegal value error
			 texture_width,
			 texture_tiles_height * texture_slices, // int               texture_slice_height,
			 texture_rbga_stride,                   // const size_t      texture_rbga_stride,     // in floats 8*stride
			 gpu_texture_tiles) ;                   // float           * gpu_texture_tiles);
	 // Run 8 times - first 4 1-tile offsets  inner tiles (w/o verifying margins), then - 4 times with verification and ignoring 4-pixel
	 // oversize (border 16x 16 tiles overhang by 4 pixels)

 	checkCudaErrors(cudaDeviceSynchronize());  // not needed yet, just for testing
	 for (int pass = 0; pass < 8; pass++){
		 int num_cams_per_thread = NUM_THREADS / TEXTURE_THREADS_PER_TILE; // 4 cameras parallel, then repeat
		 dim3 threads_texture(TEXTURE_THREADS_PER_TILE, num_cams_per_thread, 1); // TEXTURE_TILES_PER_BLOCK, 1);

		 int border_tile =  pass >> 2;
		 int ntt = *(cpu_num_texture_tiles + ((pass & 3) << 1) + border_tile);
		 dim3 grid_texture((ntt + TEXTURE_TILES_PER_BLOCK-1) / TEXTURE_TILES_PER_BLOCK,1,1); // TEXTURE_TILES_PER_BLOCK = 1
		 int ti_offset = (pass & 3) * (width * (tilesya >> 2)); //  (TILES-X * (TILES-YA >> 2));  // 1/4
		 if (border_tile){
			 ti_offset += width * (tilesya >> 2) - ntt; // TILES-X * (TILES-YA >> 2) - ntt;
		 }
#ifdef DEBUG8A
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
		 int shared_size = host_get_textures_shared_size( // in bytes
				 num_cams,     // int                num_cams,     // actual number of cameras
				 colors,   // int                num_colors,   // actual number of colors: 3 for RGB, 1 for LWIR/mono
				 0);           // int *              offsets);     // in floats
    	 printf("\n2. shared_size=%d, num_cams=%d, colors=%d\n",shared_size,num_cams, colors);
		 cudaFuncSetAttribute(textures_accumulate, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size); // 65536); // for CC 7.5
		 cudaFuncSetAttribute(textures_accumulate, cudaFuncAttributePreferredSharedMemoryCarveout,cudaSharedmemCarveoutMaxShared);
		 textures_accumulate <<<grid_texture,threads_texture, shared_size>>>(
				 num_cams,                        // int               num_cams,           // number of cameras used
				 gpu_woi,                             // int             * woi,                // x, y, width,height
				 gpu_clt,                         // float          ** gpu_clt,            // [num_cams] ->[TILES-Y][TILES-X][colors][DTT_SIZE*DTT_SIZE]
				 ntt,                             // size_t            num_texture_tiles,  // number of texture tiles to process
				 ti_offset,                       //                gpu_texture_indices_offset,// add to gpu_texture_indices
				 gpu_texture_indices, //  + ti_offset, // int             * gpu_texture_indices,// packed tile + bits (now only (1 << 7)
//				 gpu_texture_indices + ti_offset, // int             * gpu_texture_indices,// packed tile + bits (now only (1 << 7)
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
				 0,                               // int               keep_weights,       // return channel weights after A in RGBA (was removed) (should be 0 if gpu_texture_rbg)?
				 // combining both non-overlap and overlap (each calculated if pointer is not null )
				 texture_rbga_stride,             // size_t      texture_rbg_stride, // in floats
				 gpu_texture_tiles,               // float           * gpu_texture_rbg,     // (number of colors +1 + ?)*16*16 rgba texture tiles
				 0,                               // size_t      texture_stride,     // in floats (now 256*4 = 1024)
				 (float *) 0, // gpu_texture_tiles, // float           * gpu_texture_tiles);  // (number of colors +1 + ?)*16*16 rgba texture tiles
				 0, // 1, // int               linescan_order,     // if !=0 then output gpu_diff_rgb_combo in linescan order, else  - in gpu_texture_indices order
				 (float *)0, //);//gpu_diff_rgb_combo);             // float           * gpu_diff_rgb_combo) // diff[num_cams], R[num_cams], B[num_cams],G[num_cams]
				 width);
	    	checkCudaErrors(cudaDeviceSynchronize()); // not needed yet, just for testing
		 /* */

	 }
//	 checkCudaErrors(cudaFree(gpu_woi));
//	 checkCudaErrors(cudaFree(gpu_num_texture_tiles));
	 //	 __syncthreads();
}



/**
**************************************************************************
*  Program entry point
*
* \param argc       [IN] - Number of command-line arguments
* \param argv       [IN] - Array of command-line arguments
*
* \return Status code
*/


int main(int argc, char **argv)
{
    //
    // Sample initialization
    //
    printf("%s Starting...\n\n", argv[0]);
    printf("sizeof(float*)=%d\n",(int)sizeof(float*));

    //initialize CUDA
    findCudaDevice(argc, (const char **)argv);

#if TEST_LWIR
    const char* kernel_file[] = {
    		"clt/aux_chn0_transposed.kernel",
			"clt/aux_chn1_transposed.kernel",
			"clt/aux_chn2_transposed.kernel",
			"clt/aux_chn3_transposed.kernel",
			"clt/aux_chn4_transposed.kernel",
			"clt/aux_chn5_transposed.kernel",
			"clt/aux_chn6_transposed.kernel",
			"clt/aux_chn7_transposed.kernel",
			"clt/aux_chn8_transposed.kernel",
			"clt/aux_chn9_transposed.kernel",
			"clt/aux_chn10_transposed.kernel",
			"clt/aux_chn11_transposed.kernel",
			"clt/aux_chn12_transposed.kernel",
			"clt/aux_chn13_transposed.kernel",
			"clt/aux_chn14_transposed.kernel",
			"clt/aux_chn15_transposed.kernel"};

    const char* kernel_offs_file[] = {
    		"clt/aux_chn0_transposed.kernel_offsets",
			"clt/aux_chn1_transposed.kernel_offsets",
			"clt/aux_chn2_transposed.kernel_offsets",
			"clt/aux_chn3_transposed.kernel_offsets",
			"clt/aux_chn4_transposed.kernel_offsets",
			"clt/aux_chn5_transposed.kernel_offsets",
			"clt/aux_chn6_transposed.kernel_offsets",
			"clt/aux_chn7_transposed.kernel_offsets",
			"clt/aux_chn8_transposed.kernel_offsets",
			"clt/aux_chn9_transposed.kernel_offsets",
			"clt/aux_chn10_transposed.kernel_offsets",
			"clt/aux_chn11_transposed.kernel_offsets",
			"clt/aux_chn12_transposed.kernel_offsets",
			"clt/aux_chn13_transposed.kernel_offsets",
			"clt/aux_chn14_transposed.kernel_offsets",
			"clt/aux_chn15_transposed.kernel_offsets"};

    const char* image_files[] = {
    		"clt/aux_chn0.bayer",
			"clt/aux_chn1.bayer",
			"clt/aux_chn2.bayer",
			"clt/aux_chn3.bayer",
			"clt/aux_chn4.bayer",
			"clt/aux_chn5.bayer",
			"clt/aux_chn6.bayer",
			"clt/aux_chn7.bayer",
			"clt/aux_chn8.bayer",
			"clt/aux_chn9.bayer",
			"clt/aux_chn10.bayer",
			"clt/aux_chn11.bayer",
			"clt/aux_chn12.bayer",
			"clt/aux_chn13.bayer",
			"clt/aux_chn14.bayer",
			"clt/aux_chn15.bayer"};

    const char* ports_offs_xy_file[] = {
    		"clt/aux_chn0.portsxy",
			"clt/aux_chn1.portsxy",
			"clt/aux_chn2.portsxy",
			"clt/aux_chn3.portsxy",
			"clt/aux_chn4.portsxy",
			"clt/aux_chn5.portsxy",
			"clt/aux_chn6.portsxy",
			"clt/aux_chn7.portsxy",
			"clt/aux_chn8.portsxy",
			"clt/aux_chn9.portsxy",
			"clt/aux_chn10.portsxy",
			"clt/aux_chn11.portsxy",
			"clt/aux_chn12.portsxy",
			"clt/aux_chn13.portsxy",
			"clt/aux_chn14.portsxy",
			"clt/aux_chn15.portsxy"};

//#ifndef DBG_TILE
#ifdef SAVE_CLT
    const char* ports_clt_file[] = { // never referenced
    		"clt/aux_chn0.clt",
			"clt/aux_chn1.clt",
			"clt/aux_chn2.clt",
			"clt/aux_chn3.clt",
    		"clt/aux_chn4.clt",
			"clt/aux_chn5.clt",
			"clt/aux_chn6.clt",
			"clt/aux_chn7.clt",
    		"clt/aux_chn8.clt",
			"clt/aux_chn9.clt",
			"clt/aux_chn10.clt",
			"clt/aux_chn11.clt",
    		"clt/aux_chn12.clt",
			"clt/aux_chn13.clt",
			"clt/aux_chn14.clt",
			"clt/aux_chn15.clt"};

#endif
    const char* result_rbg_file[] = {
    		"clt/aux_chn0.rbg",
			"clt/aux_chn1.rbg",
			"clt/aux_chn2.rbg",
			"clt/aux_chn3.rbg",
			"clt/aux_chn4.rbg",
			"clt/aux_chn5.rbg",
			"clt/aux_chn6.rbg",
			"clt/aux_chn7.rbg",
			"clt/aux_chn8.rbg",
			"clt/aux_chn9.rbg",
			"clt/aux_chn10.rbg",
			"clt/aux_chn11.rbg",
			"clt/aux_chn12.rbg",
			"clt/aux_chn13.rbg",
			"clt/aux_chn14.rbg",
			"clt/aux_chn15.rbg"};
//#endif
    const char* result_corr_file =          "clt/aux_corr.corr";
    const char* result_corr_quad_file =     "clt/aux_corr-quad.corr";
    const char* result_corr_td_norm_file =  "clt/aux_corr-td-norm.corr";
///    const char* result_corr_cross_file = "clt/aux_corr-cross.corr";
    const char* result_inter_td_norm_file = "clt/aux_inter-td-norm.corr";
    const char* result_textures_file =      "clt/aux_texture_nodp.rgba";
    const char* result_diff_rgb_combo_file ="clt/aux_diff_rgb_combo_nodp.drbg";
    const char* result_textures_rgba_file = "clt/aux_texture_rgba_nodp.rgba";

    const char* result_textures_file_dp =      "clt/aux_texture_dp.rgba";
    const char* result_diff_rgb_combo_file_dp ="clt/aux_diff_rgb_combo_dp.drbg";
    const char* result_textures_rgba_file_dp = "clt/aux_texture_rgba_dp.rgba";


    const char* rByRDist_file =             "clt/aux.rbyrdist";
    const char* correction_vector_file =    "clt/aux.correction_vector";
    const char* geometry_correction_file =  "clt/aux.geometry_correction";

    float color_weights [] = {
            1.0,              // float             weight0,            // scale for R 0.5 / (1.0 + 0.5 +0.2)
            1.0,              // float             weight1,            // scale for B 0.2 / (1.0 + 0.5 +0.2)
            1.0};             // float             weight2,            // scale for G 1.0 / (1.0 + 0.5 +0.2)
	float generate_RBGA_params[]={
			10.0,                  // float             min_shot,           // 10.0
            3.0,                   // float             scale_shot,         // 3.0
            10.0, // 1.5f,                  // float             diff_sigma,         // pixel value/pixel change
            10.0f,                 // float             diff_threshold,     // pixel value/pixel change
            12.0    // 3.0         // float             min_agree,          // minimal number of channels to agree on a point (real number to work with fuzzy averages)
	};


#else
    const char* kernel_file[] = {
    		"clt/main_chn0_transposed.kernel",
			"clt/main_chn1_transposed.kernel",
			"clt/main_chn2_transposed.kernel",
			"clt/main_chn3_transposed.kernel"};

    const char* kernel_offs_file[] = {
    		"clt/main_chn0_transposed.kernel_offsets",
			"clt/main_chn1_transposed.kernel_offsets",
			"clt/main_chn2_transposed.kernel_offsets",
			"clt/main_chn3_transposed.kernel_offsets"};

    const char* image_files[] = {
    		"clt/main_chn0.bayer",
			"clt/main_chn1.bayer",
			"clt/main_chn2.bayer",
			"clt/main_chn3.bayer"};

    const char* ports_offs_xy_file[] = {
    		"clt/main_chn0.portsxy",
			"clt/main_chn1.portsxy",
			"clt/main_chn2.portsxy",
			"clt/main_chn3.portsxy"};
#ifdef SAVE_CLT
    const char* ports_clt_file[] = { // never referenced
    		"clt/main_chn0.clt",
			"clt/main_chn1.clt",
			"clt/main_chn2.clt",
			"clt/main_chn3.clt"};
#endif
    const char* result_rbg_file[] = {
    		"clt/main_chn0.rbg",
			"clt/main_chn1.rbg",
			"clt/main_chn2.rbg",
			"clt/main_chn3.rbg"};
//#endif
    const char* result_corr_file = "clt/main_corr.corr";
    const char* result_corr_quad_file =  "clt/main_corr-quad.corr";
    const char* result_corr_td_norm_file =  "clt/aux_corr-td-norm.corr";
///    const char* result_corr_cross_file = "clt/main_corr-cross.corr";
    const char* result_inter_td_norm_file =  "clt/aux_inter-td-norm.corr";
    const char* result_textures_file =       "clt/main_texture_nodp.rgba";
    const char* result_diff_rgb_combo_file ="clt/main_diff_rgb_combo_nodp.drbg";
    const char* result_textures_rgba_file = "clt/main_texture_rgba_nodp.rgba";

    const char* result_textures_file_dp =       "clt/main_texture_dp.rgba";
    const char* result_diff_rgb_combo_file_dp = "clt/main_diff_rgb_combo_dp.drbg";
    const char* result_textures_rgba_file_dp =  "clt/main_texture_rgba_dp.rgba";


    const char* rByRDist_file =            "clt/main.rbyrdist";
    const char* correction_vector_file =   "clt/main.correction_vector";
    const char* geometry_correction_file = "clt/main.geometry_correction";

    float color_weights [] = {
            0.294118,              // float             weight0,            // scale for R 0.5 / (1.0 + 0.5 +0.2)
            0.117647,              // float             weight1,            // scale for B 0.2 / (1.0 + 0.5 +0.2)
            0.588235};              // float             weight2,            // scale for G 1.0 / (1.0 + 0.5 +0.2)
	float generate_RBGA_params[]={
			10.0,                  // float             min_shot,           // 10.0
            3.0,                   // float             scale_shot,         // 3.0
            1.5f,                  // float             diff_sigma,         // pixel value/pixel change
            10.0f,                 // float             diff_threshold,     // pixel value/pixel change
            3.0                    // float             min_agree,          // minimal number of channels to agree on a point (real number to work with fuzzy averages)
	};

#endif



	int sel_pairs[4];

#if TEST_LWIR
    // testing with 16 LWIR
    int num_cams =   16;
    int num_colors = 1;
    sel_pairs[0] =   0xffffffff;
    sel_pairs[1] =   0xffffffff;
    sel_pairs[2] =   0xffffffff;
    sel_pairs[3] =   0x00ffffff;
    int num_pairs = 120;
#else
    // testing with quad RGB
    int num_cams =   4;
    int num_colors = 3;
    sel_pairs[0] = 0x3f;
    sel_pairs[1] =   0;
    sel_pairs[2] =   0;
    sel_pairs[3] =   0;
    int num_pairs =  6;
#endif

	int task_size = get_task_size(num_cams); // sizeof(struct tp_task)/sizeof(float) - 6 * (NUM_CAMS - num_cams);

	// FIXME: update to use new correlations and num_cams
    float port_offsets4[4][2] =  {// used only in textures to scale differences
			{-0.5, -0.5},
			{ 0.5, -0.5},
			{-0.5,  0.5},
			{ 0.5,  0.5}};

    float port_offsets[NUM_CAMS][2];
    if (num_cams == 4){
    	for (int ncam = 0; ncam < 4; ncam++){
    		port_offsets[ncam][0]= port_offsets4[ncam][0];
    		port_offsets[ncam][1]= port_offsets4[ncam][1];
    	}
    } else {
		 for (int ncam = 0; ncam < num_cams; ncam++) {
			 double alpha = 2 * M_PI * (ncam) /num_cams; // math.h
			 port_offsets[ncam][0] =  0.5 * sin((alpha));
			 port_offsets[ncam][1] = -0.5 * cos((alpha));
		 }
    }

    int keep_texture_weights = 0; // 1; // try with 0 also
    int texture_colors = num_colors; // 3; // result will be 3+1 RGBA (for mono - 2)

    int KERN_TILES = KERNELS_HOR *  KERNELS_VERT * num_colors; // NUM_COLORS;
    int KERN_SIZE =  KERN_TILES * 4 * 64;

    int CORR_SIZE = (2 * CORR_OUT_RAD + 1) * (2 * CORR_OUT_RAD + 1);




    float            * host_kern_buf =  (float *)malloc(KERN_SIZE * sizeof(float));
// static - see https://stackoverflow.com/questions/20253267/segmentation-fault-before-main
///    static struct tp_task     task_data   [TILESX*TILESY]; // maximal length - each tile
///    static struct tp_task     task_data1  [TILESX*TILESY]; // maximal length - each tile

    float * ftask_data  =  (float *) malloc(TILESX * TILESY * task_size * sizeof(float));
    float * ftask_data1  = (float *) malloc(TILESX * TILESY * task_size * sizeof(float));

    trot_deriv  rot_deriv;
///    int                corr_indices         [NUM_PAIRS*TILESX*TILESY];
    int                texture_indices      [TILESX*TILESYA];
    int                cpu_woi              [4];


    // host array of pointers to GPU memory
    float            * gpu_kernels_h        [num_cams];
    struct CltExtra  * gpu_kernel_offsets_h [num_cams];
    float            * gpu_images_h         [num_cams];
    float              tile_coords_h        [num_cams][TILESX * TILESY][2];
    float            * gpu_clt_h            [num_cams];
    float            * gpu_corr_images_h    [num_cams];

    float            * gpu_corrs;               // correlation tiles (per tile, per pair) in pixel domain
    float            * gpu_corrs_td;            // correlation tiles (per tile, per pair) in transform domain
    int              * gpu_corr_indices;        // shared by gpu_corrs gpu_corrs_td
    float            * gpu_corrs_combo;         // correlation tiles combined (1 per tile), pixel domain
    float            * gpu_corrs_combo_td;      // correlation tiles combined (1 per tile), transform domain
    int              * gpu_corrs_combo_indices; // shared by gpu_corrs_combo and gpu_corrs_combo_td

    float            * gpu_textures;
    float            * gpu_diff_rgb_combo;
    float            * gpu_textures_rbga;
    int              * gpu_texture_indices;
    int              * gpu_woi;
    int              * gpu_num_texture_tiles;
    float            * gpu_port_offsets;
    float            * gpu_color_weights;
    float            * gpu_generate_RBGA_params;
    int                num_corrs;
    int                num_textures;
/// int                num_ports = NUM_CAMS;
    // GPU pointers to GPU pointers to memory
    float           ** gpu_kernels; //           [NUM_CAMS];
    struct CltExtra ** gpu_kernel_offsets; //    [NUM_CAMS];
    float           ** gpu_images; //            [NUM_CAMS];
    float           ** gpu_clt;    //            [NUM_CAMS];
    float           ** gpu_corr_images; //       [NUM_CAMS];



    // GPU pointers to GPU memory
///    struct tp_task  * gpu_tasks;  // TODO: ***** remove ! **** DONE
    float *           gpu_ftasks; // TODO: ***** allocate ! **** DONE
    int *             gpu_active_tiles;
    int *             gpu_num_active;
    int *             gpu_num_corr_tiles;

    checkCudaErrors (cudaMalloc((void **)&gpu_active_tiles, TILESX * TILESY * sizeof(int)));
    checkCudaErrors (cudaMalloc((void **)&gpu_num_active,                     sizeof(int)));
    checkCudaErrors (cudaMalloc((void **)&gpu_num_corr_tiles,                 sizeof(int)));

    size_t  dstride;               // in bytes !
    size_t  dstride_rslt;          // in bytes !
    size_t  dstride_corr;          // in bytes ! for one 2d phase correlation (padded 15x15x4 bytes)
    // in the future, dstride_corr can reuse that of dstride_corr_td?
    size_t  dstride_corr_td;       // in bytes ! for one 2d phase correlation (padded 4x8x8x4 bytes)
    size_t  dstride_corr_combo;    // in bytes ! for one 2d phase correlation (padded 15x15x4 bytes)
    size_t  dstride_corr_combo_td; // in bytes ! for one 2d phase correlation (padded 4x8x8x4 bytes)

    size_t  dstride_textures; // in bytes ! for one rgba/ya 16x16 tile
    size_t  dstride_textures_rbga; // in bytes ! for one rgba/ya 16x16 tile

    struct gc fgeometry_correction;
    float*  correction_vector;
    int     correction_vector_length;
    float * rByRDist;
    int     rByRDist_length;

    struct gc          * gpu_geometry_correction;
    struct corr_vector * gpu_correction_vector;
    float              * gpu_rByRDist;
    trot_deriv   * gpu_rot_deriv;

    readFloatsFromFile(
    		(float *) &fgeometry_correction, // float * data, // allocated array
			geometry_correction_file); // 			   char *  path) // file path

    rByRDist = readAllFloatsFromFile(
    		rByRDist_file, // const char *  path,
    		&rByRDist_length); // int * len_in_floats)
    correction_vector =  readAllFloatsFromFile(
    		correction_vector_file, // const char *  path,
    		&correction_vector_length); // int * len_in_floats)

    gpu_geometry_correction =  (struct gc *) copyalloc_kernel_gpu(
    		(float *) &fgeometry_correction,
    		sizeof(fgeometry_correction)/sizeof(float));

    gpu_correction_vector =  (struct corr_vector * ) copyalloc_kernel_gpu(
    		correction_vector,
			correction_vector_length);

    gpu_rByRDist =  copyalloc_kernel_gpu(
    		rByRDist,
			rByRDist_length);

    checkCudaErrors(cudaMalloc((void **)&gpu_rot_deriv, sizeof(trot_deriv)));

///    for (int ncam = 0; ncam < NUM_CAMS; ncam++) {
    for (int ncam = 0; ncam < num_cams; ncam++) {
        readFloatsFromFile(
        		host_kern_buf, // float * data, // allocated array
				kernel_file[ncam]); // 			   char *  path) // file path
        gpu_kernels_h[ncam] = copyalloc_kernel_gpu(host_kern_buf, KERN_SIZE);

        readFloatsFromFile(
        		host_kern_buf, // float * data, // allocated array
				kernel_offs_file[ncam]); // 			   char *  path) // file path
        gpu_kernel_offsets_h[ncam] = (struct CltExtra *) copyalloc_kernel_gpu(
        		host_kern_buf,
				KERN_TILES * (sizeof( struct CltExtra)/sizeof(float)));
        // will get results back
        gpu_clt_h[ncam] = alloc_kernel_gpu(TILESY * TILESX * num_colors * 4 * DTT_SIZE * DTT_SIZE);
        printf("Allocating GPU memory, 0x%x floats\n", (TILESY * TILESX * num_colors * 4 * DTT_SIZE * DTT_SIZE)) ;
        // allocate result images (3x height to accommodate 3 colors

        // Image is extended by 4 pixels each side to avoid checking (mclt tiles extend by 4)
        //host array of pointers to GPU arrays
        gpu_corr_images_h[ncam] = alloc_image_gpu(
        		&dstride_rslt,                // size_t* dstride, // in bytes!!
				IMG_WIDTH + DTT_SIZE,         // int width,
//				3*(IMG_HEIGHT + DTT_SIZE));   // int height);
				num_colors*(IMG_HEIGHT + DTT_SIZE));   // int height);
    }
    // allocates one correlation kernel per line (15x15 floats), number of rows - number of tiles * number of pairs
    gpu_corrs = alloc_image_gpu(
    		&dstride_corr,                  // in bytes ! for one 2d phase correlation (padded 15x15x4 bytes)
			CORR_SIZE,                      // int width,
			num_pairs * TILESX * TILESY);   // int height);
    // read channel images (assuming host_kern_buf size > image size, reusing it)
// allocate all other correlation data, some may be
    gpu_corrs_td = alloc_image_gpu(
    		&dstride_corr_td,               // in bytes ! for one 2d phase correlation (padded 15x15x4 bytes)
			4 * DTT_SIZE * DTT_SIZE,         // int width,
			num_pairs * TILESX * TILESY);    // int height);

    gpu_corrs_combo = alloc_image_gpu(
    		&dstride_corr_combo,             // in bytes ! for one 2d phase correlation (padded 15x15x4 bytes)
			CORR_SIZE,                       // int width,
			TILESX * TILESY);                // int height);

    gpu_corrs_combo_td = alloc_image_gpu(
    		&dstride_corr_combo_td,          // in bytes ! for one 2d phase correlation (padded 15x15x4 bytes)
			4 * DTT_SIZE * DTT_SIZE,         // int width,
			TILESX * TILESY);                // int height);


//    for (int ncam = 0; ncam < NUM_CAMS; ncam++) {
    for (int ncam = 0; ncam < num_cams; ncam++) {
        readFloatsFromFile(
        		host_kern_buf, // float * data, // allocated array
				image_files[ncam]); // 			   char *  path) // file path
        gpu_images_h[ncam] =  copyalloc_image_gpu(
        		host_kern_buf, // float * image_host,
				&dstride,      // size_t* dstride,
				IMG_WIDTH,     // int width,
				IMG_HEIGHT);   // int height);
    }
//#define DBG_TILE  (174*324 +118)

//  for (int ncam = 0; ncam < NUM_CAMS; ncam++) {
    for (int ncam = 0; ncam < num_cams; ncam++) {
        readFloatsFromFile(
			    (float *) &tile_coords_h[ncam],
				ports_offs_xy_file[ncam]); // 			   char *  path) // file path
    }

    for (int ty = 0; ty < TILESY; ty++){
        for (int tx = 0; tx < TILESX; tx++){
            int nt = ty * TILESX + tx;
//            int task_task = 0xf | (((1 << NUM_PAIRS)-1) << TASK_CORR_BITS);
            int task_task = 0xf | (1 << TASK_CORR_BITS); // just 1 bit, correlation selection is defined by common corr_sel bits
            int task_txy = tx + (ty << 16);
            float task_target_disparity = DBG_DISPARITY;
            float * tp = ftask_data + task_size * nt;
            *(tp++) =  *(float *) &task_task;
            *(tp++) =  *(float *) &task_txy;
            *(tp++) =  task_target_disparity;
            tp += 2; // skip centerX, centerY
            for (int ncam = 0; ncam < num_cams; ncam++) {
            	*(tp++) = tile_coords_h[ncam][nt][0];
            	*(tp++) = tile_coords_h[ncam][nt][1];
            }
        }
    }

    int tp_task_size =  TILESX * TILESY; // sizeof(ftask_data)/sizeof(float)/task_size; // number of task tiles
    int num_active_tiles; // will be calculated by convert_direct
	int rslt_corr_size;
	int corr_img_size;

    gpu_ftasks = (float  *) copyalloc_kernel_gpu(ftask_data, tp_task_size * task_size); // (sizeof(struct tp_task)/sizeof(float)));

    // just allocate
    checkCudaErrors (cudaMalloc((void **)&gpu_corr_indices,        num_pairs * TILESX * TILESY*sizeof(int)));
    checkCudaErrors (cudaMalloc((void **)&gpu_corrs_combo_indices,             TILESX * TILESY*sizeof(int)));
    num_textures = 0;
    for (int ty = 0; ty < TILESY; ty++){
    	for (int tx = 0; tx < TILESX; tx++){
    		int nt = ty * TILESX + tx;
            float *tp = ftask_data + task_size * nt;
    		int cm = (*(int *) tp) & TASK_TEXTURE_BITS;
    		if (cm){
    			texture_indices[num_textures++] = (nt << CORR_NTILE_SHIFT) | (1 << LIST_TEXTURE_BIT);
    		}
    	}
    }


    // num_textures now has the total number of textures
    // copy corr_indices to gpu
    gpu_texture_indices = (int  *) copyalloc_kernel_gpu(
    		(float * ) texture_indices,
			num_textures,
			TILESX * TILESYA); // number of rows - multiple of 4
    // just allocate
    checkCudaErrors(cudaMalloc((void **)&gpu_woi,               4 * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&gpu_num_texture_tiles, 8 * sizeof(float))); // for each subsequence - number of non-border,
    // number of border tiles

    // copy port indices to gpu

	gpu_port_offsets =         (float *) copyalloc_kernel_gpu((float * ) port_offsets, num_cams * 2); // num_ports * 2);
    gpu_color_weights =        (float *) copyalloc_kernel_gpu((float * ) color_weights, sizeof(color_weights));
    gpu_generate_RBGA_params = (float *) copyalloc_kernel_gpu((float * ) generate_RBGA_params, sizeof(generate_RBGA_params));

///    int tile_texture_size = (texture_colors + 1 + (keep_texture_weights? (NUM_CAMS + texture_colors + 1): 0)) *256;
    int tile_texture_layers = (texture_colors + 1 + (keep_texture_weights? (num_cams + texture_colors + 1): 0));
    int tile_texture_size = tile_texture_layers *256;

    gpu_textures = alloc_image_gpu(
    		&dstride_textures,              // in bytes ! for one rgba/ya 16x16 tile
			tile_texture_size,              // int width (floats),
			TILESX * TILESY);               // int height);

    int rgba_width =   (TILESX+1) * DTT_SIZE;
    int rgba_height =  (TILESY+1) * DTT_SIZE;
    int rbga_slices =  texture_colors + 1; // 4/1

    gpu_textures_rbga = alloc_image_gpu(
    		&dstride_textures_rbga,              // in bytes ! for one rgba/ya 16x16 tile
			rgba_width,              // int width (floats),
			rgba_height * rbga_slices);               // int height);
///    checkCudaErrors(cudaMalloc((void **)&gpu_diff_rgb_combo,  TILESX * TILESY * NUM_CAMS * (NUM_COLORS + 1) * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&gpu_diff_rgb_combo,  TILESX * TILESY * num_cams * (num_colors + 1) * sizeof(float)));

    // Now copy arrays of per-camera pointers to GPU memory to GPU itself
    gpu_kernels =        copyalloc_pointers_gpu (gpu_kernels_h,     num_cams); // NUM_CAMS);
    gpu_kernel_offsets = (struct CltExtra **) copyalloc_pointers_gpu ((float **) gpu_kernel_offsets_h, num_cams); // NUM_CAMS);
    gpu_images =         copyalloc_pointers_gpu (gpu_images_h,      num_cams); // NUM_CAMS);
    gpu_clt =            copyalloc_pointers_gpu (gpu_clt_h,         num_cams); // NUM_CAMS);
    gpu_corr_images =    copyalloc_pointers_gpu (gpu_corr_images_h, num_cams); // NUM_CAMS);

#ifdef DBG_TILE
    const int numIterations = 1; //0;
    const int i0 =  0; // -1;
#else
    const int numIterations = 10; // 0; //0;
    const int i0 = -1; // 0; // -1;
#endif

	int corr_size =        2 * CORR_OUT_RAD + 1;
	int num_tiles = tp_task_size; // TILESX * TILESYA; //Was this on 01/22/2022
	int num_corr_indices = num_pairs * num_tiles;

	float * corr_img; //  = (float *)malloc(corr_img_size * sizeof(float));
	float * cpu_corr; //  = (float *)malloc(rslt_corr_size * sizeof(float));
	int * cpu_corr_indices; //  = (int *) malloc(num_corr_indices * sizeof(int));




#define TEST_ROT_MATRICES
#ifdef  TEST_ROT_MATRICES
    dim3 threads_rot(3,3,3);
///    dim3 grid_rot   (NUM_CAMS, 1, 1);
    dim3 grid_rot   (num_cams, 1, 1);

    printf("ROT_MATRICES: threads_list=(%d, %d, %d)\n",threads_rot.x,threads_rot.y,threads_rot.z);
    printf("ROT_MATRICES: grid_list=(%d, %d, %d)\n",grid_rot.x,grid_rot.y,grid_rot.z);
    StopWatchInterface *timerROT_MATRICES = 0;
    sdkCreateTimer(&timerROT_MATRICES);
    for (int i = i0; i < numIterations; i++)
    {
    	if (i == 0)
    	{
    		checkCudaErrors(cudaDeviceSynchronize());
    		sdkResetTimer(&timerROT_MATRICES);
    		sdkStartTimer(&timerROT_MATRICES);
    	}

    	calc_rot_deriv<<<grid_rot,threads_rot>>> (
    			num_cams,                // int                  num_cams,
    			gpu_correction_vector ,  // struct corr_vector * gpu_correction_vector,
    			gpu_rot_deriv);          // union trot_deriv   * gpu_rot_deriv);


    	getLastCudaError("Kernel failure");
    	checkCudaErrors(cudaDeviceSynchronize());
    	printf("test pass: %d\n",i);
    }
    ///	cudaProfilerStop();
    sdkStopTimer(&timerROT_MATRICES);
    float avgTimeROT_MATRICES = (float)sdkGetTimerValue(&timerROT_MATRICES) / (float)numIterations;
    sdkDeleteTimer(&timerROT_MATRICES);
    printf("Average calc_rot_matrices run time =%f ms\n",  avgTimeROT_MATRICES);
	checkCudaErrors(cudaMemcpy(
			&rot_deriv,
			gpu_rot_deriv,
			sizeof(trot_deriv),
			cudaMemcpyDeviceToHost));

#endif // TEST_ROT_MATRICES

#define TEST_REVERSE_DISTORTIONS
#ifdef  TEST_REVERSE_DISTORTIONS
    dim3 threads_rd(3,3,3);
    dim3 grid_rd   (NUM_CAMS, 1, 1); // can get rid of NUM_CAMS
//    dim3 grid_rd   (num_cams, 1, 1);

    printf("REVERSE DISTORTIONS: threads_list=(%d, %d, %d)\n",threads_rd.x,threads_rd.y,threads_rd.z);
    printf("REVERSE DISTORTIONS: grid_list=(%d, %d, %d)\n",grid_rd.x,grid_rd.y,grid_rd.z);
    StopWatchInterface *timerREVERSE_DISTORTIONS = 0;
    sdkCreateTimer(&timerREVERSE_DISTORTIONS);
    for (int i = i0; i < numIterations; i++)
    {
    	if (i == 0)
    	{
    		checkCudaErrors(cudaDeviceSynchronize());
    		sdkResetTimer(&timerREVERSE_DISTORTIONS);
    		sdkStartTimer(&timerREVERSE_DISTORTIONS);
    	}

    	calcReverseDistortionTable<<<grid_rd,threads_rd>>>(
    			gpu_geometry_correction, // 		struct gc          * gpu_geometry_correction,
				gpu_rByRDist);


    	getLastCudaError("Kernel failure");
    	checkCudaErrors(cudaDeviceSynchronize());
    	printf("test pass: %d\n",i);
    }
    ///	cudaProfilerStop();
    sdkStopTimer(&timerREVERSE_DISTORTIONS);
    float avgTimeREVERSE_DISTORTIONS = (float)sdkGetTimerValue(&timerREVERSE_DISTORTIONS) / (float)numIterations;
    sdkDeleteTimer(&timerREVERSE_DISTORTIONS);
    printf("Average calcReverseDistortionTable  run time =%f ms\n",  avgTimeREVERSE_DISTORTIONS);

    float * rByRDist_gen = (float *) malloc(RBYRDIST_LEN * sizeof(float));
	checkCudaErrors(cudaMemcpy(
			rByRDist_gen,
			gpu_rByRDist,
			RBYRDIST_LEN * sizeof(float),
			cudaMemcpyDeviceToHost));
	float max_err = 0;
	for (int i = 0; i < RBYRDIST_LEN; i++){
		float err = abs(rByRDist_gen[i] - rByRDist[i]);
		if (err > max_err){
			max_err = err;
		}
#ifdef VERBOSE
///		printf ("%5d: %8.6f %8.6f %f %f\n", i, rByRDist[i], rByRDist_gen[i] , err, max_err);
#endif // #ifdef VERBOSE
	}
	printf("Maximal rByRDist error = %f\n",max_err);
	free (rByRDist_gen);
#if 0
    // temporarily restore
    checkCudaErrors(cudaMemcpy(
    		gpu_rByRDist,
			rByRDist,
			RBYRDIST_LEN * sizeof(float),
            cudaMemcpyHostToDevice));
#endif // #if 1

#endif // TEST_REVERSE_DISTORTIONS




#define TEST_GEOM_CORR
#ifdef  TEST_GEOM_CORR
///    dim3 threads_geom(NUM_CAMS,TILES_PER_BLOCK_GEOM, 1);
    dim3 threads_geom(num_cams,TILES_PER_BLOCK_GEOM, 1);
    dim3 grid_geom   ((tp_task_size+TILES_PER_BLOCK_GEOM-1)/TILES_PER_BLOCK_GEOM, 1, 1);
    printf("GEOM: threads_list=(%d, %d, %d)\n",threads_geom.x,threads_geom.y,threads_geom.z);
    printf("GEOM: grid_list=(%d, %d, %d)\n",grid_geom.x,grid_geom.y,grid_geom.z);
    StopWatchInterface *timerGEOM = 0;
    sdkCreateTimer(&timerGEOM);
    for (int i = i0; i < numIterations; i++)
    {
    	if (i == 0)
    	{
    		checkCudaErrors(cudaDeviceSynchronize());
    		sdkResetTimer(&timerGEOM);
    		sdkStartTimer(&timerGEOM);
    	}
/*
    	get_tiles_offsets<<<grid_geom,threads_geom>>> (
    			num_cams,                // int                  num_cams,
    			gpu_tasks,               // struct tp_task     * gpu_tasks,
				tp_task_size,            // int                  num_tiles,          // number of tiles in task list
				gpu_geometry_correction, //	struct gc          * gpu_geometry_correction,
				gpu_correction_vector,   //	struct corr_vector * gpu_correction_vector,
				gpu_rByRDist,            //	float *              gpu_rByRDist)      // length should match RBYRDIST_LEN
				gpu_rot_deriv);          // union trot_deriv   * gpu_rot_deriv);
				*/
    	calculate_tiles_offsets<<<1,1>>> (
    			1,                       // int                  uniform_grid, //==0: use provided centers (as for interscene) , !=0 calculate uniform grid
    			num_cams,                // int                  num_cams,
				gpu_ftasks,              // float              * gpu_ftasks,         // flattened tasks, 27 floats for quad EO, 99 floats for LWIR16
//    			gpu_tasks,               // struct tp_task     * gpu_tasks,
				tp_task_size,            // int                  num_tiles,          // number of tiles in task list
				gpu_geometry_correction, //	struct gc          * gpu_geometry_correction,
				gpu_correction_vector,   //	struct corr_vector * gpu_correction_vector,
				gpu_rByRDist,            //	float *              gpu_rByRDist)      // length should match RBYRDIST_LEN
				gpu_rot_deriv);          // union trot_deriv   * gpu_rot_deriv);

    	getLastCudaError("Kernel failure");
    	checkCudaErrors(cudaDeviceSynchronize());
    	printf("test pass: %d\n",i);
    }
    ///	cudaProfilerStop();
    sdkStopTimer(&timerGEOM);
    float avgTimeGEOM = (float)sdkGetTimerValue(&timerGEOM) / (float)numIterations;
    sdkDeleteTimer(&timerGEOM);
    printf("Average TextureList run time =%f ms\n",  avgTimeGEOM);
/*
	checkCudaErrors(cudaMemcpy( // copy modified/calculated tasks
			&task_data1,
			gpu_tasks,
			tp_task_size * sizeof(struct tp_task),
			cudaMemcpyDeviceToHost));
*/
	checkCudaErrors(cudaMemcpy( // copy modified/calculated tasks
			ftask_data1,
			gpu_ftasks,
			tp_task_size * task_size *sizeof(float),
			cudaMemcpyDeviceToHost));

//task_size
#if 0 // for manual browsing
	struct tp_task * old_task = &task_data [DBG_TILE];
	struct tp_task * new_task = &task_data1[DBG_TILE];
#endif
#ifdef 	DBG_TILE
    printf("old_task txy = 0x%x\n", *(int *) (ftask_data + task_size * DBG_TILE + 1)) ; // task_data [DBG_TILE].txy);
    printf("new_task txy = 0x%x\n", *(int *) (ftask_data1 + task_size * DBG_TILE + 1)) ; // task_data1[DBG_TILE].txy);
        for (int ncam = 0; ncam < num_cams; ncam++){
            printf("camera %d pX old %f new %f diff = %f\n", ncam,
            		 *(ftask_data  + task_size * DBG_TILE + tp_task_xy_offset + 2*ncam + 0),
            		 *(ftask_data1 + task_size * DBG_TILE + tp_task_xy_offset + 2*ncam + 0),
            		 (*(ftask_data + task_size * DBG_TILE + tp_task_xy_offset + 2*ncam + 0)) -
            		 (*(ftask_data1 + task_size * DBG_TILE + tp_task_xy_offset + 2*ncam + 0)));
            printf("camera %d pY old %f new %f diff = %f\n", ncam,
           		 *(ftask_data  + task_size * DBG_TILE + tp_task_xy_offset + 2*ncam + 1),
           		 *(ftask_data1 + task_size * DBG_TILE + tp_task_xy_offset + 2*ncam + 1),
           		 (*(ftask_data + task_size * DBG_TILE + tp_task_xy_offset + 2*ncam + 1)) -
           		 (*(ftask_data1 + task_size * DBG_TILE + tp_task_xy_offset + 2*ncam + 1)));
        }
#endif //#ifdef 	DBG_TILE
#endif // TEST_GEOM_CORR


    //create and start CUDA timer
    StopWatchInterface *timerTP = 0;
    sdkCreateTimer(&timerTP);

    dim3 threads_tp(1, 1, 1);
    dim3 grid_tp(1, 1, 1);
    printf("threads_tp=(%d, %d, %d)\n",threads_tp.x,threads_tp.y,threads_tp.z);
    printf("grid_tp=   (%d, %d, %d)\n",grid_tp.x,   grid_tp.y,   grid_tp.z);

    ///    cudaProfilerStart();
    float ** fgpu_kernel_offsets = (float **) gpu_kernel_offsets; //  [num_cams]  [NUM_CAMS];

    for (int i = i0; i < numIterations; i++)
    {
    	if (i == 0)
    	{
    		checkCudaErrors(cudaDeviceSynchronize());
    		sdkResetTimer(&timerTP);
    		sdkStartTimer(&timerTP);
    	}
    	convert_direct<<<grid_tp,threads_tp>>>( // called with a single block, CONVERT_DIRECT_INDEXING_THREADS threads
    			num_cams,              // int                num_cams,           // actual number of cameras
				num_colors,            // int                num_colors,         // actual number of colors: 3 for RGB, 1 for LWIR/mono
    			fgpu_kernel_offsets,   // struct CltExtra ** gpu_kernel_offsets,
				gpu_kernels,           // float           ** gpu_kernels,
				gpu_images,            // float           ** gpu_images,
				gpu_ftasks,            // float            * gpu_ftasks,         // flattened tasks, 27 floats for quad EO, 99 floats for LWIR16
				gpu_clt,               // float           ** gpu_clt,            // [num_cams][TILESY][TILESX][num_colors][DTT_SIZE*DTT_SIZE]
				dstride/sizeof(float), // size_t             dstride, // for gpu_images
				tp_task_size,          // int                num_tiles) // number of tiles in task
				0,                     // int                lpf_mask)            // apply lpf to colors : bit 0 - red, bit 1 - blue, bit2 - green
				IMG_WIDTH,             // int                woi_width,
				IMG_HEIGHT,            // int                woi_height,
				KERNELS_HOR,           // int                kernels_hor,
				KERNELS_VERT,          // int                kernels_vert);
				gpu_active_tiles,      // int *              gpu_active_tiles,      // pointer to the calculated number of non-zero tiles
    			gpu_num_active, //);       // int *              pnum_active_tiles);  //  indices to gpu_tasks
				TILESX); // int                tilesx)


    	getLastCudaError("Kernel execution failed");
    	checkCudaErrors(cudaDeviceSynchronize());
//    	printf("%d\n",i);
    }
    sdkStopTimer(&timerTP);
    float avgTime = (float)sdkGetTimerValue(&timerTP) / (float)numIterations;
    sdkDeleteTimer(&timerTP);

	checkCudaErrors(cudaMemcpy(
			&num_active_tiles,
			gpu_num_active,
			sizeof(int),
			cudaMemcpyDeviceToHost));
    printf("Run time =%f ms, num active tiles = %d\n",  avgTime, num_active_tiles);


#ifdef SAVE_CLT
    int rslt_size = (TILESY * TILESX * num_colors * 4 * DTT_SIZE * DTT_SIZE);
    float * cpu_clt = (float *)malloc(rslt_size*sizeof(float));
//  for (int ncam = 0; ncam < NUM_CAMS; ncam++) {
    for (int ncam = 0; ncam < num_cams; ncam++) {
    	checkCudaErrors(cudaMemcpy( // segfault
    			cpu_clt,
				gpu_clt_h[ncam],
				rslt_size * sizeof(float),
    			cudaMemcpyDeviceToHost));
        printf("Writing CLT data to %s\n",  ports_clt_file[ncam]);
    	writeFloatsToFile(cpu_clt, // float *       data, // allocated array
    			rslt_size, // int           size, // length in elements
				ports_clt_file[ncam]); // 			   const char *  path) // file path
    }
#endif

#ifdef TEST_IMCLT
     {
    	// testing imclt
    	dim3 threads_imclt(IMCLT_THREADS_PER_TILE, IMCLT_TILES_PER_BLOCK, 1);
    	dim3 grid_imclt(1,1,1);
    	printf("threads_imclt=(%d, %d, %d)\n",threads_imclt.x,threads_imclt.y,threads_imclt.z);
    	printf("grid_imclt=   (%d, %d, %d)\n",grid_imclt.x,   grid_imclt.y,   grid_imclt.z);
//      for (int ncam = 0; ncam < NUM_CAMS; ncam++) {
   	    for (int ncam = 0; ncam < num_cams; ncam++) {
    		test_imclt<<<grid_imclt,threads_imclt>>>(
    				gpu_clt_h[ncam], // ncam]); //                //       float           ** gpu_clt,            // [num_cams][TILESY][TILESX][num_colors][DTT_SIZE*DTT_SIZE]
					ncam);                                        // int             ncam); // just for debug print
    	}
    	getLastCudaError("Kernel execution failed");
    	checkCudaErrors(cudaDeviceSynchronize());
    	printf("test_imclt() DONE\n");
    }
#endif

    StopWatchInterface *timerIMCLT = 0;
    sdkCreateTimer(&timerIMCLT);

    for (int i = i0; i < numIterations; i++)
    {
    	if (i == 0)
    	{
    		checkCudaErrors(cudaDeviceSynchronize());
    		sdkResetTimer(&timerIMCLT);
    		sdkStartTimer(&timerIMCLT);
    	}
        dim3 threads_imclt_all(1, 1, 1);
		dim3 grid_imclt_all(1, 1, 1);
        printf("threads_imclt_all=(%d, %d, %d)\n",threads_imclt_all.x,threads_imclt_all.y,threads_imclt_all.z);
        printf("grid_imclt_all=   (%d, %d, %d)\n",grid_imclt_all.x,   grid_imclt_all.y,   grid_imclt_all.z);
        imclt_rbg_all<<<grid_imclt_all,threads_imclt_all>>>(
        		num_cams,                    // int                num_cams,
        		gpu_clt,                     // float           ** gpu_clt,            // [num_cams][TILESY][TILESX][num_colors][DTT_SIZE*DTT_SIZE]
				gpu_corr_images,             // float           ** gpu_corr_images,    // [num_cams][WIDTH, 3 * HEIGHT]
				1,                           // int               apply_lpf,
				num_colors,                  // int               colors,               // defines lpf filter
				TILESX,                      // int               woi_twidth,
				TILESY,                      // int               woi_theight,
				dstride_rslt/sizeof(float)); // const size_t      dstride);            // in floats (pixels)
    	getLastCudaError("Kernel failure");
    	checkCudaErrors(cudaDeviceSynchronize());
    	printf("test pass: %d\n",i);
    }

    // TODO: *** Stop here for initial testing ***

    sdkStopTimer(&timerIMCLT);
    float avgTimeIMCLT = (float)sdkGetTimerValue(&timerIMCLT) / (float)numIterations;
    sdkDeleteTimer(&timerIMCLT);
    printf("Average IMCLT run time =%f ms\n",  avgTimeIMCLT);

    int rslt_img_size =       num_colors * (IMG_HEIGHT + DTT_SIZE) * (IMG_WIDTH + DTT_SIZE);
    float * cpu_corr_image = (float *)malloc(rslt_img_size * sizeof(float));



//  for (int ncam = 0; ncam < NUM_CAMS; ncam++) {
    for (int ncam = 0; ncam < num_cams; ncam++) {
    	checkCudaErrors(cudaMemcpy2D( // segfault
    			cpu_corr_image,
				(IMG_WIDTH + DTT_SIZE) * sizeof(float),
				gpu_corr_images_h[ncam],
				dstride_rslt,
				(IMG_WIDTH + DTT_SIZE) * sizeof(float),
//				3* (IMG_HEIGHT + DTT_SIZE),
				num_colors* (IMG_HEIGHT + DTT_SIZE),
    			cudaMemcpyDeviceToHost));
        printf("Writing RBG data to %s\n",  result_rbg_file[ncam]);
    	writeFloatsToFile( // will have margins
    			cpu_corr_image, // float *       data, // allocated array
				rslt_img_size, // int           size, // length in elements
				result_rbg_file[ncam]); // 			   const char *  path) // file path
    }

    free(cpu_corr_image);


#ifndef NOCORR
//    cudaProfilerStart();
// testing corr
    StopWatchInterface *timerCORR = 0;
    sdkCreateTimer(&timerCORR);

    for (int i = i0; i < numIterations; i++)
    {
    	if (i == 0)
    	{
    		checkCudaErrors(cudaDeviceSynchronize());
    		sdkResetTimer(&timerCORR);
    		sdkStartTimer(&timerCORR);
    	}
    	correlate2D<<<1,1>>>(
    			num_cams,                      // int               num_cams,
				sel_pairs[0], // int               sel_pairs0           // unused bits should be 0
				sel_pairs[1], // int               sel_pairs1,           // unused bits should be 0
				sel_pairs[2], // int               sel_pairs2,           // unused bits should be 0
				sel_pairs[3], // int               sel_pairs3,           // unused bits should be 0
				gpu_clt,                    // float          ** gpu_clt,            // [NUM_CAMS] ->[TILESY][TILESX][NUM_COLORS][DTT_SIZE*DTT_SIZE]
				num_colors,                 // int               colors,             // number of colors (3/1)
				color_weights[0], // 0.25,  // float             scale0,             // scale for R
				color_weights[1], // 0.25,  // float             scale1,             // scale for B
				color_weights[2], // 0.5,   // float             scale2,             // scale for G
				30.0 * 30.0,                // float             fat_zero2,           // here - absolute
				gpu_ftasks,                 // float            * gpu_ftasks,         // flattened tasks, 27 floats for quad EO, 99 floats for LWIR16
				tp_task_size,               // int               num_tiles) // number of tiles in task
				TILESX,                     // int               tilesx,             // number of tile rows
				gpu_corr_indices,           // int             * gpu_corr_indices,   // packed tile+pair
				gpu_num_corr_tiles,         // int             * pnum_corr_tiles,    // pointer to a number of correlation tiles to process
				dstride_corr/sizeof(float), // const size_t      corr_stride,        // in floats
				CORR_OUT_RAD,               // int               corr_radius,        // radius of the output correlation (7 for 15x15)
				gpu_corrs);                 // float           * gpu_corrs);          // correlation output data

    	getLastCudaError("Kernel failure");
    	checkCudaErrors(cudaDeviceSynchronize());
    	printf("test pass: %d\n",i);
    }

    sdkStopTimer(&timerCORR);
    float avgTimeCORR = (float)sdkGetTimerValue(&timerCORR) / (float)numIterations;
    sdkDeleteTimer(&timerCORR);
    //    printf("Average CORR run time =%f ms, num cor tiles (old) = %d\n",  avgTimeCORR, num_corrs);

    checkCudaErrors(cudaMemcpy(
    		&num_corrs,
			gpu_num_corr_tiles,
			sizeof(int),
			cudaMemcpyDeviceToHost));
    printf("Average CORR run time =%f ms, num cor tiles (new) = %d\n",  avgTimeCORR, num_corrs);

//    int corr_size =        2 * CORR_OUT_RAD + 1;
//    int rslt_corr_size =   num_corrs * corr_size * corr_size;
//    float * cpu_corr = (float *)malloc(rslt_corr_size * sizeof(float));

	rslt_corr_size =   num_corrs * corr_size * corr_size;
	corr_img_size = num_corr_indices * 16*16; // NAN
	corr_img = (float *)malloc(corr_img_size * sizeof(float));
	cpu_corr = (float *)malloc(rslt_corr_size * sizeof(float));
	cpu_corr_indices = (int *) malloc(num_corr_indices * sizeof(int));


    checkCudaErrors(cudaMemcpy2D(
    		cpu_corr,
			(corr_size * corr_size) * sizeof(float),
			gpu_corrs,
			dstride_corr,
			(corr_size * corr_size) * sizeof(float),
			num_corrs,
			cudaMemcpyDeviceToHost));
    //    checkCudaErrors (cudaMalloc((void **)&gpu_corr_indices,        num_pairs * TILESX * TILESY*sizeof(int)));
//    int num_tiles = TILESX * TILESYA;
//   int num_corr_indices = num_pairs * num_tiles;
//    int * cpu_corr_indices = (int *) malloc(num_corr_indices * sizeof(int));
    checkCudaErrors(cudaMemcpy(
    		cpu_corr_indices,
			gpu_corr_indices,
			num_corr_indices * sizeof(int),
			cudaMemcpyDeviceToHost));
//    int corr_img_size = num_corr_indices * 16*16; // NAN
//    float * corr_img = (float *)malloc(corr_img_size * sizeof(float));
    for (int i = 0; i < corr_img_size; i++){
    	corr_img[i] = NAN;
    }
    for (int ict = 0; ict < num_corr_indices; ict++){
    	//    	int ct = cpu_corr_indices[ict];
    	int ctt = ( cpu_corr_indices[ict] >>  CORR_NTILE_SHIFT);
    	int cpair = cpu_corr_indices[ict] & ((1 << CORR_NTILE_SHIFT) - 1);
    	int ty = ctt / TILESX;
    	int tx = ctt % TILESX;
    	//		int src_offs0 = ict * num_pairs * corr_size * corr_size;
    	int src_offs0 = ict * corr_size * corr_size;
    	int dst_offs0 = cpair * (num_tiles * 16 * 16) +  (ty * 16 * TILESX * 16) + (tx * 16);

    	for (int iy = 0; iy < corr_size; iy++){
    		int src_offs = src_offs0 + iy * corr_size; // ict * num_pairs * corr_size * corr_size;
    		int dst_offs = dst_offs0 + iy * (TILESX * 16);
    		for (int ix = 0; ix < corr_size; ix++){
    			corr_img[dst_offs++] = cpu_corr[src_offs++];
    		}
    	}
    }
    // num_pairs

#ifndef NSAVE_CORR
    printf("Writing phase correlation data to %s, width = %d, height=%d, slices=%d, length=%ld bytes\n",
    		result_corr_file, (TILESX*16),(TILESYA*16), num_pairs, (corr_img_size * sizeof(float)) ) ;
    /*
    		writeFloatsToFile(
    				cpu_corr,    // float *       data, // allocated array
					rslt_corr_size,    // int           size, // length in elements
					result_corr_file); // 			   const char *  path) // file path
     */
    writeFloatsToFile(
    		corr_img,    // float *       data, // allocated array
			corr_img_size,    // int           size, // length in elements
			result_corr_file); // 			   const char *  path) // file path
#endif
    free (cpu_corr);
    free (cpu_corr_indices);
    free (corr_img);
#endif // ifndef NOCORR

#ifndef NOCORR_TD
//    cudaProfilerStart();
// testing corr
    StopWatchInterface *timerCORRTD = 0;
    sdkCreateTimer(&timerCORRTD);
    int num_corr_combo;
    for (int i = i0; i < numIterations; i++)
    {
    	if (i == 0)
    	{
    		checkCudaErrors(cudaDeviceSynchronize());
    		sdkResetTimer(&timerCORRTD);
    		sdkStartTimer(&timerCORRTD);
    	}
    	// FIXME: provide sel_pairs
        correlate2D<<<1,1>>>( // output TD tiles, no normalization
        		num_cams,                      // int               num_cams,
				sel_pairs[0], // int               sel_pairs0           // unused bits should be 0
				sel_pairs[1], // int               sel_pairs1,           // unused bits should be 0
				sel_pairs[2], // int               sel_pairs2,           // unused bits should be 0
				sel_pairs[3], // int               sel_pairs3,           // unused bits should be 0
        		gpu_clt,                       // float          ** gpu_clt,            // [NUM_CAMS] ->[TILESY][TILESX][NUM_COLORS][DTT_SIZE*DTT_SIZE]
				num_colors,                    // int               colors,             // number of colors (3/1)
				color_weights[0], // 0.25,     // float             scale0,             // scale for R
				color_weights[1], // 0.25,     // float             scale1,             // scale for B
				color_weights[2], // 0.5,      // float             scale2,             // scale for G
				30.0*30.0,                     // float             fat_zero2,          // here - absolute (squared)
				gpu_ftasks,                    // float            * gpu_ftasks,        // flattened tasks, 27 floats for quad EO, 99 floats for LWIR16
				tp_task_size,                  // int               num_tiles) // number of tiles in task
				TILESX,                        // int               tilesx,             // number of tile rows
				gpu_corr_indices,              // int             * gpu_corr_indices,   // packed tile+pair
				gpu_num_corr_tiles,            // int             * pnum_corr_tiles,    // pointer to a number of correlation tiles to process
				dstride_corr_td/sizeof(float), // const size_t      corr_stride,        // in floats
				0,                             // int               corr_radius,        // radius of the output correlation (7 for 15x15)
				gpu_corrs_td);                 // float           * gpu_corrs);         // correlation output data
    	getLastCudaError("Kernel failure:correlate2D");
    	checkCudaErrors(cudaDeviceSynchronize());
    	printf("correlate2D-TD pass: %d\n",i);

    	checkCudaErrors(cudaMemcpy(
    			&num_corrs,
    			gpu_num_corr_tiles,
    			sizeof(int),
    			cudaMemcpyDeviceToHost));

#ifdef QUAD_COMBINE
    	num_corr_combo = num_corrs/num_pairs;
    	corr2D_combine<<<1,1>>>( // Combine quad (2 hor, 2 vert) pairs
    			num_corr_combo, // tp_task_size,     // int               num_tiles,          // number of tiles to process (each with num_pairs)
				num_pairs,                           // int               num_pairs,          // num pairs per tile (should be the same)
    			1,                                   // int               init_output,        // !=0 - reset output tiles to zero before accumulating
    			0x0f,                                // int               pairs_mask,         // selected pairs (0x3 - horizontal, 0xc - vertical, 0xf - quad, 0x30 - cross)
				gpu_corr_indices,                    // int             * gpu_corr_indices,   // packed tile+pair
				gpu_corrs_combo_indices,             // int             * gpu_combo_indices,  // output if noty null: packed tile+pairs_mask (will point to the first used pair
				dstride_corr_td/sizeof(float),       // const size_t      corr_stride,        // (in floats) stride for the input TD correlations
				gpu_corrs_td,                       // float           * gpu_corrs,          // input correlation tiles
				dstride_corr_combo_td/sizeof(float), // const size_t      corr_stride_combo,  // (in floats) stride for the output TD correlations (same as input)
				gpu_corrs_combo_td);                 // float           * gpu_corrs_combo);   // combined correlation output (one per tile)

    	getLastCudaError("Kernel failure:corr2D_combine");
    	checkCudaErrors(cudaDeviceSynchronize());
    	printf("corr2D_combine pass: %d\n",i);

    	corr2D_normalize<<<1,1>>>(
    			num_corr_combo, //tp_task_size,      // int               num_corr_tiles,     // number of correlation tiles to process
				dstride_corr_combo_td/sizeof(float), // const size_t      corr_stride_td,     // in floats
				gpu_corrs_combo_td,                  // float           * gpu_corrs_td,       // correlation tiles in transform domain
				(float *) 0, // corr_weights,                  // float           * corr_weights,       // null or per-tile weight (fat_zero2 will be divided by it)
				dstride_corr_combo/sizeof(float),    // const size_t      corr_stride,        // in floats
				gpu_corrs_combo,                     // float           * gpu_corrs,          // correlation output data (pixel domain)
				30.0 * 30.0,                         // float             fat_zero2,           // here - absolute
				CORR_OUT_RAD);                       // int               corr_radius);        // radius of the output correlation (7 for 15x15)
#else
    	checkCudaErrors(cudaDeviceSynchronize());

    	corr2D_normalize<<<1,1>>>(
    			num_corrs, //tp_task_size,           // int               num_corr_tiles,     // number of correlation tiles to process
				dstride_corr_td/sizeof(float),       // const size_t      corr_stride_td,     // in floats
				gpu_corrs_td,                        // float           * gpu_corrs_td,       // correlation tiles in transform domain
				(float *) 0, // corr_weights,        // float           * corr_weights,       // null or per-tile weight (fat_zero2 will be divided by it)
				dstride_corr/sizeof(float),          // const size_t      corr_stride,        // in floats
				gpu_corrs,                           // float           * gpu_corrs,          // correlation output data (pixel domain)
				30.0 * 30.0,                         // float             fat_zero2,           // here - absolute
				CORR_OUT_RAD);                       // int               corr_radius);        // radius of the output correlation (7 for 15x15)
#endif
    	getLastCudaError("Kernel failure:corr2D_normalize");
    	checkCudaErrors(cudaDeviceSynchronize());
    	printf("corr2D_normalize pass: %d\n",i);
    }


    sdkStopTimer(&timerCORRTD);
    float avgTimeCORRTD = (float)sdkGetTimerValue(&timerCORRTD) / (float)numIterations;
    sdkDeleteTimer(&timerCORRTD);
    printf("Average CORR-TD and companions run time =%f ms, num cor tiles (old) = %d\n",  avgTimeCORRTD, num_corrs);

#ifdef QUAD_COMBINE
    int corr_size_combo =        2 * CORR_OUT_RAD + 1;
    int rslt_corr_size_combo =   num_corr_combo * corr_size_combo * corr_size_combo;
    float * cpu_corr_combo =    (float *)malloc(rslt_corr_size_combo * sizeof(float));

    checkCudaErrors(cudaMemcpy2D(
    		cpu_corr_combo,
			(corr_size_combo * corr_size_combo) * sizeof(float),
			gpu_corrs_combo,
			dstride_corr_combo,
			(corr_size_combo * corr_size_combo) * sizeof(float),
			num_corr_combo,
			cudaMemcpyDeviceToHost));
    //    const char* result_corr_quad_file =  "clt/main_corr-quad.corr";
    //    const char* result_corr_cross_file = "clt/main_corr-cross.corr";

#ifndef NSAVE_CORR
    printf("Writing phase correlation data to %s\n",  result_corr_quad_file);
    writeFloatsToFile(
    		cpu_corr_combo,         // float *       data, // allocated array
			rslt_corr_size_combo,   // int           size, // length in elements
			result_corr_quad_file); // 			     const char *  path) // file path
#endif
    free(cpu_corr_combo);

#else // QUAD_COMBINE
    // Reading / formatting / saving 	correlate2D(TD) + corr2D_normalize

    checkCudaErrors(cudaMemcpy(
    		&num_corrs,
			gpu_num_corr_tiles,
			sizeof(int),
			cudaMemcpyDeviceToHost));
  //  printf("Average CORR run time =%f ms, num cor tiles (new) = %d\n",  avgTimeCORR, num_corrs);

//    int corr_size =        2 * CORR_OUT_RAD + 1;
//    int rslt_corr_size =   num_corrs * corr_size * corr_size;
//    float * cpu_corr = (float *)malloc(rslt_corr_size * sizeof(float));

	rslt_corr_size =   num_corrs * corr_size * corr_size;
	corr_img_size = num_corr_indices * 16*16; // NAN
	corr_img = (float *)malloc(corr_img_size * sizeof(float));
	cpu_corr = (float *)malloc(rslt_corr_size * sizeof(float));
	cpu_corr_indices = (int *) malloc(num_corr_indices * sizeof(int));



    checkCudaErrors(cudaMemcpy2D(
    		cpu_corr,
			(corr_size * corr_size) * sizeof(float),
			gpu_corrs,
			dstride_corr,
			(corr_size * corr_size) * sizeof(float),
			num_corrs,
			cudaMemcpyDeviceToHost));
    //    checkCudaErrors (cudaMalloc((void **)&gpu_corr_indices,        num_pairs * TILESX * TILESY*sizeof(int)));
//    int num_tiles = TILESX * TILESYA;
//    int num_corr_indices = num_pairs * num_tiles;
//    int * cpu_corr_indices = (int *) malloc(num_corr_indices * sizeof(int));
    checkCudaErrors(cudaMemcpy(
    		cpu_corr_indices,
			gpu_corr_indices,
			num_corr_indices * sizeof(int),
			cudaMemcpyDeviceToHost));
//    int corr_img_size = num_corr_indices * 16*16; // NAN
//    float * corr_img = (float *)malloc(corr_img_size * sizeof(float));
    for (int i = 0; i < corr_img_size; i++){
    	corr_img[i] = NAN;
    }
    for (int ict = 0; ict < num_corr_indices; ict++){
    	//    	int ct = cpu_corr_indices[ict];
    	int ctt = ( cpu_corr_indices[ict] >>  CORR_NTILE_SHIFT);
    	int cpair = cpu_corr_indices[ict] & ((1 << CORR_NTILE_SHIFT) - 1);
    	int ty = ctt / TILESX;
    	int tx = ctt % TILESX;
    	//		int src_offs0 = ict * num_pairs * corr_size * corr_size;
    	int src_offs0 = ict * corr_size * corr_size;
    	int dst_offs0 = cpair * (num_tiles * 16 * 16) +  (ty * 16 * TILESX * 16) + (tx * 16);

    	for (int iy = 0; iy < corr_size; iy++){
    		int src_offs = src_offs0 + iy * corr_size; // ict * num_pairs * corr_size * corr_size;
    		int dst_offs = dst_offs0 + iy * (TILESX * 16);
    		for (int ix = 0; ix < corr_size; ix++){
    			corr_img[dst_offs++] = cpu_corr[src_offs++];
    		}
    	}
    }
    // num_pairs

#ifndef NSAVE_CORR
    printf("Writing phase correlation data to %s, width = %d, height=%d, slices=%d, length=%ld bytes\n",
    		result_corr_td_norm_file, (TILESX*16),(TILESYA*16), num_pairs, (corr_img_size * sizeof(float)) ) ;
    writeFloatsToFile(
    		corr_img,                  // float *       data, // allocated array
			corr_img_size,             // int           size, // length in elements
			result_corr_td_norm_file); // 			   const char *  path) // file path
#endif
    free (cpu_corr);
    free (cpu_corr_indices);
    free (corr_img);

#endif // // QUAD_COMBINE#else
#endif // ifndef NOCORR_TD


// Testing "interframe" correlation with itself, assuming direct convert already ran




#ifdef CORR_INTER_SELF
    int sel_sensors = 0xffff; // 0x7fff; // 0xffff;
    int num_sel_senosrs = 16; // 15; // 16;
    num_pairs = num_sel_senosrs+1;
    num_corr_indices = num_pairs * num_tiles;
    StopWatchInterface *timerINTERSELF = 0;
    sdkCreateTimer(&timerINTERSELF);
//    int num_corr_combo_inter;
    for (int i = i0; i < numIterations; i++)
    {
    	if (i == 0)
    	{
    		checkCudaErrors(cudaDeviceSynchronize());
    		sdkResetTimer(&timerINTERSELF);
    		sdkStartTimer(&timerINTERSELF);
    	}
    	correlate2D_inter<<<1,1>>>( // only results in TD
    			num_cams,                      // int               num_cams,
				sel_sensors,                   // int               sel_sensors,
				gpu_clt,                       // float          ** gpu_clt,            // [num_cams] ->[TILES-Y][TILES-X][colors][DTT_SIZE*DTT_SIZE]
				gpu_clt,                       // float          ** gpu_clt_ref,        // [num_cams] ->[TILES-Y][TILES-X][colors][DTT_SIZE*DTT_SIZE]
				num_colors,                    // int               colors,             // number of colors (3/1)
				color_weights[0], // 0.25,     // float             scale0,             // scale for R
				color_weights[1], // 0.25,     // float             scale1,             // scale for B
				color_weights[2], // 0.5,      // float             scale2,             // scale for G
				gpu_ftasks,                    // float            * gpu_ftasks,        // flattened tasks, 27 floats for quad EO, 99 floats for LWIR16
				tp_task_size,                  // int               num_tiles) // number of tiles in task
				TILESX,                        // int               tilesx,             // number of tile rows
				gpu_corr_indices,              // int             * gpu_corr_indices,   // packed tile+pair
				gpu_num_corr_tiles,            // int             * pnum_corr_tiles,    // pointer to a number of correlation tiles to process
				dstride_corr_td/sizeof(float), // const size_t      corr_stride,        // in floats
				gpu_corrs_td);                 // float           * gpu_corrs);         // correlation output data
        getLastCudaError("Kernel failure:correlate2D_inter");
    	checkCudaErrors(cudaDeviceSynchronize());
    	printf("correlate2D_inter-TD pass: %d\n",i);
    	checkCudaErrors(cudaMemcpy(
    			&num_corrs,
    			gpu_num_corr_tiles,
    			sizeof(int),
    			cudaMemcpyDeviceToHost));
    	checkCudaErrors(cudaDeviceSynchronize());

    	corr2D_normalize<<<1,1>>>(
    			num_corrs, //tp_task_size,           // int               num_corr_tiles,     // number of correlation tiles to process
				dstride_corr_td/sizeof(float),       // const size_t      corr_stride_td,     // in floats
				gpu_corrs_td,                        // float           * gpu_corrs_td,       // correlation tiles in transform domain
				(float *) 0, // corr_weights,        // float           * corr_weights,       // null or per-tile weight (fat_zero2 will be divided by it)
				dstride_corr/sizeof(float),          // const size_t      corr_stride,        // in floats
				gpu_corrs,                           // float           * gpu_corrs,          // correlation output data (pixel domain)
				30.0 * 30.0,                         // float             fat_zero2,           // here - absolute
				CORR_OUT_RAD);                       // int               corr_radius);        // radius of the output correlation (7 for 15x15)
    	getLastCudaError("Kernel failure:corr2D_normalize");
    	checkCudaErrors(cudaDeviceSynchronize());
    	printf("corr2D_normalize pass: %d\n",i);
    }
    sdkStopTimer(&timerINTERSELF);
    float avgTimeINTERSELF = (float)sdkGetTimerValue(&timerINTERSELF) / (float)numIterations;
    sdkDeleteTimer(&timerINTERSELF);
    printf("Average CORR-TD and companions run time =%f ms, num cor tiles (old) = %d\n",  avgTimeINTERSELF, num_corrs);


	rslt_corr_size =   num_corrs * corr_size * corr_size;
	corr_img_size = num_corr_indices * 16*16; // NAN
	corr_img = (float *)malloc(corr_img_size * sizeof(float));
	cpu_corr = (float *)malloc(rslt_corr_size * sizeof(float));
	cpu_corr_indices = (int *) malloc(num_corr_indices * sizeof(int));

    checkCudaErrors(cudaMemcpy2D(
    		cpu_corr,
			(corr_size * corr_size) * sizeof(float),
			gpu_corrs,
			dstride_corr,
			(corr_size * corr_size) * sizeof(float),
			num_corrs,
			cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaMemcpy(
    		cpu_corr_indices,
			gpu_corr_indices,
			num_corr_indices * sizeof(int),
			cudaMemcpyDeviceToHost));

    for (int i = 0; i < corr_img_size; i++){
    	corr_img[i] = NAN;
    }
//  int num_pairs = 120;
//  int sel_sensors = 0xffff;
//  int num_sel_senosrs = 16;
//	int corr_size =        2 * CORR_OUT_RAD + 1; // 15
//	int num_tiles = tp_task_size; // TILESX * TILESYA; //Was this on 01/22/2022
//	int num_corr_indices = num_pairs * num_tiles;

    for (int ict = 0; ict < num_corr_indices; ict++){
    	int ctt = ( cpu_corr_indices[ict] >>  CORR_NTILE_SHIFT);
    	int cpair = cpu_corr_indices[ict] & ((1 << CORR_NTILE_SHIFT) - 1);
    	if (cpair == 0xff){
    		cpair = num_sel_senosrs;
    	}
    	int ty = ctt / TILESX;
    	int tx = ctt % TILESX;
    	int src_offs0 = ict * corr_size * corr_size;
    	int dst_offs0 = cpair * (num_tiles * 16 * 16) +  (ty * 16 * TILESX * 16) + (tx * 16);
    	for (int iy = 0; iy < corr_size; iy++){
    		int src_offs = src_offs0 + iy * corr_size; // ict * num_pairs * corr_size * corr_size;
    		int dst_offs = dst_offs0 + iy * (TILESX * 16);
    		for (int ix = 0; ix < corr_size; ix++){
    			corr_img[dst_offs++] = cpu_corr[src_offs++];
    		}
    	}
    }

#ifndef NSAVE_CORR
    printf("Writing interscene phase correlation data to %s, width = %d, height=%d, slices=%d, length=%ld bytes\n",
    		result_inter_td_norm_file, (TILESX*16),(TILESYA*16), num_pairs, (corr_img_size * sizeof(float)) ) ;
    writeFloatsToFile(
    		corr_img,                  // float *       data, // allocated array
			corr_img_size,             // int           size, // length in elements
			result_inter_td_norm_file); // 			   const char *  path) // file path
#endif
    free (cpu_corr);
    free (cpu_corr_indices);
    free (corr_img);
#endif    // #ifdef CORR_INTER_SELF









// -----------------
#ifndef NOTEXTURES

    		 dim3 threads0(CONVERT_DIRECT_INDEXING_THREADS, 1, 1);
    		 dim3 blocks0 ((tp_task_size + CONVERT_DIRECT_INDEXING_THREADS -1) >> CONVERT_DIRECT_INDEXING_THREADS_LOG2,1, 1);
    		int  linescan_order = 1; // output low-res in linescan order, 0 - in gpu_texture_indices order
     		printf("threads0=(%d, %d, %d)\n",threads0.x,threads0.y,threads0.z);
     		printf("blocks0=(%d, %d, %d)\n",blocks0.x,blocks0.y,blocks0.z);
     		int   cpu_pnum_texture_tiles = 0;
     	    int * gpu_pnum_texture_tiles;
     	    checkCudaErrors (cudaMalloc((void **)&gpu_pnum_texture_tiles, sizeof(int)));


    		    StopWatchInterface *timerTEXTURE = 0;
    		    sdkCreateTimer(&timerTEXTURE);

    		    for (int i = i0; i < numIterations; i++)
    		    {
    		    	if (i == 0)
    		    	{
    		    		checkCudaErrors(cudaDeviceSynchronize());
    		    		sdkResetTimer(&timerTEXTURE);
    		    		sdkStartTimer(&timerTEXTURE);
    		    	}
    		    	int shared_size = host_get_textures_shared_size( // in bytes
    		    			num_cams,         // int                num_cams,     // actual number of cameras
							texture_colors,   // int                num_colors,   // actual number of colors: 3 for RGB, 1 for LWIR/mono
							0);           // int *              offsets);     // in floats
    				printf("\n1. shared_size=%d, num_cams=%d, colors=%d\n",shared_size,num_cams, texture_colors);
    				//*pnum_texture_tiles = 0;
    				cpu_pnum_texture_tiles = 0;
    				checkCudaErrors(cudaMemcpy(
    						gpu_pnum_texture_tiles,
							&cpu_pnum_texture_tiles,
							sizeof(int),
							cudaMemcpyHostToDevice));
    				cudaFuncSetAttribute(textures_accumulate, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size); // 65536); // for CC 7.5
    				cudaFuncSetAttribute(textures_accumulate, cudaFuncAttributePreferredSharedMemoryCarveout,cudaSharedmemCarveoutMaxShared);

#ifdef NO_DP
    				 create_nonoverlap_list<<<blocks0,threads0>>>(
    						 num_cams,                // int                num_cams,
    						 gpu_ftasks,              // float            * gpu_ftasks,         // flattened tasks, 27 floats for quad EO, 99 floats for LWIR16
							 tp_task_size,            // int                num_tiles,           // number of tiles in task
							 TILESX,                  // int                width,               // number of tiles in a row
    						 gpu_texture_indices,     // int *              nonoverlap_list,     // pointer to the calculated number of non-zero tiles
							 gpu_pnum_texture_tiles); // int *              pnonoverlap_length)  //  indices to gpu_tasks  // should be initialized to zero
    				 cudaDeviceSynchronize();

    				 checkCudaErrors(cudaMemcpy(
    						 &cpu_pnum_texture_tiles,
							 gpu_pnum_texture_tiles,
							 sizeof(int),
							 cudaMemcpyDeviceToHost));
    				 printf("cpu_pnum_texture_tiles = %d\n",  cpu_pnum_texture_tiles);



    				 int num_cams_per_thread = NUM_THREADS / TEXTURE_THREADS_PER_TILE; // 4 cameras parallel, then repeat
    				 dim3 threads_texture1(TEXTURE_THREADS_PER_TILE, num_cams_per_thread, 1); // TEXTURE_TILES_PER_BLOCK, 1);
    				 dim3 grid_texture1((cpu_pnum_texture_tiles + TEXTURE_TILES_PER_BLOCK-1) / TEXTURE_TILES_PER_BLOCK,1,1);
    		     		printf("threads_texture1=(%d, %d, %d)\n",threads_texture1.x,threads_texture1.y,threads_texture1.z);
    		     		printf("grid_texture1=(%d, %d, %d)\n",grid_texture1.x,grid_texture1.y,grid_texture1.z);
    				textures_accumulate <<<grid_texture1,threads_texture1,  shared_size>>>( // 65536>>>( //
    						num_cams,                        // 	int               num_cams,           // number of cameras used
							(int *) 0,                       // int             * woi,                // x, y, width,height
							gpu_clt,                         // float          ** gpu_clt,            // [num_cams] ->[TILES-Y][TILES-X][colors][DTT_SIZE*DTT_SIZE]
							cpu_pnum_texture_tiles,          // *pnum_texture_tiles,             // size_t            num_texture_tiles,  // number of texture tiles to process
							0,                               //                gpu_texture_indices_offset,// add to gpu_texture_indices
							gpu_texture_indices,             // int             * gpu_texture_indices,// packed tile + bits (now only (1 << 7)
							gpu_geometry_correction,         // struct gc       * gpu_geometry_correction,
							texture_colors,                  // int               colors,             // number of colors (3/1)
							(texture_colors == 1),           // int               is_lwir,            // do not perform shot correction
							generate_RBGA_params[0], // min_shot,      // float             min_shot,           // 10.0
							generate_RBGA_params[1], // scale_shot,    // float             scale_shot,         // 3.0
							generate_RBGA_params[2], // diff_sigma,    // float             diff_sigma,         // pixel value/pixel change
							generate_RBGA_params[3], // diff_threshold,// float             diff_threshold,     // pixel value/pixel change
							generate_RBGA_params[4], // min_agree,     // float             min_agree,          // minimal number of channels to agree on a point (real number to work with fuzzy averages)
							gpu_color_weights,               // float             weights[3],         // scale for R,B,G
							1,                       // dust_remove,                     // int               dust_remove,        // Do not reduce average weight when only one image differs much from the average
							keep_texture_weights, // 0, // 1                               // int               keep_weights,       // return channel weights after A in RGBA (was removed) (should be 0 if gpu_texture_rbg)?
							// combining both non-overlap and overlap (each calculated if pointer is not null )
							0,                               // size_t      texture_rbg_stride, // in floats
							(float *) 0,                     // float           * gpu_texture_rbg,     // (number of colors +1 + ?)*16*16 rgba texture tiles
							dstride_textures /sizeof(float),          // texture_stride,                  // size_t      texture_stride,     // in floats (now 256*4 = 1024)
							gpu_textures, // (float *) 0,             // gpu_texture_tiles,               //(float *)0);// float           * gpu_texture_tiles);  // (number of colors +1 + ?)*16*16 rgba texture tiles
							linescan_order,          // int               linescan_order,     // if !=0 then output gpu_diff_rgb_combo in linescan order, else  - in gpu_texture_indices order
							gpu_diff_rgb_combo, //);             // float           * gpu_diff_rgb_combo) // diff[num_cams], R[num_cams], B[num_cams],G[num_cams]
							TILESX);
#else // #ifdef NO_DP
    				//keep_texture_weights is assumed 0 in textures_nonoverlap
    		    	textures_nonoverlap<<<1,1>>> ( //,65536>>> (
    		    			num_cams,              // int                num_cams,           // number of cameras used
    						gpu_ftasks,            // float            * gpu_ftasks,         // flattened tasks, 27 floats for quad EO, 99 floats
    		                tp_task_size,          // int                num_tiles,          // number of tiles in task list
    		    	// declare arrays in device code?
    						gpu_texture_indices,   // int             * gpu_texture_indices,// packed tile + bits (now only (1 << 7)
    						gpu_pnum_texture_tiles, // int             * pnum_texture_tiles,  // returns total number of elements in gpu_texture_indices array
    				        gpu_clt ,              // float          ** gpu_clt,            // [NUM_CAMS] ->[TILESY][TILESX][NUM_COLORS][DTT_SIZE*DTT_SIZE]
    						gpu_geometry_correction, // struct gc     * gpu_geometry_correction,
    						texture_colors,        // int               colors,             // number of colors (3/1)
    						(texture_colors == 1), // int               is_lwir,            // do not perform shot correction
    						gpu_generate_RBGA_params,
    						gpu_color_weights,     // float             weights[3],         // scale for R
    						1,                     // int               dust_remove,        // Do not reduce average weight when only one image differes much from the average
    		    	// combining both non-overlap and overlap (each calculated if pointer is not null )
    						dstride_textures/sizeof(float), // size_t            texture_stride,     // in floats (now 256*4 = 1024)  // may be 0 if not needed
    						gpu_textures,         // float           * gpu_texture_tiles,
    						linescan_order,       // int               linescan_order,
    						gpu_diff_rgb_combo,   //);  // float           * gpu_diff_rgb_combo); // diff[NUM_CAMS], R[NUM_CAMS], B[NUM_CAMS],G[NUM_CAMS] // may be 0 if not needed
    						TILESX);
#endif
    				getLastCudaError("Kernel failure");
    				checkCudaErrors(cudaDeviceSynchronize());

    				printf("test pass: %d\n",i);
    		    }
    		    sdkStopTimer(&timerTEXTURE);
    		    float avgTimeTEXTURES = (float)sdkGetTimerValue(&timerTEXTURE) / (float)numIterations;
    		    sdkDeleteTimer(&timerTEXTURE);
    		    printf("Average Texture run time =%f ms\n",  avgTimeTEXTURES);
#ifdef NO_DP

#else
				 checkCudaErrors(cudaMemcpy(
						 &cpu_pnum_texture_tiles,
						 gpu_pnum_texture_tiles,
						 sizeof(int),
						 cudaMemcpyDeviceToHost));
				 printf("cpu_pnum_texture_tiles = %d\n",  cpu_pnum_texture_tiles);
#endif


    		    int rslt_texture_size =   num_textures * tile_texture_size;
    			checkCudaErrors(cudaMemcpy(
    					(float * ) texture_indices,
						gpu_texture_indices,
						cpu_pnum_texture_tiles  * sizeof(float),
    					cudaMemcpyDeviceToHost));

    		    float * cpu_textures = (float *)malloc(rslt_texture_size * sizeof(float));
    		    checkCudaErrors(cudaMemcpy2D(
    		    		cpu_textures,
    					tile_texture_size * sizeof(float),
    					gpu_textures,
    					dstride_textures,
    					tile_texture_size * sizeof(float),
    					num_textures,
    		    		cudaMemcpyDeviceToHost));
//    		    float non_overlap_layers [tile_texture_layers][TILESY*16][TILESX*16];
    		    int num_nonoverlap_pixels = tile_texture_layers * TILESY*16 * TILESX*16;
    		    float * non_overlap_layers = (float *)malloc(num_nonoverlap_pixels* sizeof(float));
    		    for (int i = 0; i < num_nonoverlap_pixels; i++){
    		    	non_overlap_layers[i] = NAN;
    		    }
    		    for (int itile = 0; itile < cpu_pnum_texture_tiles; itile++) { // if (texture_indices[itile] & ((1 << LIST_TEXTURE_BIT))){
    		    	int ntile = texture_indices[itile] >> CORR_NTILE_SHIFT;
    		    	int tileX = ntile % TILESX;
    		    	int tileY = ntile / TILESX;
    		    	for (int ilayer = 0; ilayer < tile_texture_layers; ilayer++){
    		    		int src_index0 = itile * tile_texture_size + 256 * ilayer;
    		    		int dst_index0 =  ilayer * (TILESX * TILESYA * 256) + (tileY * 16) * (16 * TILESX) + (tileX * 16);
    		    		for (int iy = 0; iy < 16; iy++){
    		    			int src_index1 = src_index0 + 16 * iy;
    		    			int dst_index1 = dst_index0 + iy * (16 * TILESX);
        		    		for (int ix = 0; ix < 16; ix++){
        		    			int src_index= itile * tile_texture_size + 256 * ilayer + 16 * iy + ix;
        		    			int dst_index = ilayer * (TILESX * TILESY * 256) + (tileY * 16 + iy) * (16 * TILESX) + (tileX * 16) + ix;
        		    			non_overlap_layers[dst_index] = cpu_textures[src_index];
        		    		}
    		    		}
    		    	}
    		    }


    		    int ntiles = TILESX * TILESY;
    		    int nlayers = num_cams * (num_colors + 1);
    		    int diff_rgb_combo_size = ntiles * nlayers;
    		    float * cpu_diff_rgb_combo = (float *)malloc(diff_rgb_combo_size  * sizeof(float));
    			checkCudaErrors(cudaMemcpy(
    					cpu_diff_rgb_combo,
    					gpu_diff_rgb_combo,
    					diff_rgb_combo_size  * sizeof(float),
    					cudaMemcpyDeviceToHost));
    		    float * cpu_diff_rgb_combo_out = (float *)malloc(diff_rgb_combo_size  * sizeof(float));
    		    for (int nl = 0; nl <nlayers; nl++){
    		    	for (int ntile = 0; ntile < ntiles; ntile++){
    		    		cpu_diff_rgb_combo_out[nl * ntiles + ntile] = cpu_diff_rgb_combo[ntile * nlayers + nl];
    		    	}
    		    }



#ifndef NSAVE_TEXTURES
#ifdef NO_DP
    		    		printf("Writing phase texture data to %s\n",  result_textures_file);
    		    		writeFloatsToFile(
    		    				non_overlap_layers,    // float *       data, // allocated array
    							rslt_texture_size,     // int           size, // length in elements
    							result_textures_file); // 			   const char *  path) // file path
    		    		printf("Writing low-res data to %s\n",  result_diff_rgb_combo_file);
    		    		writeFloatsToFile(
    		    				cpu_diff_rgb_combo_out, // cpu_diff_rgb_combo,    // float *       data, // allocated array
    							diff_rgb_combo_size,    // int           size, // length in elements
    							result_diff_rgb_combo_file); // 			   const char *  path) // file path
#else
    		    		printf("Writing phase texture data to %s\n",  result_textures_file_dp);
    		    		writeFloatsToFile(
    		    				non_overlap_layers,    // float *       data, // allocated array
    							rslt_texture_size,     // int           size, // length in elements
    							result_textures_file_dp); // 			   const char *  path) // file path
    		    		printf("Writing low-res data to %s\n",  result_diff_rgb_combo_file_dp);
    		    		writeFloatsToFile(
    		    				cpu_diff_rgb_combo_out, // cpu_diff_rgb_combo,    // float *       data, // allocated array
    							diff_rgb_combo_size,    // int           size, // length in elements
    							result_diff_rgb_combo_file_dp); // 			   const char *  path) // file path
#endif

#ifdef 	DBG_TILE
    		#ifdef DEBUG10
    		    		int texture_offset = DBG_TILE * tile_texture_size;
    		    		int chn = 0;
    		    		for (int i = 0; i < tile_texture_size; i++){
    		    			if ((i % 256) == 0){
    		    				printf("\nchn = %d\n", chn++);
    		    			}
    		    			printf("%10.4f", *(cpu_textures + texture_offset + i));
    		    			if (((i + 1) % 16) == 0){
    		    				printf("\n");
    		    			} else {
    		    				printf(" ");
    		    			}
    		    		}
    		#endif // DEBUG9
#endif //#ifdef 	DBG_TILE
    		#endif
    		    		free(cpu_textures);
    		    		free (cpu_diff_rgb_combo);
    		    		free (cpu_diff_rgb_combo_out);
    		    		checkCudaErrors(cudaFree(gpu_pnum_texture_tiles));

#endif //NOTEXTURES


#ifndef NOTEXTURE_RGBAXXX
    dim3 threads_rgba(1, 1, 1);
    dim3 grid_rgba(1,1,1);
    printf("threads_rgba=(%d, %d, %d)\n", threads_rgba.x,threads_rgba.y,threads_rgba.z);
    printf("grid_rgba=(%d, %d, %d)\n",    grid_rgba.x,grid_rgba.y,grid_rgba.z);
    StopWatchInterface *timerRGBA = 0;
    sdkCreateTimer(&timerRGBA);

    for (int i = i0; i < numIterations; i++)
    {
    	if (i == 0)
    	{
    		checkCudaErrors(cudaDeviceSynchronize());
    		sdkResetTimer(&timerRGBA);
    		sdkStartTimer(&timerRGBA);
    	}
    	// FIXME: update to use new correlations and num_cams
#ifdef NO_DP
    	generate_RBGA_host (
    			num_cams,              // int                num_cams,           // number of cameras used
    	// Parameters to generate texture tasks
				gpu_ftasks,         // float            * gpu_ftasks,         // flattened tasks, 27 floats for quad EO, 99 floats for LWIR16
                tp_task_size,          // int                num_tiles,          // number of tiles in task list
		// Does not require initialized gpu_texture_indices to be initialized - just allocated, will generate.
	            gpu_texture_indices,   // int              * gpu_texture_indices,// packed tile + bits (now only (1 << 7)
	            gpu_num_texture_tiles, // int              * num_texture_tiles,  // number of texture tiles to process (8 elements)
	            gpu_woi,               // int              * woi,                // x,y,width,height of the woi
	            TILESX,                // int                width,  // <= TILESX, use for faster processing of LWIR images (should be actual + 1)
	            TILESY,                // int                height); // <= TILESY, use for faster processing of LWIR images
    	// Parameters for the texture generation
	            gpu_clt ,              // float          ** gpu_clt,            // [NUM_CAMS] ->[TILESY][TILESX][NUM_COLORS][DTT_SIZE*DTT_SIZE]
				gpu_geometry_correction, // struct gc     * gpu_geometry_correction,
	            texture_colors,        // int               colors,             // number of colors (3/1)
	            (texture_colors == 1), // int               is_lwir,            // do not perform shot correction
				generate_RBGA_params, //		float             cpu_params[5],      // mitigating CUDA_ERROR_INVALID_PTX
				gpu_color_weights,     // float             weights[3],         // scale for R
	            1,                     // int               dust_remove,        // Do not reduce average weight when only one image differes much from the average
	            0,                     // int               keep_weights,       // return channel weights after A in RGBA
				dstride_textures_rbga/sizeof(float), // 	const size_t      texture_rbga_stride,     // in floats
				gpu_textures_rbga);     // 	float           * gpu_texture_tiles)    // (number of colors +1 + ?)*16*16 rgba texture tiles
#else
    	int shared_size = host_get_textures_shared_size( // in bytes
    	        num_cams,     // int                num_cams,     // actual number of cameras
				texture_colors, // colors,   // int                num_colors,   // actual number of colors: 3 for RGB, 1 for LWIR/mono
    	        0);           // int *              offsets);     // in floats
    	printf("\n1. shared_size=%d, num_cams=%d, colors=%d\n",shared_size,num_cams, texture_colors);

		cudaFuncSetAttribute(textures_accumulate, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size); // 60000); // 5536); // for CC 7.5
    	generate_RBGA<<<1,1>>> (
    			num_cams,              // int                num_cams,           // number of cameras used
    	// Parameters to generate texture tasks
				gpu_ftasks,         // float            * gpu_ftasks,         // flattened tasks, 27 floats for quad EO, 99 floats for LWIR16
//                gpu_tasks,             // struct tp_task   * gpu_tasks,
                tp_task_size,          // int                num_tiles,          // number of tiles in task list
		// Does not require initialized gpu_texture_indices to be initialized - just allocated, will generate.
	            gpu_texture_indices,   // int              * gpu_texture_indices,// packed tile + bits (now only (1 << 7)
	            gpu_num_texture_tiles, // int              * num_texture_tiles,  // number of texture tiles to process (8 elements)
	            gpu_woi,               // int              * woi,                // x,y,width,height of the woi
	            TILESX,                // int                width,  // <= TILESX, use for faster processing of LWIR images (should be actual + 1)
	            TILESY,                // int                height); // <= TILESY, use for faster processing of LWIR images
    	// Parameters for the texture generation
	            gpu_clt ,              // float          ** gpu_clt,            // [NUM_CAMS] ->[TILESY][TILESX][NUM_COLORS][DTT_SIZE*DTT_SIZE]
				gpu_geometry_correction, // struct gc     * gpu_geometry_correction,
	            texture_colors,        // int               colors,             // number of colors (3/1)
	            (texture_colors == 1), // int               is_lwir,            // do not perform shot correction
				gpu_generate_RBGA_params,
				gpu_color_weights,     // float             weights[3],         // scale for R
	            1,                     // int               dust_remove,        // Do not reduce average weight when only one image differes much from the average
	            0,                     // int               keep_weights,       // return channel weights after A in RGBA
				dstride_textures_rbga/sizeof(float), // 	const size_t      texture_rbga_stride,     // in floats
				gpu_textures_rbga);     // 	float           * gpu_texture_tiles)    // (number of colors +1 + ?)*16*16 rgba texture tiles

    	getLastCudaError("Kernel failure");
    	checkCudaErrors(cudaDeviceSynchronize());
    	printf("test pass: %d\n",i);
#endif
    }
    sdkStopTimer(&timerRGBA);
    float avgTimeRGBA = (float)sdkGetTimerValue(&timerRGBA) / (float)numIterations;
    sdkDeleteTimer(&timerRGBA);
    printf("Average Texture run time =%f ms\n",  avgTimeRGBA);

	checkCudaErrors(cudaMemcpy(
			cpu_woi,
			gpu_woi,
			4 * sizeof(float),
			cudaMemcpyDeviceToHost));
	printf("WOI x=%d, y=%d, width=%d, height=%d\n", cpu_woi[0], cpu_woi[1], cpu_woi[2], cpu_woi[3]);


	// temporarily use larger array (4 pixels each size, switch to cudaMemcpy2DFromArray()
    int rgba_woi_width =  (cpu_woi[2] + 1) * DTT_SIZE;
    int rgba_woi_height = (cpu_woi[3] + 1)* DTT_SIZE;

    int rslt_rgba_size =     rgba_woi_width * rgba_woi_height * rbga_slices;
    float * cpu_textures_rgba = (float *)malloc(rslt_rgba_size * sizeof(float));

    checkCudaErrors(cudaMemcpy2D(
    		cpu_textures_rgba,
			rgba_width * sizeof(float),
			gpu_textures_rbga,
			dstride_textures_rbga,
			rgba_width * sizeof(float),
			rgba_height * rbga_slices,
    		cudaMemcpyDeviceToHost));

#ifndef NSAVE_TEXTURES
#ifdef NO_DP
    printf("Writing RBGA texture slices to %s\n",  result_textures_rgba_file);
    writeFloatsToFile(
    		cpu_textures_rgba,    // float *       data, // allocated array
			rslt_rgba_size,    // int           size, // length in elements
			result_textures_rgba_file); // 			   const char *  path) // file path
#else
    printf("Writing RBGA texture slices to %s\n",  result_textures_rgba_file_dp);
    writeFloatsToFile(
    		cpu_textures_rgba,    // float *       data, // allocated array
			rslt_rgba_size,    // int           size, // length in elements
			result_textures_rgba_file_dp); // 			   const char *  path) // file path
#endif
#endif

#ifdef 	DBG_TILE
#ifdef DEBUG11
    int rgba_offset = (DBG_TILE_Y - cpu_woi[1]) * DTT_SIZE * rgba_woi_width  + (DBG_TILE_X - cpu_woi[0]);
    for (int chn = 0; chn < rbga_slices; chn++){
    	printf("\nchn = %d\n", chn);
    	int rgba_offset_chn = rgba_offset + chn * rgba_woi_width * rgba_woi_height;

    	for (int i = 0; i < 8; i++){
    		for (int j = 0; j < 8; j++){
    			printf("%10.4f ", *(cpu_textures_rgba + rgba_offset_chn + i * rgba_woi_width + j));
    		}
    		printf("\n");
    	}
    }
#endif // DEBUG11
#endif //#ifdef 	DBG_TILE
    free(cpu_textures_rgba);
#endif // ifndef NOTEXTURES




#ifdef SAVE_CLT
    free(cpu_clt);
#endif

    free (host_kern_buf);
    // TODO: move somewhere when all is done
    for (int ncam = 0; ncam < num_cams; ncam++) {
    	checkCudaErrors(cudaFree(gpu_kernels_h[ncam]));
    	checkCudaErrors(cudaFree(gpu_kernel_offsets_h[ncam]));
    	checkCudaErrors(cudaFree(gpu_images_h[ncam]));
    	checkCudaErrors(cudaFree(gpu_clt_h[ncam]));
    	checkCudaErrors(cudaFree(gpu_corr_images_h[ncam]));
    }
//	checkCudaErrors(cudaFree(gpu_tasks));
	checkCudaErrors(cudaFree(gpu_ftasks));
	checkCudaErrors(cudaFree(gpu_active_tiles));
	checkCudaErrors(cudaFree(gpu_num_active));
	checkCudaErrors(cudaFree(gpu_kernels));
	checkCudaErrors(cudaFree(gpu_kernel_offsets));
	checkCudaErrors(cudaFree(gpu_images));
	checkCudaErrors(cudaFree(gpu_clt));
	checkCudaErrors(cudaFree(gpu_corr_images));
	checkCudaErrors(cudaFree(gpu_corrs));
	checkCudaErrors(cudaFree(gpu_corrs_td));
	checkCudaErrors(cudaFree(gpu_corr_indices));
	checkCudaErrors(cudaFree(gpu_corrs_combo));
	checkCudaErrors(cudaFree(gpu_corrs_combo_td));
	checkCudaErrors(cudaFree(gpu_corrs_combo_indices));

	checkCudaErrors(cudaFree(gpu_num_corr_tiles));
	checkCudaErrors(cudaFree(gpu_texture_indices));
	checkCudaErrors(cudaFree(gpu_port_offsets));
	checkCudaErrors(cudaFree(gpu_color_weights));
	checkCudaErrors(cudaFree(gpu_generate_RBGA_params));
	checkCudaErrors(cudaFree(gpu_textures));
	checkCudaErrors(cudaFree(gpu_textures_rbga));
	checkCudaErrors(cudaFree(gpu_diff_rgb_combo));
	checkCudaErrors(cudaFree(gpu_woi));
	checkCudaErrors(cudaFree(gpu_num_texture_tiles));
	checkCudaErrors(cudaFree(gpu_geometry_correction));
    checkCudaErrors(cudaFree(gpu_correction_vector));
    checkCudaErrors(cudaFree(gpu_rByRDist));
    checkCudaErrors(cudaFree(gpu_rot_deriv));


	free (rByRDist);
	free (correction_vector);
	free (ftask_data);
	free (ftask_data1);

	exit(0);
}
