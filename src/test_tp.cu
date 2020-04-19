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

//#include "dtt8x8.cuh"
#include "dtt8x8.h"
#include "geometry_correction.h"
#include "TileProcessor.cuh"
///#include "cuda_profiler_api.h"
//#include "cudaProfiler.h"


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

//  std::ifstream input(path, std::ios::binary );
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

    // CLT testing

    const char* kernel_file[] = {
    		"/data_ssd/git/tile_processor_gpu/clt/main_chn0_transposed.kernel",
			"/data_ssd/git/tile_processor_gpu/clt/main_chn1_transposed.kernel",
			"/data_ssd/git/tile_processor_gpu/clt/main_chn2_transposed.kernel",
			"/data_ssd/git/tile_processor_gpu/clt/main_chn3_transposed.kernel"};

    const char* kernel_offs_file[] = {
    		"/data_ssd/git/tile_processor_gpu/clt/main_chn0_transposed.kernel_offsets",
			"/data_ssd/git/tile_processor_gpu/clt/main_chn1_transposed.kernel_offsets",
			"/data_ssd/git/tile_processor_gpu/clt/main_chn2_transposed.kernel_offsets",
			"/data_ssd/git/tile_processor_gpu/clt/main_chn3_transposed.kernel_offsets"};

    const char* image_files[] = {
    		"/data_ssd/git/tile_processor_gpu/clt/main_chn0.bayer",
			"/data_ssd/git/tile_processor_gpu/clt/main_chn1.bayer",
			"/data_ssd/git/tile_processor_gpu/clt/main_chn2.bayer",
			"/data_ssd/git/tile_processor_gpu/clt/main_chn3.bayer"};

    const char* ports_offs_xy_file[] = {
    		"/data_ssd/git/tile_processor_gpu/clt/main_chn0.portsxy",
			"/data_ssd/git/tile_processor_gpu/clt/main_chn1.portsxy",
			"/data_ssd/git/tile_processor_gpu/clt/main_chn2.portsxy",
			"/data_ssd/git/tile_processor_gpu/clt/main_chn3.portsxy"};

#ifndef DBG_TILE
    const char* ports_clt_file[] = { // never referenced
    		"/data_ssd/git/tile_processor_gpu/clt/main_chn0.clt",
			"/data_ssd/git/tile_processor_gpu/clt/main_chn1.clt",
			"/data_ssd/git/tile_processor_gpu/clt/main_chn2.clt",
			"/data_ssd/git/tile_processor_gpu/clt/main_chn3.clt"};
    const char* result_rbg_file[] = {
    		"/data_ssd/git/tile_processor_gpu/clt/main_chn0.rbg",
			"/data_ssd/git/tile_processor_gpu/clt/main_chn1.rbg",
			"/data_ssd/git/tile_processor_gpu/clt/main_chn2.rbg",
			"/data_ssd/git/tile_processor_gpu/clt/main_chn3.rbg"};
#endif
    const char* result_corr_file = "/data_ssd/git/tile_processor_gpu/clt/main_corr.corr";
    const char* result_textures_file =       "/data_ssd/git/tile_processor_gpu/clt/texture.rgba";
    const char* result_textures_rgba_file = "/data_ssd/git/tile_processor_gpu/clt/texture_rgba.rgba";

    const char* rByRDist_file =            "/data_ssd/git/tile_processor_gpu/clt/main.rbyrdist";
    const char* correction_vector_file =   "/data_ssd/git/tile_processor_gpu/clt/main.correction_vector";
    const char* geometry_correction_file = "/data_ssd/git/tile_processor_gpu/clt/main.geometry_correction";

    // not yet used
///    float lpf_sigmas[3] = {0.9f, 0.9f, 0.9f}; // G, B, G

    float port_offsets[NUM_CAMS][2] =  {// used only in textures to scale differences
			{-0.5, -0.5},
			{ 0.5, -0.5},
			{-0.5,  0.5},
			{ 0.5,  0.5}};

    int keep_texture_weights = 1; // try with 0 also
    int texture_colors = 3; // result will be 3+1 RGBA (for mono - 2)

    int KERN_TILES = KERNELS_HOR *  KERNELS_VERT * NUM_COLORS;
    int KERN_SIZE =  KERN_TILES * 4 * 64;

//    int CORR_SIZE = (2 * DTT_SIZE -1) * (2 * DTT_SIZE -1);
    int CORR_SIZE = (2 * CORR_OUT_RAD + 1) * (2 * CORR_OUT_RAD + 1);



    float            * host_kern_buf =  (float *)malloc(KERN_SIZE * sizeof(float));
// static - see https://stackoverflow.com/questions/20253267/segmentation-fault-before-main
    static struct tp_task     task_data  [TILESX*TILESY]; // maximal length - each tile
    static struct tp_task     task_data1 [TILESX*TILESY]; // maximal length - each tile
    trot_deriv  rot_deriv;
    int                corr_indices         [NUM_PAIRS*TILESX*TILESY];
//    int                texture_indices      [TILESX*TILESY];
    int                texture_indices      [TILESX*TILESYA];
    int                cpu_woi              [4];

    // host array of pointers to GPU memory
    float            * gpu_kernels_h        [NUM_CAMS];
    struct CltExtra  * gpu_kernel_offsets_h [NUM_CAMS];
    float            * gpu_images_h         [NUM_CAMS];
    float              tile_coords_h        [NUM_CAMS][TILESX * TILESY][2];
    float            * gpu_clt_h            [NUM_CAMS];
///    float            * gpu_lpf_h            [NUM_COLORS]; // never used
    float            * gpu_corr_images_h    [NUM_CAMS];

    float            * gpu_corrs;
    int              * gpu_corr_indices;

    float            * gpu_textures;
    float            * gpu_diff_rgb_combo;
    float            * gpu_textures_rbga;
    int              * gpu_texture_indices;
    int              * gpu_woi;
    int              * gpu_num_texture_tiles;
    float            * gpu_port_offsets;
    float            * gpu_color_weights;
    int                num_corrs;
    int                num_textures;
    int                num_ports = NUM_CAMS;
    // GPU pointers to GPU pointers to memory
    float           ** gpu_kernels; //           [NUM_CAMS];
    struct CltExtra ** gpu_kernel_offsets; //    [NUM_CAMS];
    float           ** gpu_images; //            [NUM_CAMS];
    float           ** gpu_clt;    //            [NUM_CAMS];
    float           ** gpu_corr_images; //       [NUM_CAMS];
///    float           ** gpu_lpf;    //            [NUM_CAMS]; // never referenced

    // GPU pointers to GPU memory
//    float * gpu_tasks;
    struct tp_task  * gpu_tasks;
    int *             gpu_active_tiles;
    int *             gpu_num_active;
    int *             gpu_num_corr_tiles;

    checkCudaErrors (cudaMalloc((void **)&gpu_active_tiles, TILESX * TILESY * sizeof(int)));
    checkCudaErrors (cudaMalloc((void **)&gpu_num_active,                     sizeof(int)));
    checkCudaErrors (cudaMalloc((void **)&gpu_num_corr_tiles,                 sizeof(int)));

    size_t  dstride;          // in bytes !
    size_t  dstride_rslt;     // in bytes !
    size_t  dstride_corr;     // in bytes ! for one 2d phase correlation (padded 15x15x4 bytes)
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
/*
    float lpf_rbg[3][64]; // not used
    for (int ncol = 0; ncol < 3; ncol++) {
    	if (lpf_sigmas[ncol] > 0.0) {
    		set_clt_lpf (
    				lpf_rbg[ncol], // float * lpf,    // size*size array to be filled out
					lpf_sigmas[ncol], // float   sigma,
					8); // int     dct_size)
    		gpu_lpf_h[ncol] = copyalloc_kernel_gpu(lpf_rbg[ncol], 64);
    	} else {
    		gpu_lpf_h[ncol] = NULL;
    	}
    }
*/
    for (int ncam = 0; ncam < NUM_CAMS; ncam++) {
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
        gpu_clt_h[ncam] = alloc_kernel_gpu(TILESY * TILESX * NUM_COLORS * 4 * DTT_SIZE * DTT_SIZE);
        printf("Allocating GPU memory, 0x%x floats\n", (TILESY * TILESX * NUM_COLORS * 4 * DTT_SIZE * DTT_SIZE)) ;
        // allocate result images (3x height to accommodate 3 colors

        // Image is extended by 4 pixels each side to avoid checking (mclt tiles extend by 4)
        //host array of pointers to GPU arrays
        gpu_corr_images_h[ncam] = alloc_image_gpu(
        		&dstride_rslt,                // size_t* dstride, // in bytes!!
				IMG_WIDTH + DTT_SIZE,         // int width,
				3*(IMG_HEIGHT + DTT_SIZE));   // int height);
    }
    // allocates one correlation kernel per line (15x15 floats), number of rows - number of tiles * number of pairs
    gpu_corrs = alloc_image_gpu(
    		&dstride_corr,                  // in bytes ! for one 2d phase correlation (padded 15x15x4 bytes)
			CORR_SIZE,                      // int width,
			NUM_PAIRS * TILESX * TILESY);   // int height);
    // read channel images (assuming host_kern_buf size > image size, reusing it)
    for (int ncam = 0; ncam < NUM_CAMS; ncam++) {
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

    for (int ncam = 0; ncam < NUM_CAMS; ncam++) {
        readFloatsFromFile(
			    (float *) &tile_coords_h[ncam],
				ports_offs_xy_file[ncam]); // 			   char *  path) // file path
    }

    // build TP task that processes all tiles in linescan order
    for (int ty = 0; ty < TILESY; ty++){
        for (int tx = 0; tx < TILESX; tx++){
            int nt = ty * TILESX + tx;
            task_data[nt].task = 0xf | (((1 << NUM_PAIRS)-1) << TASK_CORR_BITS);
            task_data[nt].txy = tx + (ty << 16);
            for (int ncam = 0; ncam < NUM_CAMS; ncam++) {
                task_data[nt].xy[ncam][0] = tile_coords_h[ncam][nt][0];
                task_data[nt].xy[ncam][1] = tile_coords_h[ncam][nt][1];
                task_data[nt].target_disparity = DBG_DISPARITY;
            }
        }
    }

    int tp_task_size =  sizeof(task_data)/sizeof(struct tp_task);
    int num_active_tiles; // will be calculated by convert_direct




#ifdef DBG0
//#define NUM_TEST_TILES 128
#define NUM_TEST_TILES 1
    for (int t = 0; t < NUM_TEST_TILES; t++) {
    	task_data[t].task = 1;
    	task_data[t].txy = ((DBG_TILE + t) - 324* ((DBG_TILE + t) / 324)) + (((DBG_TILE + t) / 324)) << 16;
    	int nt = task_data[t].ty * TILESX + task_data[t].tx;

    	for (int ncam = 0; ncam < NUM_CAMS; ncam++) {
    		task_data[t].xy[ncam][0] = tile_coords_h[ncam][nt][0];
    		task_data[t].xy[ncam][1] = tile_coords_h[ncam][nt][1];
    	}
    }
    tp_task_size =  NUM_TEST_TILES; // sizeof(task_data)/sizeof(float);

#endif

    // segfault in the next
    gpu_tasks = (struct tp_task  *) copyalloc_kernel_gpu((float * ) &task_data, tp_task_size * (sizeof(struct tp_task)/sizeof(float)));

    // build corr_indices
    num_corrs = 0;
    for (int ty = 0; ty < TILESY; ty++){
    	for (int tx = 0; tx < TILESX; tx++){
    		int nt = ty * TILESX + tx;
    		int cm = (task_data[nt].task >> TASK_CORR_BITS) & ((1 << NUM_PAIRS)-1);
    		if (cm){
    			for (int b = 0; b < NUM_PAIRS; b++) if ((cm & (1 << b)) != 0) {
    				corr_indices[num_corrs++] = (nt << CORR_NTILE_SHIFT) | b;
    			}
    		}
    	}
    }
    // num_corrs now has the total number of correlations
    // copy corr_indices to gpu
    gpu_corr_indices = (int  *) copyalloc_kernel_gpu(
    		(float * ) corr_indices,
			num_corrs,
			NUM_PAIRS * TILESX * TILESY);

    // build texture_indices
    num_textures = 0;
    for (int ty = 0; ty < TILESY; ty++){
    	for (int tx = 0; tx < TILESX; tx++){
    		int nt = ty * TILESX + tx;
//    		int cm = (task_data[nt].task >> TASK_TEXTURE_BIT) & 1;
    		int cm = task_data[nt].task & TASK_TEXTURE_BITS;
    		if (cm){
    			texture_indices[num_textures++] = (nt << CORR_NTILE_SHIFT) | (1 << LIST_TEXTURE_BIT);
    		}
    	}
    }
    // num_textures now has the total number of textures
    // copy corr_indices to gpu
//  gpu_texture_indices = (int  *) copyalloc_kernel_gpu((float * ) texture_indices, num_textures);
    gpu_texture_indices = (int  *) copyalloc_kernel_gpu(
    		(float * ) texture_indices,
			num_textures,
			TILESX * TILESYA); // number of rows - multiple of 4
    // just allocate
    checkCudaErrors(cudaMalloc((void **)&gpu_woi,               4 * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&gpu_num_texture_tiles, 8 * sizeof(float))); // for each subsequence - number of non-border,
    // number of border tiles

    // copy port indices to gpu
    float color_weights [] = {
            0.294118,              // float             weight0,            // scale for R
            0.117647,              // float             weight1,            // scale for B
            0.588235};              // float             weight2,            // scale for G

    gpu_port_offsets = (float *) copyalloc_kernel_gpu((float * ) port_offsets, num_ports * 2);
    gpu_color_weights = (float *) copyalloc_kernel_gpu((float * ) color_weights, sizeof(color_weights));



//    int keep_texture_weights = 1; // try with 0 also
//    int texture_colors = 3; // result will be 3+1 RGBA (for mono - 2)

//		double [][] rgba = new double[numcol + 1 + (keep_weights?(ports + numcol + 1):0)][];

    int tile_texture_size = (texture_colors + 1 + (keep_texture_weights? (NUM_CAMS + texture_colors + 1): 0)) *256;

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
//    checkCudaErrors(cudaMalloc((void **)&gpu_diff_rgb_combo,  TILESX * TILESY * NUM_CAMS * (NUM_COLS+1)* sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&gpu_diff_rgb_combo,  TILESX * TILESY * NUM_CAMS * (NUM_COLORS + 1) * sizeof(float)));

    // Now copy arrays of per-camera pointers to GPU memory to GPU itself

    gpu_kernels =        copyalloc_pointers_gpu (gpu_kernels_h,     NUM_CAMS);
    gpu_kernel_offsets = (struct CltExtra **) copyalloc_pointers_gpu ((float **) gpu_kernel_offsets_h, NUM_CAMS);
    gpu_images =         copyalloc_pointers_gpu (gpu_images_h,      NUM_CAMS);
    gpu_clt =            copyalloc_pointers_gpu (gpu_clt_h,         NUM_CAMS);
    gpu_corr_images =    copyalloc_pointers_gpu (gpu_corr_images_h, NUM_CAMS);

#ifdef DBG_TILE
    const int numIterations = 1; //0;
    const int i0 =  0; // -1;
#else
    const int numIterations = 10; // 0; //0;
    const int i0 = -1; // 0; // -1;
#endif



#define TEST_ROT_MATRICES
#ifdef  TEST_ROT_MATRICES
//    dim3 threads_rot(3,3,NUM_CAMS);
//   dim3 grid_rot   (1, 1, 1);
    dim3 threads_rot(3,3,3);
    dim3 grid_rot   (NUM_CAMS, 1, 1);

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
    			gpu_correction_vector ,           // 		struct corr_vector * gpu_correction_vector,
    			gpu_rot_deriv);                  // union trot_deriv   * gpu_rot_deriv);


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
#if 0
	const char* matrices_names[] = {
	    		"rot","d_daz","d_tilt","d_roll","d_zoom"};
    for (int i = 0; i < 5;i++){
		printf("Matrix %s for camera\n",matrices_names[i]);
		for (int row = 0; row<3; row++){
			for (int ncam = 0; ncam<NUM_CAMS;ncam++){
				for (int col = 0; col <3; col++){

#ifdef NVRTC_BUG
					//abuse - exceeding first dimension
					printf("%9.6f,",rot_deriv.rots[i*NUM_CAMS+ncam][row][col]);
#else
					printf("%9.6f,",rot_deriv.matrices[i][ncam][row][col]);
#endif
					if (col == 2){
						if (ncam == (NUM_CAMS-1)){
							printf("\n");
						} else {
							printf("   ");
						}
					} else {
						printf(" ");
					}
				}
			}
		}
    }
#endif //#if 0


#endif // TEST_ROT_MATRICES

#define TEST_REVERSE_DISTORTIONS
#ifdef  TEST_REVERSE_DISTORTIONS
    dim3 threads_rd(3,3,3);
    dim3 grid_rd   (NUM_CAMS, 1, 1);

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
    dim3 threads_geom(NUM_CAMS,TILES_PER_BLOCK_GEOM, 1);
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
    	get_tiles_offsets<<<grid_geom,threads_geom>>> (
    			gpu_tasks,               // struct tp_task     * gpu_tasks,
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

//    gpu_tasks = (struct tp_task  *) copyalloc_kernel_gpu((float * ) &task_data, tp_task_size * (sizeof(struct tp_task)/sizeof(float)));
//    static struct tp_task     task_data1 [TILESX*TILESY]; // maximal length - each tile
/// DBG_TILE

	checkCudaErrors(cudaMemcpy( // copy modified/calculated tasks
			&task_data1,
			gpu_tasks,
			tp_task_size * sizeof(struct tp_task),
			cudaMemcpyDeviceToHost));
#if 0 // for manual browsing
	struct tp_task * old_task = &task_data [DBG_TILE];
	struct tp_task * new_task = &task_data1[DBG_TILE];
#endif
    printf("old_task txy = 0x%x\n",  task_data [DBG_TILE].txy);
    printf("new_task txy = 0x%x\n",  task_data1[DBG_TILE].txy);

    for (int ncam = 0; ncam < NUM_CAMS; ncam++){
        printf("camera %d pX old %f new %f diff = %f\n", ncam,
        		task_data [DBG_TILE].xy[ncam][0],  task_data1[DBG_TILE].xy[ncam][0],
				task_data [DBG_TILE].xy[ncam][0] - task_data1[DBG_TILE].xy[ncam][0]);
        printf("camera %d pY old %f new %f diff = %f\n", ncam,
        		task_data [DBG_TILE].xy[ncam][1],  task_data1[DBG_TILE].xy[ncam][1],
				task_data [DBG_TILE].xy[ncam][1]-  task_data1[DBG_TILE].xy[ncam][1]);
    }
#if 0
    // temporarily restore tasks
    checkCudaErrors(cudaMemcpy(
    		gpu_tasks,
			&task_data,
			tp_task_size * sizeof(struct tp_task),
            cudaMemcpyHostToDevice));
#endif
#endif // TEST_GEOM_CORR





    //create and start CUDA timer
    StopWatchInterface *timerTP = 0;
    sdkCreateTimer(&timerTP);

#if 0
    dim3 threads_tp(THREADSX, TILES_PER_BLOCK, 1);
    dim3 grid_tp((tp_task_size + TILES_PER_BLOCK -1 )/TILES_PER_BLOCK, 1);
#else
    dim3 threads_tp(1, 1, 1);
    dim3 grid_tp(1, 1, 1);
#endif
    printf("threads_tp=(%d, %d, %d)\n",threads_tp.x,threads_tp.y,threads_tp.z);
    printf("grid_tp=   (%d, %d, %d)\n",grid_tp.x,   grid_tp.y,   grid_tp.z);



//    cudaFuncSetCacheConfig(convert_correct_tiles, cudaFuncCachePreferShared);
//    cudaFuncSetCacheConfig(convert_correct_tiles, cudaFuncCachePreferShared);
    ///    cudaProfilerStart();
    float ** fgpu_kernel_offsets = (float **) gpu_kernel_offsets; //    [NUM_CAMS];

    for (int i = i0; i < numIterations; i++)
    {
    	if (i == 0)
    	{
    		checkCudaErrors(cudaDeviceSynchronize());
    		sdkResetTimer(&timerTP);
    		sdkStartTimer(&timerTP);
    	}
    	convert_direct<<<grid_tp,threads_tp>>>( // called with a single block, CONVERT_DIRECT_INDEXING_THREADS threads
    			fgpu_kernel_offsets,   // struct CltExtra ** gpu_kernel_offsets,
				gpu_kernels,           // float           ** gpu_kernels,
				gpu_images,            // float           ** gpu_images,
				gpu_tasks,             // struct tp_task   * gpu_tasks,
				gpu_clt,               // float           ** gpu_clt,            // [NUM_CAMS][TILESY][TILESX][NUM_COLORS][DTT_SIZE*DTT_SIZE]
				dstride/sizeof(float), // size_t             dstride, // for gpu_images
				tp_task_size,          // int                num_tiles) // number of tiles in task
				0,                     // int                lpf_mask)            // apply lpf to colors : bit 0 - red, bit 1 - blue, bit2 - green
				IMG_WIDTH,             // int                woi_width,
				IMG_HEIGHT,            // int                woi_height,
				KERNELS_HOR,           // int                kernels_hor,
				KERNELS_VERT,          // int                kernels_vert);
				gpu_active_tiles,      // int *              gpu_active_tiles,      // pointer to the calculated number of non-zero tiles
    			gpu_num_active);       // int *              pnum_active_tiles);  //  indices to gpu_tasks

    	getLastCudaError("Kernel execution failed");
    	checkCudaErrors(cudaDeviceSynchronize());
    	printf("%d\n",i);
    }
    //    checkCudaErrors(cudaDeviceSynchronize());
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
    int rslt_size = (TILESY * TILESX * NUM_COLORS * 4 * DTT_SIZE * DTT_SIZE);
    float * cpu_clt = (float *)malloc(rslt_size*sizeof(float));
    for (int ncam = 0; ncam < NUM_CAMS; ncam++) {
    	checkCudaErrors(cudaMemcpy( // segfault
    			cpu_clt,
				gpu_clt_h[ncam],
				rslt_size * sizeof(float),
    			cudaMemcpyDeviceToHost));
#ifndef DBG_TILE
        printf("Writing CLT data to %s\n",  ports_clt_file[ncam]);
    	writeFloatsToFile(cpu_clt, // float *       data, // allocated array
    			rslt_size, // int           size, // length in elements
				ports_clt_file[ncam]); // 			   const char *  path) // file path
#endif
    }
#endif

#ifdef TEST_IMCLT
     {
    	// testing imclt
    	dim3 threads_imclt(IMCLT_THREADS_PER_TILE, IMCLT_TILES_PER_BLOCK, 1);
    	dim3 grid_imclt(1,1,1);
    	printf("threads_imclt=(%d, %d, %d)\n",threads_imclt.x,threads_imclt.y,threads_imclt.z);
    	printf("grid_imclt=   (%d, %d, %d)\n",grid_imclt.x,   grid_imclt.y,   grid_imclt.z);
    	for (int ncam = 0; ncam < NUM_CAMS; ncam++) {
    		test_imclt<<<grid_imclt,threads_imclt>>>(
    				gpu_clt_h[ncam], // ncam]); //                //       float           ** gpu_clt,            // [NUM_CAMS][TILESY][TILESX][NUM_COLORS][DTT_SIZE*DTT_SIZE]
					ncam);                                        // int             ncam); // just for debug print
    	}
    	getLastCudaError("Kernel execution failed");
    	checkCudaErrors(cudaDeviceSynchronize());
    	printf("test_imclt() DONE\n");
    }
#endif


    // testing imclt
//    dim3 threads_imclt(IMCLT_THREADS_PER_TILE, IMCLT_TILES_PER_BLOCK, 1);
//    printf("threads_imclt=(%d, %d, %d)\n",threads_imclt.x,threads_imclt.y,threads_imclt.z);
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
        		gpu_clt,                     // float           ** gpu_clt,            // [NUM_CAMS][TILESY][TILESX][NUM_COLORS][DTT_SIZE*DTT_SIZE]
				gpu_corr_images,             // float           ** gpu_corr_images,    // [NUM_CAMS][WIDTH, 3 * HEIGHT]
				1,                           // int               apply_lpf,
				NUM_COLORS,                  // int               colors,               // defines lpf filter
				TILESX,                      // int               woi_twidth,
				TILESY,                      // int               woi_theight,
				dstride_rslt/sizeof(float)); // const size_t      dstride);            // in floats (pixels)
    	getLastCudaError("Kernel failure");
    	checkCudaErrors(cudaDeviceSynchronize());
    	printf("test pass: %d\n",i);
    }

    sdkStopTimer(&timerIMCLT);
    float avgTimeIMCLT = (float)sdkGetTimerValue(&timerIMCLT) / (float)numIterations;
    sdkDeleteTimer(&timerIMCLT);
    printf("Average IMCLT run time =%f ms\n",  avgTimeIMCLT);

    int rslt_img_size =       NUM_COLORS * (IMG_HEIGHT + DTT_SIZE) * (IMG_WIDTH + DTT_SIZE);
    float * cpu_corr_image = (float *)malloc(rslt_img_size * sizeof(float));



    for (int ncam = 0; ncam < NUM_CAMS; ncam++) {
    	checkCudaErrors(cudaMemcpy2D( // segfault
    			cpu_corr_image,
				(IMG_WIDTH + DTT_SIZE) * sizeof(float),
				gpu_corr_images_h[ncam],
				dstride_rslt,
				(IMG_WIDTH + DTT_SIZE) * sizeof(float),
				3* (IMG_HEIGHT + DTT_SIZE),
    			cudaMemcpyDeviceToHost));

#ifndef DBG_TILE
        printf("Writing RBG data to %s\n",  result_rbg_file[ncam]);
    	writeFloatsToFile( // will have margins
    			cpu_corr_image, // float *       data, // allocated array
				rslt_img_size, // int           size, // length in elements
				result_rbg_file[ncam]); // 			   const char *  path) // file path
#endif
    }

    free(cpu_corr_image);


#ifndef NOCORR
//    cudaProfilerStart();
    // testing corr
//    dim3 threads_corr(CORR_THREADS_PER_TILE, CORR_TILES_PER_BLOCK, 1);
    //        dim3 grid_corr((num_corrs + CORR_TILES_PER_BLOCK-1) / CORR_TILES_PER_BLOCK,1,1);
 //   printf("threads_corr=(%d, %d, %d)\n",threads_corr.x,threads_corr.y,threads_corr.z);
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
		gpu_clt,                    // float          ** gpu_clt,            // [NUM_CAMS] ->[TILESY][TILESX][NUM_COLORS][DTT_SIZE*DTT_SIZE]
		3,                          // int               colors,             // number of colors (3/1)
		0.25,                       // float             scale0,             // scale for R
		0.25,                       // float             scale1,             // scale for B
		0.5,                        // float             scale2,             // scale for G
		30.0,                       // float             fat_zero,           // here - absolute
		gpu_tasks,                  // struct tp_task  * gpu_tasks,
		tp_task_size,               // int               num_tiles) // number of tiles in task
		gpu_corr_indices,           //  int            * gpu_corr_indices,   // packed tile+pair
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
    printf("Average CORR run time =%f ms, num cor tiles (old) = %d\n",  avgTimeCORR, num_corrs);

	checkCudaErrors(cudaMemcpy(
			&num_corrs,
			gpu_num_corr_tiles,
			sizeof(int),
			cudaMemcpyDeviceToHost));
    printf("Average CORR run time =%f ms, num cor tiles (new) = %d\n",  avgTimeCORR, num_corrs);

    int corr_size =        2 * CORR_OUT_RAD + 1;
    int rslt_corr_size =   num_corrs * corr_size * corr_size;
    float * cpu_corr = (float *)malloc(rslt_corr_size * sizeof(float));

    checkCudaErrors(cudaMemcpy2D(
    		cpu_corr,
			(corr_size * corr_size) * sizeof(float),
			gpu_corrs,
			dstride_corr,
			(corr_size * corr_size) * sizeof(float),
			num_corrs,
    		cudaMemcpyDeviceToHost));

#ifndef NSAVE_CORR
    		printf("Writing phase correlation data to %s\n",  result_corr_file);
    		writeFloatsToFile(
    				cpu_corr,    // float *       data, // allocated array
					rslt_corr_size,    // int           size, // length in elements
					result_corr_file); // 			   const char *  path) // file path
#endif
    		free(cpu_corr);
#endif // ifndef NOCORR


// -----------------

#ifndef NOTEXTURES
//    cudaProfilerStart();
    // testing textures
    dim3 threads_texture(TEXTURE_THREADS_PER_TILE, NUM_CAMS, 1); // TEXTURE_TILES_PER_BLOCK, 1);
    dim3 grid_texture((num_textures + TEXTURE_TILES_PER_BLOCK-1) / TEXTURE_TILES_PER_BLOCK,1,1);
    printf("threads_texture=(%d, %d, %d)\n",threads_texture.x,threads_texture.y,threads_texture.z);
    printf("grid_texture=(%d, %d, %d)\n",grid_texture.x,grid_texture.y,grid_texture.z);
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

		// Channel0 weight = 0.294118
		// Channel1 weight = 0.117647
		// Channel2 weight = 0.588235
    	textures_nonoverlap<<<1,1>>> (
                gpu_tasks,             // struct tp_task   * gpu_tasks,
                tp_task_size,          // int                num_tiles,          // number of tiles in task list
    	// declare arrays in device code?
				gpu_texture_indices,   // int             * gpu_texture_indices,// packed tile + bits (now only (1 << 7)
				gpu_num_texture_tiles, // int             * pnum_texture_tiles,  // returns total number of elements in gpu_texture_indices array
		        gpu_clt ,              // float          ** gpu_clt,            // [NUM_CAMS] ->[TILESY][TILESX][NUM_COLORS][DTT_SIZE*DTT_SIZE]
    			// TODO: use geometry_correction rXY !
				gpu_geometry_correction, // struct gc     * gpu_geometry_correction,
				texture_colors,        // int               colors,             // number of colors (3/1)
				(texture_colors == 1), // int               is_lwir,            // do not perform shot correction
				10.0,                  // float             min_shot,           // 10.0
				3.0,                   // float             scale_shot,         // 3.0
				1.5f,                  // float             diff_sigma,         // pixel value/pixel change
				10.0f,                 // float             diff_threshold,     // pixel value/pixel change
				3.0,                   // float             min_agree,          // minimal number of channels to agree on a point (real number to work with fuzzy averages)
				gpu_color_weights,     // float             weights[3],         // scale for R
				1,                     // int               dust_remove,        // Do not reduce average weight when only one image differes much from the average
    	// combining both non-overlap and overlap (each calculated if pointer is not null )
				0, // dstride_textures/sizeof(float), // size_t            texture_stride,     // in floats (now 256*4 = 1024)  // may be 0 if not needed
//				gpu_textures,         // float           * gpu_texture_tiles,  // (number of colors +1 + ?)*16*16 rgba texture tiles    // may be 0 if not needed
				(float *) 0,          // gpu_textures,         // float           * gpu_texture_tiles,  // (number of colors +1 + ?)*16*16 rgba texture tiles    // may be 0 if not needed
				gpu_diff_rgb_combo);  // float           * gpu_diff_rgb_combo); // diff[NUM_CAMS], R[NUM_CAMS], B[NUM_CAMS],G[NUM_CAMS] // may be 0 if not needed
    	getLastCudaError("Kernel failure");
    	checkCudaErrors(cudaDeviceSynchronize());
    	printf("test pass: %d\n",i);
    }
///	cudaProfilerStop();
    sdkStopTimer(&timerTEXTURE);
    float avgTimeTEXTURES = (float)sdkGetTimerValue(&timerTEXTURE) / (float)numIterations;
    sdkDeleteTimer(&timerTEXTURE);
    printf("Average Texture run time =%f ms\n",  avgTimeTEXTURES);

    int rslt_texture_size =   num_textures * tile_texture_size;
    float * cpu_textures = (float *)malloc(rslt_texture_size * sizeof(float));



    checkCudaErrors(cudaMemcpy2D(
    		cpu_textures,
			tile_texture_size * sizeof(float),
			gpu_textures,
			dstride_textures,
			tile_texture_size * sizeof(float),
			num_textures,
    		cudaMemcpyDeviceToHost));

#ifndef NSAVE_TEXTURES
    		printf("Writing phase texture data to %s\n",  result_textures_file);
    		writeFloatsToFile(
    				cpu_textures,    // float *       data, // allocated array
					rslt_texture_size,    // int           size, // length in elements
					result_textures_file); // 			   const char *  path) // file path

//DBG_TILE
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
//    int tile_texture_size = (texture_colors + 1 + (keep_texture_weights? (NUM_CAMS + texture_colors + 1): 0)) *256;
#endif // DEBUG9
#endif
    		free(cpu_textures);
#endif // ifndef NOTEXTURES


#undef GEN_TEXTURE_LIST
#ifdef  GEN_TEXTURE_LIST
    		dim3 threads_list(1,1, 1); // TEXTURE_TILES_PER_BLOCK, 1);
    		dim3 grid_list   (1,1,1);
    		printf("threads_list=(%d, %d, %d)\n",threads_list.x,threads_list.y,threads_list.z);
    		printf("grid_list=(%d, %d, %d)\n",grid_list.x,grid_list.y,grid_list.z);
    		StopWatchInterface *timerTEXTURELIST = 0;
    		sdkCreateTimer(&timerTEXTURELIST);
    		for (int i = i0; i < numIterations; i++)
    		{
    			if (i == 0)
    			{
    				checkCudaErrors(cudaDeviceSynchronize());
    				sdkResetTimer(&timerTEXTURELIST);
    				sdkStartTimer(&timerTEXTURELIST);
    			}

    			prepare_texture_list<<<grid_list,threads_list>>> (
    					gpu_tasks,             // struct tp_task   * gpu_tasks,
						tp_task_size,          // int                num_tiles,          // number of tiles in task list
						gpu_texture_indices,   // int              * gpu_texture_indices,// packed tile + bits (now only (1 << 7)
						gpu_num_texture_tiles, // int              * num_texture_tiles,  // number of texture tiles to process (8 elements)
						gpu_woi,               // int              * woi,                // x,y,width,height of the woi
						TILESX,                // int                width,  // <= TILESX, use for faster processing of LWIR images (should be actual + 1)
						TILESY);               // int                height); // <= TILESY, use for faster processing of LWIR images

    			getLastCudaError("Kernel failure");
    			checkCudaErrors(cudaDeviceSynchronize());
    			printf("test pass: %d\n",i);
    		}
    		///	cudaProfilerStop();
    		sdkStopTimer(&timerTEXTURELIST);
    		float avgTimeTEXTURESLIST = (float)sdkGetTimerValue(&timerTEXTURELIST) / (float)numIterations;
    		sdkDeleteTimer(&timerTEXTURELIST);
    		printf("Average TextureList run time =%f ms\n",  avgTimeTEXTURESLIST);

    		int cpu_num_texture_tiles[8];
    		checkCudaErrors(cudaMemcpy(
    				cpu_woi,
					gpu_woi,
					4 * sizeof(float),
					cudaMemcpyDeviceToHost));
    		printf("WOI x=%d, y=%d, width=%d, height=%d\n", cpu_woi[0], cpu_woi[1], cpu_woi[2], cpu_woi[3]);
    		checkCudaErrors(cudaMemcpy(
    				cpu_num_texture_tiles,
					gpu_num_texture_tiles,
					8 * sizeof(float), // 8 sequences (0,2,4,6 - non-border, growing up;
					//1,3,5,7 - border, growing down from the end of the corresponding non-border buffers
					cudaMemcpyDeviceToHost));
    		printf("cpu_num_texture_tiles=(%d(%d), %d(%d), %d(%d), %d(%d) -> %d tp_task_size=%d)\n",
    				cpu_num_texture_tiles[0], cpu_num_texture_tiles[1],
					cpu_num_texture_tiles[2], cpu_num_texture_tiles[3],
					cpu_num_texture_tiles[4], cpu_num_texture_tiles[5],
					cpu_num_texture_tiles[6], cpu_num_texture_tiles[7],
    				cpu_num_texture_tiles[0] + cpu_num_texture_tiles[1] +
					cpu_num_texture_tiles[2] + cpu_num_texture_tiles[3] +
					cpu_num_texture_tiles[4] + cpu_num_texture_tiles[5] +
					cpu_num_texture_tiles[6] + cpu_num_texture_tiles[7],
					tp_task_size
					);
    		for (int q = 0; q < 4; q++) {
    			checkCudaErrors(cudaMemcpy(
    					texture_indices  + q * TILESX * (TILESYA >> 2),
						gpu_texture_indices  + q * TILESX * (TILESYA >> 2),
						cpu_num_texture_tiles[q] * sizeof(float), // change to cpu_num_texture_tiles when ready
						cudaMemcpyDeviceToHost));
    		}
    		for (int q = 0; q < 4; q++) {
        		printf("%d: %3x:%3x %3x:%3x %3x:%3x %3x:%3x %3x:%3x %3x:%3x %3x:%3x %3x:%3x \n",q,
        				(texture_indices[q * TILESX * (TILESYA >> 2) + 0] >> 8) / TILESX, (texture_indices[q * TILESX * (TILESYA >> 2) + 0] >> 8) % TILESX,
        				(texture_indices[q * TILESX * (TILESYA >> 2) + 1] >> 8) / TILESX, (texture_indices[q * TILESX * (TILESYA >> 2) + 1] >> 8) % TILESX,
        				(texture_indices[q * TILESX * (TILESYA >> 2) + 2] >> 8) / TILESX, (texture_indices[q * TILESX * (TILESYA >> 2) + 2] >> 8) % TILESX,
        				(texture_indices[q * TILESX * (TILESYA >> 2) + 3] >> 8) / TILESX, (texture_indices[q * TILESX * (TILESYA >> 2) + 3] >> 8) % TILESX,
        				(texture_indices[q * TILESX * (TILESYA >> 2) + 4] >> 8) / TILESX, (texture_indices[q * TILESX * (TILESYA >> 2) + 4] >> 8) % TILESX,
        				(texture_indices[q * TILESX * (TILESYA >> 2) + 5] >> 8) / TILESX, (texture_indices[q * TILESX * (TILESYA >> 2) + 5] >> 8) % TILESX,
        				(texture_indices[q * TILESX * (TILESYA >> 2) + 6] >> 8) / TILESX, (texture_indices[q * TILESX * (TILESYA >> 2) + 6] >> 8) % TILESX,
        				(texture_indices[q * TILESX * (TILESYA >> 2) + 7] >> 8) / TILESX, (texture_indices[q * TILESX * (TILESYA >> 2) + 7] >> 8) % TILESX);
    		}
#endif //GEN_TEXTURE_LIST



#ifndef NOTEXTURE_RGBA
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

    	generate_RBGA<<<1,1>>> (
    	// Parameters to generate texture tasks
                gpu_tasks,             // struct tp_task   * gpu_tasks,
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
	            10.0,                  // float             min_shot,           // 10.0
	            3.0,                   // float             scale_shot,         // 3.0
	            1.5f,                  // float             diff_sigma,         // pixel value/pixel change
	            10.0f,                 // float             diff_threshold,     // pixel value/pixel change
	            3.0,                   // float             min_agree,          // minimal number of channels to agree on a point (real number to work with fuzzy averages)
				gpu_color_weights,     // float             weights[3],         // scale for R
	            1,                     // int               dust_remove,        // Do not reduce average weight when only one image differes much from the average
	            0,                     // int               keep_weights,       // return channel weights after A in RGBA
				dstride_textures_rbga/sizeof(float), // 	const size_t      texture_rbga_stride,     // in floats
				gpu_textures_rbga);     // 	float           * gpu_texture_tiles)    // (number of colors +1 + ?)*16*16 rgba texture tiles
//				(float *) 0 ); // gpu_diff_rgb_combo);   // float           * gpu_diff_rgb_combo) // diff[NUM_CAMS], R[NUM_CAMS], B[NUM_CAMS],G[NUM_CAMS]

    	getLastCudaError("Kernel failure");
    	checkCudaErrors(cudaDeviceSynchronize());
    	printf("test pass: %d\n",i);
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
    printf("Writing RBGA texture slices to %s\n",  result_textures_rgba_file);
    writeFloatsToFile(
    		cpu_textures_rgba,    // float *       data, // allocated array
			rslt_rgba_size,    // int           size, // length in elements
			result_textures_rgba_file); // 			   const char *  path) // file path
#endif
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
    free(cpu_textures_rgba);
#endif // ifndef NOTEXTURES




#ifdef SAVE_CLT
    free(cpu_clt);
#endif

    free (host_kern_buf);
    // TODO: move somewhere when all is done
    for (int ncam = 0; ncam < NUM_CAMS; ncam++) {
    	checkCudaErrors(cudaFree(gpu_kernels_h[ncam]));
    	checkCudaErrors(cudaFree(gpu_kernel_offsets_h[ncam]));
    	checkCudaErrors(cudaFree(gpu_images_h[ncam]));
    	checkCudaErrors(cudaFree(gpu_clt_h[ncam]));
    	checkCudaErrors(cudaFree(gpu_corr_images_h[ncam]));
    }
	checkCudaErrors(cudaFree(gpu_tasks));
	checkCudaErrors(cudaFree(gpu_active_tiles));
	checkCudaErrors(cudaFree(gpu_num_active));
	checkCudaErrors(cudaFree(gpu_kernels));
	checkCudaErrors(cudaFree(gpu_kernel_offsets));
	checkCudaErrors(cudaFree(gpu_images));
	checkCudaErrors(cudaFree(gpu_clt));
	checkCudaErrors(cudaFree(gpu_corr_images));
	checkCudaErrors(cudaFree(gpu_corrs));
	checkCudaErrors(cudaFree(gpu_corr_indices));
	checkCudaErrors(cudaFree(gpu_num_corr_tiles));
	checkCudaErrors(cudaFree(gpu_texture_indices));
	checkCudaErrors(cudaFree(gpu_port_offsets));
	checkCudaErrors(cudaFree(gpu_color_weights));
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

	exit(0);
}
