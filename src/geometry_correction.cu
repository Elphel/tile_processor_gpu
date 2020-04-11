/**
 **
 ** geometry_correction.cu
 **
 ** Copyright (C) 2020 Elphel, Inc.
 **
 ** -----------------------------------------------------------------------------**
 **
 **  geometry_correction.cu is free software: you can redistribute it and/or modify
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
* \file geometry_correction.cu
* \brief header file for geometry correction - per-tile/per camera calculation of the tile offset

*/
#ifndef JCUDA
#include "tp_defines.h"
#include "dtt8x8.h"
#include "geometry_correction.h"
#endif // #ifndef JCUDA

// Using NUM_CAMS threads per tile
#define THREADS_PER_BLOCK_GEOM (TILES_PER_BLOCK_GEOM * NUM_CAMS)
#define CYCLES_COPY_GC   ((sizeof(struct gc)/sizeof(float) + THREADS_PER_BLOCK_GEOM - 1) / THREADS_PER_BLOCK_GEOM)
#define CYCLES_COPY_CV   ((sizeof(struct corr_vector)/sizeof(float) + THREADS_PER_BLOCK_GEOM - 1) / THREADS_PER_BLOCK_GEOM)
#define CYCLES_COPY_RBRD ((RBYRDIST_LEN + THREADS_PER_BLOCK_GEOM - 1) / THREADS_PER_BLOCK_GEOM)
#define CYCLES_COPY_ROTS ((NUM_CAMS * 3 *3 + THREADS_PER_BLOCK_GEOM - 1) / THREADS_PER_BLOCK_GEOM)

#define DBG_CAM 3

__device__ void printGeometryCorrection(struct gc * g);
__device__ void printExtrinsicCorrection(corr_vector * cv);
/**
 * Calculate non-distorted radius from distorted using table approximation
 * @param rDist distorted radius
 * @return corresponding non-distorted radius
 */
inline __device__ float getRByRDist(float rDist,
		float rByRDist [RBYRDIST_LEN]); //shared memory

__constant__ float ROTS_TEMPLATE[7][3][3][3] = {//  ...{cos,sin,const}...
		{ // azimuth
				{{ 1, 0,0},{0, 0,0},{ 0,-1,0}},
				{{ 0, 0,0},{0, 0,1},{ 0, 0,0}},
				{{ 0, 1,0},{0, 0,0},{ 1, 0,0}},

		},{ // tilt
				{{ 0, 0,1},{0, 0,0},{ 0, 0,0}},
				{{ 0, 0,0},{1, 0,0},{ 0, 1,0}},
				{{ 0, 0,0},{0,-1,0},{ 1, 0,0}},
		},{ // roll*zoom
				{{ 1, 0,0},{0, 1,0},{ 0, 0,0}},
				{{ 0,-1,0},{1, 0,0},{ 0, 0,0}},
				{{ 0, 0,0},{0, 0,0},{ 0, 0,1}},

		},{ // d_azimuth
				{{ 0,-1,0},{0, 0,0},{-1, 0,0}},
				{{ 0, 0,0},{0, 0,0},{ 0, 0,0}},
				{{ 1, 0,0},{0, 0,0},{ 0,-1,0}},
		},{ // d_tilt
				{{ 0, 0,0},{0, 0,0},{ 0, 0,0}},
				{{ 0, 0,0},{0,-1,0},{ 1, 0,0}},
				{{ 0, 0,0},{-1,0,0},{ 0,-1,0}},
		},{ // d_roll
				{{ 0,-1,0},{1, 0,0},{ 0, 0,0}},
				{{-1, 0,0},{0,-1,0},{ 0, 0,0}},
				{{ 0, 0,0},{0, 0,0},{ 0, 0,0}},
		},{ // d_zoom
				{{ 1, 0,0},{0, 1,0},{ 0, 0,0}},
				{{ 0,-1,0},{1, 0,0},{ 0, 0,0}},
				{{ 0, 0,0},{0, 0,0},{ 0, 0,0}},
		}
};

__constant__ int angles_offsets [4] = {
		offsetof(corr_vector, azimuth)/sizeof(float),
		offsetof(corr_vector, tilt)   /sizeof(float),
		offsetof(corr_vector, roll)   /sizeof(float),
		offsetof(corr_vector, roll)   /sizeof(float)};
__constant__ int mm_seq [3][3][3]={
		{
				{6,5,12}, // a_t * a_z -> tmp0
				{7,6,13}, // a_r * a_t -> tmp1
				{7,9,14}, // a_r * a_dt -> tmp2
		}, {
				{7,12,0}, // a_r * tmp0 -> rot          - bad
				{13,8,1}, // tmp1 * a_daz -> deriv0     - good
				{14,5,2}, // tmp2 * a_az  -> deriv1     - good
		}, {
				{10,12,3}, // a_dr * tmp0 -> deriv2     - good
				{11,12,4}, // a_dzoom * tnmp0 -> deriv3 - good
				{-1,-1,-1} // do nothing
		}};


__device__ float rot_matrices       [NUM_CAMS][3][3];
//__device__ float rot_deriv_matrices [NUM_CAMS][4][3][3]; // /d_azimuth, /d_tilt, /d_roll, /d_zoom)


// threads (3,3,4)
extern "C" __global__ void calc_rot_matrices(
		struct corr_vector * gpu_correction_vector)
{
	__shared__ float zoom    [NUM_CAMS];
	__shared__ float sincos  [NUM_CAMS][3][2];    // {az,tilt,roll, d_az, d_tilt, d_roll, d_az}{cos,sin}
	__shared__ float matrices[NUM_CAMS][4][3][3]; // [7] - extra

	float angle;
	int ncam = threadIdx.z;
	int nangle1 = threadIdx.x + threadIdx.y * blockDim.x; // * >> 1;
	int nangle =  nangle1 >> 1;
	int is_sin = nangle1 & 1;

#ifdef DEBUG20a
	if ((threadIdx.x == 0)  && ( threadIdx.y == 0)  && ( threadIdx.z == 0)){
		printf("\nget_tiles_offsets() threadIdx.x = %d, blockIdx.x= %d\n", (int)threadIdx.x, (int) blockIdx.x);
		printExtrinsicCorrection(gpu_correction_vector);
	}
	__syncthreads();// __syncwarp();
#endif // DEBUG20


	if (nangle < 4){ // this part only for 1-st 3
		float* gangles =
				(nangle ==0)?gpu_correction_vector->azimuth:(
						(nangle ==1)?gpu_correction_vector->tilt:(
								(nangle ==2)?gpu_correction_vector->roll:
										gpu_correction_vector->zoom));
		if ((ncam < (NUM_CAMS -1)) || (nangle == 2)){ // for rolls - all 4
			angle = *(gangles + ncam);

		} else {
			angle = 0.0f;

#pragma	unroll
			for (int n = 0; n < (NUM_CAMS-1); n++){
				angle -= *(gangles + n);
			}
		}
		if (!is_sin){
			angle += M_PI/2;
		}
		if (nangle < 3) {
			sincos[ncam][nangle][is_sin]=sinf(angle);
		} else if (is_sin){
			zoom[ncam] = angle;
		}
	}
	__syncthreads();


#ifdef DEBUG20a
	if ((threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0)){
		for (int n = 0; n < NUM_CAMS; n++){
			printf("\n    Azimuth matrix for camera %d, sincos[0] = %f, sincos[1] = %f, zoom = %f\n", n, sincos[n][0][0], sincos[n][0][1], zoom[n]);
			printf("    Tilt matrix for camera %d, sincos[0] = %f, sincos[0] = %f\n", n, sincos[n][1][0], sincos[n][1][1]);
			printf("    Roll matrix for camera %d, sincos[0] = %f, sincos[2] = %f\n", n, sincos[n][2][0], sincos[n][2][1]);
		}
	}
	__syncthreads();// __syncwarp();
#endif // DEBUG20


	if (nangle == 3) {
		sincos[ncam][2][is_sin] *= (1.0 + zoom[ncam]); // modify roll
	}
	__syncthreads();


#ifdef DEBUG20a
	if ((threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0)){
		for (int n = 0; n < NUM_CAMS; n++){
			printf("\na    Azimuth matrix for camera %d, sincos[0] = %f, sincos[1] = %f, zoom = %f\n", n, sincos[n][0][0], sincos[n][0][1], zoom[n]);
			printf("a    Tilt matrix for camera %d, sincos[0] = %f, sincos[0] = %f\n", n, sincos[n][1][0], sincos[n][1][1]);
			printf("a    Roll matrix for camera %d, sincos[0] = %f, sincos[2] = %f\n", n, sincos[n][2][0], sincos[n][2][1]);
		}
	}
	__syncthreads();// __syncwarp();
#endif // DEBUG20





	// now 3x3
	for (int axis = 0; axis < 3; axis++) {
		matrices[ncam][axis][threadIdx.y][threadIdx.x] =
				ROTS_TEMPLATE[axis][threadIdx.y][threadIdx.x][0] * sincos[ncam][axis][0]+ // cos
				ROTS_TEMPLATE[axis][threadIdx.y][threadIdx.x][1] * sincos[ncam][axis][1]+ // sin
				ROTS_TEMPLATE[axis][threadIdx.y][threadIdx.x][2];                         // const
	}
	__syncthreads();


#ifdef DEBUG20a
	if ((threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0)){
		for (int n = 0; n < NUM_CAMS; n++){

			printf("\n1-Azimuth matrix for camera %d, sincos[0] = %f, sincos[1] = %f\n", n, sincos[n][0][0], sincos[n][0][1]);
			for (int i = 0; i < 3; i++){
				for (int j = 0; j < 3; j++){
					printf("%9.6f, ", matrices[n][0][i][j]);
				}
				printf("\n");
			}

			printf("1-Tilt matrix for camera %d, sincos[0] = %f, sincos[1] = %f\n", n, sincos[n][1][0], sincos[n][1][1]);
			for (int i = 0; i < 3; i++){
				for (int j = 0; j < 3; j++){
					printf("%9.6f, ", matrices[n][1][i][j]);
				}
				printf("\n");
			}

			printf("1-Roll/Zoom matrix for camera %d, sincos[0] = %f, sincos[1] = %f\n", n, sincos[n][2][0], sincos[n][2][1]);
			for (int i = 0; i < 3; i++){
				for (int j = 0; j < 3; j++){
					printf("%9.6f, ", matrices[n][2][i][j]);
				}
				printf("\n");
			}

		}
	}
	__syncthreads();// __syncwarp();
#endif // DEBUG20





    // tilt * az ->
	// multiply matrices[ncam][1] * matrices[ncam][0] -> matrices[ncam][3]
	matrices[ncam][3][threadIdx.y][threadIdx.x] =
			matrices[ncam][1][threadIdx.y][0] * matrices[ncam][0][0][threadIdx.x]+
			matrices[ncam][1][threadIdx.y][1] * matrices[ncam][0][1][threadIdx.x]+
			matrices[ncam][1][threadIdx.y][2] * matrices[ncam][0][2][threadIdx.x];
	// multiply matrices[ncam][2] * matrices[ncam][3] -> rot_matrices[ncam]
	__syncthreads();
	rot_matrices[ncam][threadIdx.y][threadIdx.x] =
			matrices[ncam][2][threadIdx.y][0] * matrices[ncam][3][0][threadIdx.x]+
			matrices[ncam][2][threadIdx.y][1] * matrices[ncam][3][1][threadIdx.x]+
			matrices[ncam][2][threadIdx.y][2] * matrices[ncam][3][2][threadIdx.x];
	__syncthreads();


#ifdef DEBUG20
	if ((threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0)){
		for (int n = 0; n < NUM_CAMS; n++){

			printf("\n2 - Azimuth matrix for camera %d, sincos[0] = %f, sincos[1] = %f\n", n, sincos[n][0][0], sincos[n][0][1]);
			for (int i = 0; i < 3; i++){
				for (int j = 0; j < 3; j++){
					printf("%9.6f, ", matrices[n][0][i][j]);
				}
				printf("\n");
			}

			printf("2 - Tilt matrix for camera %d, sincos[0] = %f, sincos[1] = %f\n", n, sincos[n][1][0], sincos[n][1][1]);
			for (int i = 0; i < 3; i++){
				for (int j = 0; j < 3; j++){
					printf("%9.6f, ", matrices[n][1][i][j]);
				}
				printf("\n");
			}

			printf("2 - Roll/Zoom matrix for camera %d, sincos[0] = %f, sincos[1] = %f\n", n, sincos[n][2][0], sincos[n][2][1]);
			for (int i = 0; i < 3; i++){
				for (int j = 0; j < 3; j++){
					printf("%9.6f, ", matrices[n][2][i][j]);
				}
				printf("\n");
			}

			printf("2 - Rotation matrix for camera %d\n", n);
			for (int i = 0; i < 3; i++){
				for (int j = 0; j < 3; j++){
					printf("%9.6f, ", rot_matrices[n][i][j]);
				}
				printf("\n");
			}
		}
	}
	__syncthreads();// __syncwarp();
#endif // DEBUG20


}

__constant__ int offset_rots =     0;                   //0
__constant__ int offset_derivs =   1;                   // 1..4 // should be next
__constant__ int offset_matrices = 5;   // 5..11
__constant__ int offset_tmp =      12; // 12..15

/**
 * Calculate rotation matrices and derivatives by az, tilt, roll, zoom
 * NUM_CAMS blocks of 3,3,3 tiles
 */
extern "C" __global__ void calc_rot_deriv(
		struct corr_vector * gpu_correction_vector,
		union trot_deriv   * gpu_rot_deriv)
{
//	__shared__ float zoom;
	__shared__ float sincos  [4][2];    // {az,tilt,roll, d_az, d_tilt, d_roll, d_az}{cos,sin}
	__shared__ float matrices[5 + 7 +4][3][3];
	float angle;
	float zoom;
	int ncam =    blockIdx.x; // threadIdx.z;
	int nangle1 = threadIdx.x + threadIdx.y * blockDim.x; // * >> 1;
	int nangle =  nangle1 >> 1;
	int is_sin = nangle1 & 1;
	if ((threadIdx.z == 0) && (nangle < 4)){ // others just idle here
		float * gangles = (float *) gpu_correction_vector + angles_offsets[nangle]; // pointer for channel 0
		if (ncam == (NUM_CAMS-1)){ // for the whole block
			angle = 0.0;
			zoom = 0.0;
#pragma	unroll
			for (int n = 0; n < (NUM_CAMS-1); n++){
				angle -= *(gangles + n);
				zoom -= gpu_correction_vector->zoom[n];
			}
			if (nangle >= 2){ // diverging for roll (last two)
				angle = *(gangles + ncam);
			}

		} else {
			angle = *(gangles + ncam);
			zoom =   gpu_correction_vector->zoom[ncam];
		}
		if (!is_sin){
			angle += M_PI/2;
		}
		float sc = sinf(angle);
		if (nangle ==2) {
			sc *= 1.0 + zoom;
		}
		sincos[nangle][is_sin]= sc;
	}
	__syncthreads();

#ifdef DEBUG20
	if ((ncam == DBG_CAM) && (threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0)){
		printf("\n    Azimuth matrix for   camera %d, sincos[0] = %f, sincos[1] = %f, zoom = %f\n", ncam, sincos[0][0], sincos[0][1], zoom);
		printf(  "    Tilt matrix for      camera %d, sincos[0] = %f, sincos[1] = %f\n",      ncam, sincos[1][0], sincos[1][1]);
		printf(  "    Roll*Zoom matrix for camera %d, sincos[0] = %f, sincos[1] = %f\n",      ncam, sincos[2][0], sincos[2][1]);
		printf(  "    Roll matrix for      camera %d, sincos[0] = %f, sincos[1] = %f\n",      ncam, sincos[3][0], sincos[3][1]);
	}
	__syncthreads();// __syncwarp();
#endif // DEBUG20



// Create 3 3x3 matrices for az, tilt, roll/zoom:
	int axis = offset_matrices+threadIdx.z; // 0..2
	int const_index = threadIdx.z; // 0..2
	matrices[axis][threadIdx.y][threadIdx.x] =
			ROTS_TEMPLATE[const_index][threadIdx.y][threadIdx.x][0] * sincos[threadIdx.z][0]+ // cos
			ROTS_TEMPLATE[const_index][threadIdx.y][threadIdx.x][1] * sincos[threadIdx.z][1]+ // sin
			ROTS_TEMPLATE[const_index][threadIdx.y][threadIdx.x][2];                   // const
	axis += 3; // skip index == 3
	const_index +=3;
	matrices[axis][threadIdx.y][threadIdx.x] =
			ROTS_TEMPLATE[const_index][threadIdx.y][threadIdx.x][0] * sincos[threadIdx.z][0]+ // cos
			ROTS_TEMPLATE[const_index][threadIdx.y][threadIdx.x][1] * sincos[threadIdx.z][1]+ // sin
			ROTS_TEMPLATE[const_index][threadIdx.y][threadIdx.x][2];                          // const
	if (threadIdx.z == 0){
		axis += 3;
		const_index +=3;
		matrices[axis][threadIdx.y][threadIdx.x] =
				ROTS_TEMPLATE[const_index][threadIdx.y][threadIdx.x][0] * sincos[3][0]+ // cos
				ROTS_TEMPLATE[const_index][threadIdx.y][threadIdx.x][1] * sincos[3][1]+ // sin
				ROTS_TEMPLATE[const_index][threadIdx.y][threadIdx.x][2];                // const

	}
	__syncthreads();

#ifdef DEBUG20
	const char* matrices_names[] = {"az","tilt","roll*zoom","d_daz","d_tilt","d_roll","d_zoom"};

	if ((ncam == DBG_CAM) && (threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0)){
		for (int i = 0; i < 7; i++) {
			printf("\n----Matrix %s for camera %d:\n", matrices_names[i], ncam);
			for (int row = 0; row < 3; row++){
				for (int col = 0; col < 3; col++){
					printf("%9.6f, ",matrices[offset_matrices + i][row][col]);
				}
				printf("\n");
			}

		}
	}
	__syncthreads();// __syncwarp();
#endif // DEBUG20

/*
	__constant__ int mm_seq [3][3][3]={
			{
					{6,5,12}, // a_t * a_z -> tmp0
					{7,6,13}, // a_r * a_t -> tmp1
					{7,9,14}, // a_r * a_dt -> tmp2
			}, {
					{7,12,0}, // a_r * tmp0 -> rot
					{13,8,1}, // tmp1 * a_daz -> deriv0
					{14,5,2}, // tmp2 * a_az  -> deriv1
			}, {
					{10,12,3}, // a_dr * tmp0 -> deriv2
					{11,12,4}, // a_dzoom * tnmp0 -> deriv3
			}};
*/
	for (int i = 0; i < 3; i++){
		int srcl = mm_seq[i][threadIdx.z][0];
		int srcr = mm_seq[i][threadIdx.z][1];
		int dst =  mm_seq[i][threadIdx.z][2];
		if (srcl >= 0){
			matrices[dst][threadIdx.y][threadIdx.x] =
					matrices[srcl][threadIdx.y][0] * matrices[srcr][0][threadIdx.x]+
					matrices[srcl][threadIdx.y][1] * matrices[srcr][1][threadIdx.x]+
					matrices[srcl][threadIdx.y][2] * matrices[srcr][2][threadIdx.x];
		}
		__syncthreads();
	}
// copy results to global memory
	int gindx = threadIdx.z;
	int lindx = offset_rots + threadIdx.z;
	gpu_rot_deriv->matrices[gindx][ncam][threadIdx.y][threadIdx.x] = matrices[lindx][threadIdx.y][threadIdx.x];
	gindx +=3;
	lindx+=3;
	if (lindx < 5) {
		gpu_rot_deriv->matrices[gindx][ncam][threadIdx.y][threadIdx.x] = matrices[lindx][threadIdx.y][threadIdx.x];
	}
	__syncthreads();

// All done - read/verify all arrays

}


/*
 * blockDim.x = NUM_CAMS
 * blockDim.y = TILES_PER_BLOCK_GEOM
 */

extern "C" __global__ void get_tiles_offsets(
		struct tp_task     * gpu_tasks,
		int                  num_tiles,          // number of tiles in task
		struct gc          * gpu_geometry_correction,
		struct corr_vector * gpu_correction_vector,
		float *              gpu_rByRDist)      // length should match RBYRDIST_LEN
{
//	int task_num = blockIdx.x * blockDim.x + threadIdx.x; //  blockIdx.x * TILES_PER_BLOCK_GEOM + threadIdx.x
	int task_num = blockIdx.x * blockDim.y + threadIdx.y; //  blockIdx.x * TILES_PER_BLOCK_GEOM + threadIdx.y
	int thread_xy = blockDim.x * threadIdx.y + threadIdx.x;
	int ncam = threadIdx.x;
	// threadIdx.x - numcam, used for per-camera
	__shared__ struct gc geometry_correction;
	__shared__ float rByRDist [RBYRDIST_LEN];
	__shared__ struct corr_vector extrinsic_corr;
	__shared__ float rots[NUM_CAMS][3][3];
	__shared__ float pXY[NUM_CAMS][2]; // result to be copied to task
	// copy data common to all threads
	{
		float * gcp_local =  (float *) &geometry_correction;
		float * gcp_global = (float *) gpu_geometry_correction;
		int offset = thread_xy;
		for (int i = 0; i < CYCLES_COPY_GC; i++){
			if (offset < sizeof(struct gc)/sizeof(float)) {
				*(gcp_local + offset) = *(gcp_global + offset);
			}
			offset += THREADS_PER_BLOCK_GEOM;
		}
	}
	{
		float * cvp_local =  (float *) &extrinsic_corr;
		float * cvp_global = (float *) gpu_correction_vector;
		int offset = thread_xy;
		for (int i = 0; i < CYCLES_COPY_CV; i++){
			if (offset < sizeof(struct corr_vector)/sizeof(float)) {
				*(cvp_local + offset) = *(cvp_global + offset);
			}
			offset += THREADS_PER_BLOCK_GEOM;
		}
	}
	// TODO: maybe it is better to use system memory and not read all table?
	{
		float * rByRDistp_local =  (float *) rByRDist;
		float * rByRDistp_global = (float *) gpu_rByRDist;
		int offset = thread_xy;
		for (int i = 0; i < CYCLES_COPY_RBRD; i++){
			if (offset < RBYRDIST_LEN) {
				*(rByRDistp_local + offset) = *(rByRDistp_global + offset);
			}
			offset += THREADS_PER_BLOCK_GEOM;
		}
	}
	// copy rotational matrices
	//	__shared__ float rots[NUM_CAMS][3][3];
	//__device__ float rot_matrices [NUM_CAMS][3][3];

	{
		float * rots_local =  (float *) rots;
		float * rots_global = (float *) rot_matrices;
		int offset = thread_xy;
		for (int i = 0; i < CYCLES_COPY_ROTS; i++){
			if (offset < sizeof(struct corr_vector)/sizeof(float)) {
				*(rots_local + offset) = *(rots_global + offset);
			}
			offset += THREADS_PER_BLOCK_GEOM;
		}
	}
	__syncthreads();

#ifdef DEBUG20
	if ((threadIdx.x == 0)  && ( blockIdx.x == 0)){
		printf("\nget_tiles_offsets() threadIdx.x = %d, blockIdx.x= %d\n", (int)threadIdx.x, (int) blockIdx.x);
		printGeometryCorrection(&geometry_correction);
		printExtrinsicCorrection(&extrinsic_corr);
	}
	__syncthreads();// __syncwarp();
#endif // DEBUG20
	//		String dbg_s = corr_vector.toString();
	/* Starting with required tile center X, Y and nominal distortion, for each sensor port:
	 * 1) unapply common distortion (maybe for different - master camera)
	 * 2) apply disparity
	 * 3) apply rotations and zoom
	 * 4) re-apply distortion
	 * 5) return port center X and Y
	 * line_time
	 */

	// common code, calculated in parallel
	int cxy = gpu_tasks[task_num].txy;
	float disparity = gpu_tasks[task_num].target_disparity;
	int tileX = (cxy & 0xffff);
	int tileY = (cxy >> 16);
	float px = tileX * DTT_SIZE + DTT_SIZE/2; //  - shiftX;
	float py = tileY * DTT_SIZE + DTT_SIZE/2; //  - shiftY;

	float pXcd = px - 0.5 * geometry_correction.pixelCorrectionWidth;
	float pYcd = py - 0.5 * geometry_correction.pixelCorrectionHeight;

	float rXY [NUM_CAMS][2];

//	for (int i = 0; i < NUM_CAMS;i++){
	rXY[ncam][0] = geometry_correction.rXY[ncam][0];
	rXY[ncam][1] = geometry_correction.rXY[ncam][1];
//	}

	float rD = sqrtf(pXcd*pXcd + pYcd*pYcd)*0.001*geometry_correction.pixelSize; // distorted radius in a virtual center camera
	float rND2R=getRByRDist(rD/geometry_correction.distortionRadius, rByRDist);
	float pXc = pXcd * rND2R; // non-distorted coordinates relative to the (0.5 * this.pixelCorrectionWidth, 0.5 * this.pixelCorrectionHeight)
	float pYc = pYcd * rND2R; // in pixels
	float xyz [3]; // getWorldCoordinates
	xyz[2] = -SCENE_UNITS_SCALE * geometry_correction.focalLength * geometry_correction.disparityRadius / (disparity * 0.001*geometry_correction.pixelSize); // "+" - near, "-" far
	xyz[0] =  SCENE_UNITS_SCALE * pXc * geometry_correction.disparityRadius / disparity;
	xyz[1] = -SCENE_UNITS_SCALE * pYc * geometry_correction.disparityRadius / disparity;
	// next radial distortion coefficients are for this, not master camera (may be the same)
//	geometry_correction.rad_coeff[i];
	float fl_pix = geometry_correction.focalLength/(0.001 * geometry_correction.pixelSize); // focal length in pixels - this camera
	float ri_scale = 0.001 * geometry_correction.pixelSize / geometry_correction.distortionRadius;

//	for (int ncam = 0; ncam < NUM_CAMS; ncam++){

			// non-distorted XY of the shifted location of the individual sensor

	// -------------- Each camera calculated by its own thread ----------------
	float pXci0 = pXc - disparity *  rXY[ncam][0]; // in pixels
	float pYci0 = pYc - disparity *  rXY[ncam][1];
	// rectilinear, end of dealing with possibly other (master) camera, below all is for this camera distortions
	// Convert a 2-d non-distorted vector to 3d at fl_pix distance in z direction
	///		double [][] avi = {{pXci0}, {pYci0},{fl_pix}};
	///		Matrix vi = new Matrix(avi); // non-distorted sensor channel view vector in pixels (z -along the common axis)
	// Apply port-individual combined rotation/zoom matrix
	///		Matrix rvi = rots[i].times(vi);
	float rvi[3];
#pragma unroll
	for (int j = 0; j< 3; j++){
		rvi[j] = rots[ncam][j][0] * pXci0 + rots[ncam][j][1] * pYci0 + rots[ncam][j][2] * fl_pix;
	}
	// get back to the projection plane by normalizing vector
	float norm_z = fl_pix/rvi[2];
	float pXci =  rvi[0] * norm_z;
	float pYci =  rvi[1] * norm_z;
	// Re-apply distortion
	float rNDi =  sqrtf(pXci*pXci + pYci*pYci); // in pixels
	float ri =    rNDi* ri_scale; // relative to distortion radius

	float rD2rND = 1.0;
	{
		float rri = 1.0;
#pragma unroll
		for (int j = 0; j < sizeof(geometry_correction.rad_coeff)/sizeof(float); j++){
			rri *= ri;
			rD2rND += geometry_correction.rad_coeff[j]*(rri - 1.0);
		}
	}
	// Get port pixel coordinates by scaling the 2d vector with Rdistorted/Dnondistorted coefficient)
	float pXid = pXci * rD2rND;
	float pYid = pYci * rD2rND;
	pXY[ncam][0] =  pXid + geometry_correction.pXY0[ncam][0];
	pXY[ncam][1] =  pYid + geometry_correction.pXY0[ncam][1];





//	}//for (int i = 0; i < NUM_CAMS; i++){
/*


			// used when calculating derivatives, TODO: combine calculations !
			double drD2rND_dri = 0.0;
			Matrix drvi_daz = null;
			Matrix drvi_dtl = null;
			Matrix drvi_drl = null;
			double dpXci_dazimuth = 0.0;
			double dpYci_dazimuth = 0.0;
			double dpXci_dtilt =    0.0;
			double dpYci_dtilt =    0.0;
			double dpXci_droll =    0.0;
			double dpYci_droll =    0.0;

			if ((disp_dist != null) || (pXYderiv != null)) {
				rri = 1.0;
				for (int j = 0; j < rad_coeff.length; j++){
					drD2rND_dri += rad_coeff[j] * (j+1) * rri;
					rri *= ri;
				}
				if (deriv_rots != null) {
					// needed for derivatives and IMU
					drvi_daz = deriv_rots[i][0].times(vi);
					drvi_dtl = deriv_rots[i][1].times(vi);
					drvi_drl = deriv_rots[i][2].times(vi);
					dpXci_dazimuth = drvi_daz.get(0, 0) * norm_z - pXci * drvi_daz.get(2, 0) / rvi.get(2, 0);
					dpYci_dazimuth = drvi_daz.get(1, 0) * norm_z - pYci * drvi_daz.get(2, 0) / rvi.get(2, 0);
					dpXci_dtilt =    drvi_dtl.get(0, 0) * norm_z - pXci * drvi_dtl.get(2, 0) / rvi.get(2, 0);
					dpYci_dtilt =    drvi_dtl.get(1, 0) * norm_z - pYci * drvi_dtl.get(2, 0) / rvi.get(2, 0);
					dpXci_droll =    drvi_drl.get(0, 0) * norm_z - pXci * drvi_drl.get(2, 0) / rvi.get(2, 0);
					dpYci_droll =    drvi_drl.get(1, 0) * norm_z - pYci * drvi_drl.get(2, 0) / rvi.get(2, 0);
				}
			}
			double delta_t = 0.0;
			double [] imu =  null;
			double [][] dpXci_pYci_imu_lin = new double[2][3]; // null
			if (disp_dist != null) {
				disp_dist[i] =   new double [4]; // dx/d_disp, dx_d_ccw_disp
				// Not clear - what should be in Z direction before rotation here?
				double [][] add0 = {
						{-rXY[i][0],  rXY[i][1], 0.0},
						{-rXY[i][1], -rXY[i][0], 0.0},
						{ 0.0,                     0.0,                    0.0}}; // what is last element???
				Matrix dd0 = new Matrix(add0);
				Matrix dd1 = rots[i].times(dd0).getMatrix(0, 1,0,1).times(norm_z); // get top left 2x2 sub-matrix
////				Matrix dd1 = dd0.getMatrix(0, 1,0,1); // get top left 2x2 sub-matrix
				// now first column of 2x2 dd1 - x, y components of derivatives by disparity, second column - derivatives by ortho to disparity (~Y in 2d correlation)
				// unity vector in the direction of radius
				double c_dist = pXci/rNDi;
				double s_dist = pYci/rNDi;

				double [][] arot2= {
						{c_dist, s_dist},
						{-s_dist, c_dist}};
				Matrix rot2 = new Matrix(arot2); // convert from non-distorted X,Y to parallel and perpendicular (CCW) to the radius

				double [][] ascale_distort = {
						{rD2rND + ri* drD2rND_dri, 0     },
						{0,                       rD2rND}};
				Matrix scale_distort = new Matrix(ascale_distort); // scale component parallel to radius as distortion derivative, perpendicular - as distortion

				Matrix dd2 = rot2.transpose().times(scale_distort).times(rot2).times(dd1);

				disp_dist[i][0] =   dd2.get(0, 0);
				disp_dist[i][1] =   dd2.get(0, 1);
				disp_dist[i][2] =   dd2.get(1, 0); // d_py/d_disp
				disp_dist[i][3] =   dd2.get(1, 1);

				imu =  extrinsic_corr.getIMU(i); // currently it is common for all channels

				// ERS linear does not yet use per-port rotations, probably not needed

//				double [][] dpXci_pYci_imu_lin = new double[2][3]; // null
				if ((imu[0] != 0.0) || (imu[1] != 0.0) ||(imu[2] != 0.0) ||(imu[3] != 0.0) ||(imu[4] != 0.0) ||(imu[5] != 0.0)) {
					delta_t = dd2.get(1, 0) * disparity * line_time; // positive for top cameras, negative - for bottom
					double ers_Xci = delta_t* (dpXci_dtilt * imu[0] + dpXci_dazimuth * imu[1]  + dpXci_droll * imu[2]);
					double ers_Yci = delta_t* (dpYci_dtilt * imu[0] + dpYci_dazimuth * imu[1]  + dpYci_droll * imu[2]);
					if (xyz != null) {
						double k = SCENE_UNITS_SCALE * this.disparityRadius;
						double wdisparity = disparity;
						double dwdisp_dz = (k * this.focalLength / (0.001*this.pixelSize)) / (xyz[2] * xyz[2]);
						dpXci_pYci_imu_lin[0][0] = -wdisparity / k; // dpx/ dworld_X
						dpXci_pYci_imu_lin[1][1] =  wdisparity / k; // dpy/ dworld_Y
						dpXci_pYci_imu_lin[0][2] =  (xyz[0] / k) * dwdisp_dz; // dpx/ dworld_Z
						dpXci_pYci_imu_lin[1][2] =  (xyz[1] / k) * dwdisp_dz; // dpy/ dworld_Z
						ers_Xci += delta_t* (dpXci_pYci_imu_lin[0][0] * imu[3] + dpXci_pYci_imu_lin[0][2] * imu[5]);
						ers_Yci += delta_t* (dpXci_pYci_imu_lin[1][1] * imu[4] + dpXci_pYci_imu_lin[1][2] * imu[5]);
					}
					pXY[i][0] +=  ers_Xci * rD2rND; // added correction to pixel X
					pXY[i][1] +=  ers_Yci * rD2rND; // added correction to pixel Y


				} else {
					imu = null;
				}



// TODO: calculate derivatives of pX, pY by 3 imu omegas
			}


 */

}

/**
 * Calculate non-distorted radius from distorted using table approximation
 * @param rDist distorted radius
 * @return corresponding non-distorted radius
 */
inline __device__ float getRByRDist(float rDist,
		float rByRDist [RBYRDIST_LEN]) //shared memory
{
	if (rDist < 0) {
		return 0.0f; // normally should not happen
	}
	float findex = rDist/RBYRDIST_STEP;
	int index= (int) floorf(findex);
	if (index < 0){
		index = 0;
	}
	if (index > (RBYRDIST_LEN - 3)) {
		index = RBYRDIST_LEN - 3;
	}
	float mu = fmaxf(findex - index, 0.0f);
	float mu2 = mu * mu;
	float y0 = (index > 0)? rByRDist[index-1] : ( 2 * rByRDist[index] - rByRDist[index+1]);
	// use Catmull-Rom
	float a0 = -0.5 * y0 + 1.5 * rByRDist[index] - 1.5 * rByRDist[index+1] + 0.5 * rByRDist[index+2];
	float a1 =        y0 - 2.5 * rByRDist[index] + 2   * rByRDist[index+1] - 0.5 * rByRDist[index+2];
	float a2 = -0.5 * y0                              + 0.5 * rByRDist[index+1];
	float a3 =  rByRDist[index];
	float result= a0*mu*mu2+a1*mu2+a2*mu+a3;
	return result;
}

__device__ void printGeometryCorrection(struct gc * g){
#ifndef JCUDA
	printf("\nGeometry Correction\n------------------\n");
	printf("%22s: %f\n","pixelCorrectionWidth",  g->pixelCorrectionWidth);
	printf("%22s: %f\n","pixelCorrectionHeight", g->pixelCorrectionHeight);
	printf("%22s: %f\n","line_time",             g->line_time);

	printf("%22s: %f\n","focalLength", g->focalLength);
	printf("%22s: %f\n","pixelSize",   g->pixelSize);
	printf("%22s: %f\n","distortionRadius",g->distortionRadius);

	printf("%22s: %f\n","distortionC", g->distortionC);
	printf("%22s: %f\n","distortionB", g->distortionB);
	printf("%22s: %f\n","distortionA", g->distortionA);
	printf("%22s: %f\n","distortionA5",g->distortionA5);
	printf("%22s: %f\n","distortionA6",g->distortionA6);
	printf("%22s: %f\n","distortionA7",g->distortionA7);
	printf("%22s: %f\n","distortionA8",g->distortionA8);

	printf("%22s: %f\n","elevation",   g->elevation);
	printf("%22s: %f\n","heading",     g->heading);

	printf("%22s: %f, %f, %f, %f \n","forward", g->forward[0], g->forward[1], g->forward[2], g->forward[3]);
	printf("%22s: %f, %f, %f, %f \n","right",   g->right[0],   g->right[1],   g->right[2],   g->right[3]);
	printf("%22s: %f, %f, %f, %f \n","height",  g->height[0],  g->height[1],  g->height[2],  g->height[3]);
	printf("%22s: %f, %f, %f, %f \n","roll",    g->roll[0],    g->roll[1],    g->roll[2],    g->roll[3]);
	printf("%22s: %f, %f \n",        "pXY0[0]", g->pXY0[0][0], g->pXY0[0][1]);
	printf("%22s: %f, %f \n",        "pXY0[1]", g->pXY0[1][0], g->pXY0[1][1]);
	printf("%22s: %f, %f \n",        "pXY0[2]", g->pXY0[2][0], g->pXY0[2][1]);
	printf("%22s: %f, %f \n",        "pXY0[3]", g->pXY0[3][0], g->pXY0[3][1]);

	printf("%22s: %f\n","common_right",   g->common_right);
	printf("%22s: %f\n","common_forward", g->common_forward);
	printf("%22s: %f\n","common_height",  g->common_height);
	printf("%22s: %f\n","common_roll",    g->common_roll);

	printf("%22s: x=%f, y=%f\n","rXY[0]", g->rXY[0][0], g->rXY[0][1]);
	printf("%22s: x=%f, y=%f\n","rXY[1]", g->rXY[1][0], g->rXY[1][1]);
	printf("%22s: x=%f, y=%f\n","rXY[2]", g->rXY[2][0], g->rXY[2][1]);
	printf("%22s: x=%f, y=%f\n","rXY[3]", g->rXY[3][0], g->rXY[3][1]);

	printf("%22s: %f\n","cameraRadius",    g->cameraRadius);
	printf("%22s: %f\n","disparityRadius", g->disparityRadius);
#endif //ifndef JCUDA
}

__device__ void printExtrinsicCorrection(corr_vector * cv)
{
#ifndef JCUDA
	printf("\nExtrinsic Correction Vector\n---------------------------\n");
	printf("%22s: %f, %f, %f\n",     "tilt",    cv->tilt[0],    cv->tilt[1],    cv->tilt[2]);
	printf("%22s: %f, %f, %f\n",     "azimuth", cv->azimuth[0], cv->azimuth[1], cv->azimuth[2]);
	printf("%22s: %f, %f, %f, %f\n", "roll",    cv->roll[0],    cv->roll[1],    cv->roll[2],      cv->roll[3]);
	printf("%22s: %f, %f, %f\n",     "zoom",    cv->zoom[0],    cv->zoom[1],    cv->zoom[2]);

	printf("%22s: %f(t), %f(a), %f(r)\n",     "imu_rot",    cv->imu_rot[0],    cv->imu_rot[1],    cv->imu_rot[2]);
	printf("%22s: %f(x), %f(y), %f(z)\n",     "imu_move",    cv->imu_move[0],    cv->imu_move[1],    cv->imu_move[2]);
#endif //ifndef JCUDA
}



