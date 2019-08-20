

#include <stdio.h>
#include "gmmintrin.h"

int main(void) {
	__m256 vf0, vf1, vf2, vf3, vf4, vf5, vf6, vf7, vf8, vf9, vf10;
	float fval = 1.2;
	__declspec(align(32)) float farray[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};

	__m256d vd0, vd1, vd2, vd3, vd4, vd5, vd6, vd7, vd8, vd9, vd10, vd11;
	double dval = 1.3;
	__declspec(align(32)) double darray[4] = {1.0, 2.0, 3.0, 4.0};
	__declspec(align(32)) double darray2[4] = {-1.0, -1.1, -1.2, -1.3};
	__declspec(align(32)) __int64 idarray[4] = {0x0A, 0x04, 0x06, 0x00};
	__declspec(align(32)) __int64 idarray2[4] = {0x02, 0x02, 0x00, 0x02};
	__declspec(align(32)) __int32 ifarray[8] = {0x02, 0x03, 0x01, 0x00, 0x03, 0x02, 0x03, 0x01};
	__declspec(align(32)) __int32 ifarray2[8] = {0x0E, 0x09, 0x03, 0x04, 0x0F, 0x07, 0x05, 0x0C};
	__m256i vid;
	__m128d xmmvd = _mm_set_pd(1.0,2.0);

	printf("\n8-way float\n");
	vf0 = _mm256_broadcast_ss(&fval);
	PRINT_M256(vf0);
	vf1 = _mm256_load_ps(&farray[0]);
	PRINT_M256(vf1);
	vf2 = _mm256_add_ps(vf0, vf1);
	PRINT_M256(vf2);
	vf3 = _mm256_xor_ps(vf0, vf0);
	PRINT_M256(vf3);
	vf4 = _mm256_permute_ps(vf1, 0x89);
	PRINT_M256(vf4);
	vid = _mm256_load_si256((__m256i *)(&ifarray[0]));
	vf5 = _mm256_permutevar_ps(vf1, vid);
	PRINT_M256(vf5);
	vf6 = _mm256_permute2f128_ps(vf1, vf2, 0x38);
	PRINT_M256(vf6);
	vf7 = _mm256_shuffle_ps(vf1, vf2, 0x6A);
	PRINT_M256(vf7);
	vid = _mm256_load_si256((__m256i *)(&ifarray2[0]));
	vf8 = _mm256_permute2_ps(vf1, vf2, vid, 0x03);
	PRINT_M256(vf8);

	printf("\n4-way double\n");
	vd0 = _mm256_broadcast_sd(&dval);
	PRINT_M256D(vd0);
	vd1 = _mm256_load_pd(&darray[0]);
	PRINT_M256D(vd1);
	vd2 = _mm256_add_pd(vd0, vd1);
	PRINT_M256D(vd2);
	vd3 = _mm256_xor_pd(vd0, vd0);
	PRINT_M256D(vd3);
	vd4 = _mm256_permute_pd(vd1, 0x0F);
	PRINT_M256D(vd4);
	vd5 = _mm256_shuffle_pd(vd1, vd2, 0x0F);
	PRINT_M256D(vd5);
	vd6 = _mm256_permute2f128_pd(vd1, vd2, 0x38);
	PRINT_M256D(vd6);
	vid = _mm256_load_si256((__m256i *)(&idarray[0]));
	vd7 = _mm256_permute2_pd(vd1, vd2, vid, 0x02);
	PRINT_M256D(vd7);
	vid = _mm256_load_si256((__m256i *)(&idarray2[0]));
	vd8 = _mm256_permutevar_pd(vd1, vid);
	PRINT_M256D(vd8);
	vd9 = _mm256_insertf128_pd(vd8, xmmvd, 0);
	PRINT_M256D(vd9);
	vd10 = _mm256_insertf128_pd(vd8, xmmvd, 1);
	PRINT_M256D(vd10);
	vd11 = _mm256_maskload_pd(darray, _mm256_set_epi32(0x00000000, 0x00000000, 0x80000000, 0x00000000, 0x80000000, 0x00000000, 0x80000000, 0x00000000));
	PRINT_M256D(vd11);
	_mm256_maskstore_pd(darray2, _mm256_set_epi32(0x00000000, 0x00000000, 0x80000000, 0x00000000, 0x80000000, 0x00000000, 0x80000000, 0x00000000), _mm256_set_pd(4.0, 3.0, 2.0, 1.0));
	PRINT_M256D(_mm256_load_pd(darray2));
	return 0;
}