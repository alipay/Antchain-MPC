/**************************************************************************
Copyright © 2021, Ant Financial and/or its affiliates. All rights reserved.

Without express written authorization from Ant Financial, no one may conduct any of the following actions:
1) reproduce, spread, present, set up a mirror of, upload, download this source code;
2) modify, translate and adapt this source code, or develop derivative products, works, and services based on this source code; or
3) distribute, lease, rent, sub-license, demise or transfer any rights in relation to this source code, or authorize the reproduction of this source code on other's computers.
**************************************************************************/

#include <string.h>
#include <stdio.h>
#include <sys/file.h>
#include "rbg.h"

//generate long long seed array
//seed, output, long long seed array
//seednum, input, number of long long seed
int rbg_getseed(long long* seed, int seednum) {
	int ret;
	int seedlen;

	seedlen = seednum * 8;

	ret = get_seed((unsigned char*)seed, seedlen);
	if (ret != SUCCESS_SEED)
		return ret;

	return SUCCESS_RBG;
}

//generate long long seed array, with file lock
//seed, output, long long seed array
//seednum, input, number of long long seed
//filepath, input, lock file path
int rbg_getseed_lock(long long *seed, int seednum, char *filepath){
	int ret;
	int fn;
	FILE *fp = NULL;

	//get file lock
	fp = fopen(filepath, "r");
	if(fp == NULL)
		return ERROR_RBG_LOCKFILE;
	fn = fileno(fp);
	ret = flock(fn, LOCK_EX);
	if(ret != 0)
		return ERROR_RBG_FLOCK;

	//get seed
	ret = rbg_getseed(seed, seednum);
	if (ret != SUCCESS_SEED)
		return ret;

	//release file lock
	ret = flock(fn, LOCK_UN);
	if(ret != 0)
		return ERROR_RBG_FLOCK;
	fclose(fp);

	return SUCCESS_RBG;
}

//nonseeded RBG, generate long long random number array
//rand, output, long long random number array
//randnum, input, number of long long random number 
int rbg_nonseeded(long long* rand, int randnum) {
	int ret;
	int randlen;

	randlen = randnum * 8;

	ret = get_rdrand((unsigned char*)rand, randlen);
	if (ret != SUCCESS_RDRAND)
		return ret;

	return SUCCESS_RBG;
}

//seeded RBG, generate long long random number array with seed
//rand, output, long long random number array
//seed, input, long long seed array
//randnum, input, number of long long random number 
int rbg_seeded(long long* rand, long long* seed, int randnum) {
	int ret;
	int randlen;
	int seedlen;

	//CTRDRBG internal state
	CTRDRBG_CTX ctx;

	//seedlen require 128B(16 long long) 
	seedlen = RBG_SEEDED_SEEDLEN;

	randlen = randnum * 8;

	//instantiate CTRDRBG with seed
	ret = ctrdrbg_instantiate(&ctx, (unsigned char*)seed, seedlen);
	if (ret != SUCCESS_CTRDRBG)
		return ret;

	//generate random bytes
	ret = ctrdrbg_getrnd(&ctx, (unsigned char*)rand, randlen);
	if (ret != SUCCESS_CTRDRBG)
		return ret;

	//clear CTRDRBG internal state
	ret = ctrdrbg_remove(&ctx);
	if (ret != SUCCESS_CTRDRBG)
		return ret;

	return SUCCESS_RBG;
}

//seeded RBG, instantiate RBG
//ctx, input & output, seeded RBG internal state
//seed, input, long long seed array
int rbg_seeded_instantiate(unsigned char* ctx, long long* seed) {
	int ret;
	int seedlen;

	//seedlen require 128B(16 long long) 
	seedlen = RBG_SEEDED_SEEDLEN;

	//instantiate CTRDRBG with seed
	ret = ctrdrbg_instantiate((CTRDRBG_CTX*)ctx, (unsigned char*)seed, seedlen);
	if (ret != SUCCESS_CTRDRBG)
		return ret;

	return SUCCESS_RBG;
}

//seeded RBG, reseed RBG
//ctx, input & output, seeded RBG internal state
//seed, input, long long seed array
int rbg_seeded_reseed(unsigned char* ctx, long long* seed) {
	int ret;
	int seedlen;

	//seedlen require 128B(16 long long) 
	seedlen = RBG_SEEDED_SEEDLEN;

	//reseed CTRDRBG with seed
	ret = ctrdrbg_reseed((CTRDRBG_CTX*)ctx, (unsigned char*)seed, seedlen);
	if (ret != SUCCESS_CTRDRBG)
		return ret;

	return SUCCESS_RBG;
}

//seeded RBG, generate long long random number array
//ctx, input & output, seeded RBG internal state
//rand, output, long long random number array
//randnum, input, number of long long random number 
int rbg_seeded_random(unsigned char* ctx, long long* rand, int randnum) {
	int ret;
	int randlen;

	randlen = randnum * 8;

	//generate random bytes
	ret = ctrdrbg_getrnd((CTRDRBG_CTX*)ctx, (unsigned char*)rand, randlen);
	if (ret != SUCCESS_CTRDRBG)
		return ret;

	return SUCCESS_RBG;
}

//seeded RBG, remove RBG
//ctx, input & output, seeded RBG internal state
int rbg_seeded_remove(unsigned char* ctx) {
	int ret;

	//clear CTRDRBG internal state
	ret = ctrdrbg_remove((CTRDRBG_CTX*)ctx);
	if (ret != SUCCESS_CTRDRBG)
		return ret;

	return SUCCESS_RBG;
}

//uniform sampler between min and max, with new random numbers from nonseeded RBG 
//unif, output, long long uniform random number array between min and max
//rand, input, long long random number array
//randnum, input, number of long long random number 
//min, input, minimum uniform random number
//max, input, maximum uniform random number
int rbg_nonseeded_uniform(long long *unif, long long *rand, int randnum, long long min, long long max) {
	int i, j;
	int flag;
	unsigned long long range;
	unsigned long long left;
	unsigned long long value;

	//check parameters
	if (unif == NULL)
		return ERROR_RBG_UNIFBUF;

	if (rand == NULL)
		return ERROR_RBG_RANDBUF;

	if (randnum < 1)
		return ERROR_RBG_RANDNUM;

	//no operation
	if ((max == RBG_UNIFORM_INT64MAX) && (min == RBG_UNIFORM_INT64MIN)){
		if(unif != rand)
			memcpy(unif, rand, randnum * sizeof(long long));
		return SUCCESS_RBG;
	}	

	//wrong parameters
	if (max < min)
		return ERROR_RBG_UNIFORMRANGE;

	//one value
	if (max == min) {
		for (i = 0; i < randnum; i++)
			unif[i] = max;
		return SUCCESS_RBG;
	}

	range = (unsigned long long)max - min + 1;
	//left = 2^64 % range
	left = ((RBG_UNIFORM_UINT64MAX % range) + 1) % range;

	//reduce all random numbers
	for (i = 0; i < randnum; i++) {
		flag = 1;
		//to unsigned long long
		value = (unsigned long long)rand[i];
		
		for (j = 0; j < RBG_UNIFORM_MAXTRYTIMES; j++) {
			//accept
			if (value <= (RBG_UNIFORM_UINT64MAX - left)) {
				flag = 0;
				break;
			}
			else //reject, regenerate random number from nonseeded RBG
				rbg_nonseeded((long long *)&value, 1);
		}
		//has tried RBG_UNIFORM_MAXTRYTIMES times
		if (flag == 1)
			return ERROR_RBG_UNIFORMMAXTRY;

		//map to [min, max]
		if (value > range)
			value = value % range;
		unif[i] = value + min;
	}

	return SUCCESS_RBG;
}

//uniform sampler between min and max, with new random numbers from seeded RBG
//ctx, input & output, seeded RBG internal state
//unif, output, long long uniform random number array between min and max
//rand, input, long long random number array
//randnum, input, number of long long random number 
//min, input, minimum uniform random number
//max, input, maximum uniform random number
int rbg_seeded_uniform(unsigned char* ctx, long long *unif, long long *rand, int randnum, long long min, long long max) {
	int i, j;
	int flag;
	int move;
	unsigned long long range;
	unsigned long long left;
	unsigned long long value;
	unsigned long long temp[RBG_SEEDED_TEMPNUM];

	//check parameters
	if (ctx == NULL)
		return ERROR_RBG_CTX;

	if (unif == NULL)
		return ERROR_RBG_UNIFBUF;

	if (rand == NULL)
		return ERROR_RBG_RANDBUF;

	if (randnum < 1)
		return ERROR_RBG_RANDNUM;

	//no operation
	if ((max == RBG_UNIFORM_INT64MAX) && (min == RBG_UNIFORM_INT64MIN)){
		if(unif != rand)
			memcpy(unif, rand, randnum * sizeof(long long));
		return SUCCESS_RBG;
	}

	//wrong parameters
	if (max < min)
		return ERROR_RBG_UNIFORMRANGE;

	//one value
	if (max == min) {
		for (i = 0; i < randnum; i++)
			unif[i] = max;
		return SUCCESS_RBG;
	}

	range = (unsigned long long)max - min + 1;
	//left = 2^64 % range
	left = ((RBG_UNIFORM_UINT64MAX % range) + 1) % range;

	//reduce all random numbers
	move = RBG_SEEDED_TEMPNUM;
	for (i = 0; i < randnum; i++) {
		flag = 1;
		//to unsigned long long
		value = (unsigned long long)rand[i];
		
		for (j = 0; j < RBG_UNIFORM_MAXTRYTIMES; j++) {
			//accept
			if (value <= (RBG_UNIFORM_UINT64MAX - left)) {
				flag = 0;
				break;
			}
			else{ //reject
				if(move == RBG_SEEDED_TEMPNUM){
					//regenerate random number from seeded RBG
					rbg_seeded_random(ctx, (long long *)temp, RBG_SEEDED_TEMPNUM);
					move = 0;
				}
				//get random number
				value = temp[move];
				move++;				
			}
		}
		//has tried RBG_UNIFORM_MAXTRYTIMES times
		if (flag == 1)
			return ERROR_RBG_UNIFORMMAXTRY;

		//map to [min, max]
		if (value > range)
			value = value % range;
		unif[i] = value + min;
	}

	return SUCCESS_RBG;
}
