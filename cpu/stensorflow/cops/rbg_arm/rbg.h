/**************************************************************************
Copyright © 2021, Ant Financial and/or its affiliates. All rights reserved.

Without express written authorization from Ant Financial, no one may conduct any of the following actions:
1) reproduce, spread, present, set up a mirror of, upload, download this source code;
2) modify, translate and adapt this source code, or develop derivative products, works, and services based on this source code; or
3) distribute, lease, rent, sub-license, demise or transfer any rights in relation to this source code, or authorize the reproduction of this source code on other's computers.
**************************************************************************/

#ifndef RBG_H
#define RBG_H

#include "seed.h"
#include "ctrdrbg.h"

// return value
#define SUCCESS_RBG 0
#define ERROR_RBG_CTX 1
#define ERROR_RBG_UNIFBUF 2
#define ERROR_RBG_RANDBUF 3
#define ERROR_RBG_RANDNUM 4
#define ERROR_RBG_UNIFORMRANGE 5
#define ERROR_RBG_UNIFORMMAXTRY 6
#define ERROR_RBG_LOCKFILE 7
#define ERROR_RBG_FLOCK 8

// fixed parameter
#define RBG_SEEDED_TEMPNUM 512
#define RBG_SEEDED_SEEDLEN 128
#define RBG_SEEDED_CONTEXTLEN 784
#define RBG_SEEDED_TEMPNUM 512
#define RBG_UNIFORM_MAXTRYTIMES 100000
#define RBG_UNIFORM_INT64MAX 0x7FFFFFFFFFFFFFFFLL          // 2^63-1  9223372036854775807
#define RBG_UNIFORM_INT64MIN (-0x7FFFFFFFFFFFFFFFLL - 1LL) //-2^63  -9223372036854775808
#define RBG_UNIFORM_UINT64MAX 0xFFFFFFFFFFFFFFFFULL
#define RBG_UNIFORM_UINT64MOVE 0x8000000000000000ULL

// generate long long seed array
int rbg_getseed(long long *seed, int seednum);
// generate long long seed array, with file lock
int rbg_getseed_lock(long long *seed, int seednum, char *filepath);

// nonseeded RBG, generate long long random number array
int rbg_nonseeded(long long *rand, int randnum);

// seeded RBG, generate long long random number array with seed
int rbg_seeded(long long *rand, long long *seed, int randnum);

// seeded RBG, instantiate RBG
int rbg_seeded_instantiate(unsigned char *ctx, long long *seed);
// seeded RBG, reseed RBG
int rbg_seeded_reseed(unsigned char *ctx, long long *seed);
// seeded RBG, generate long long random number array
int rbg_seeded_random(unsigned char *ctx, long long *rand, int randnum);
// seeded RBG, remove RBG
int rbg_seeded_remove(unsigned char *ctx);

// uniform sampler between min and max, with new random numbers from nonseeded RBG
int rbg_nonseeded_uniform(long long *unif, long long *rand, int randnum, long long min, long long max);
// uniform sampler between min and max, with new random numbers from seeded RBG
int rbg_seeded_uniform(unsigned char *ctx, long long *unif, long long *rand, int randnum, long long min, long long max);

#endif // RBG_H
