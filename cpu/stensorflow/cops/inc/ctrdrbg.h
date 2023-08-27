/**************************************************************************
Copyright © 2021, Ant Financial and/or its affiliates. All rights reserved.

Without express written authorization from Ant Financial, no one may conduct any of the following actions:
1) reproduce, spread, present, set up a mirror of, upload, download this source code;
2) modify, translate and adapt this source code, or develop derivative products, works, and services based on this source code; or
3) distribute, lease, rent, sub-license, demise or transfer any rights in relation to this source code, or authorize the reproduction of this source code on other's computers.
**************************************************************************/

#ifndef CTRDRBG_H
#define CTRDRBG_H

#include "aes.h"

// state
#define CTRDRBG_UNSEEDED 0		 // CTRDRBG is unseeded
#define CTRDRBG_SEEDED 1		 // CTRDRBG is seeded
#define CTRDRBG_RESEEDREQUIRED 2 // counter > CTRDRBG_SEED_INTERVAL, reseed is required

// fixed parameter
#define CTRDRBG_KEYLEN 16												// key len of a single CTRDRBG, equal to AES-128 key size
#define CTRDRBG_BLOCKLEN 16												// block len of a single CTRDRBG, equal toAES-128 block size
#define CTRDRBG_ROUNDKEYLEN 176											// round key len of a single CTRDRBG, equal to AES-128 round keys size
#define CTRDRBG_SEEDLEN (CTRDRBG_KEYLEN + CTRDRBG_BLOCKLEN)				// seed len of a single CTRDRBG
#define CTRDRBG_NUM 4													// number of single CTRDRBGs in a parallel CTRDRBG
#define CTRDRBG_NUM_KEYLEN (CTRDRBG_NUM * CTRDRBG_KEYLEN)				// key len of a parallel CTRDRBG
#define CTRDRBG_NUM_BLOCKLEN (CTRDRBG_NUM * CTRDRBG_BLOCKLEN)			// block len of a parallel CTRDRBG
#define CTRDRBG_NUM_ROUNDKEYLEN (CTRDRBG_NUM * CTRDRBG_ROUNDKEYLEN)		// round key len of a parallel CTRDRBG
#define CTRDRBG_NUM_SEEDLEN (CTRDRBG_NUM_KEYLEN + CTRDRBG_NUM_BLOCKLEN) // seed len of a parallel CTRDRBG

// secure parameter
#define CTRDRBG_REQUEST_MAXLEN 65536		 // 2^16, max number of bytes per request
#define CTRDRBG_SEED_INTERVAL (0x1ULL << 48) // 2^48, maximum number of requests between reseeds

// return value
#define SUCCESS_CTRDRBG 0
#define ERROR_CTRDRBG_CTX 1
#define ERROR_CTRDRBG_UNSEEDED 2
#define ERROR_CTRDRBG_RESEEDREQUIRED 3
#define ERROR_CTRDRBG_SEEDBUF 4
#define ERROR_CTRDRBG_SEEDLEN 5
#define ERROR_CTRDRBG_RANDBUF 6
#define ERROR_CTRDRBG_RANDLEN 7
#define ERROR_CTRDRBG_RANDLENBIGGER 8

// CTRDRBG internal state
typedef struct CTRDRBG_CONTEXT
{
	unsigned long long state;						 // state of CTRDRBG, uninitialized, initialized, reseed required
	unsigned long long counter;						 // the number of requests since instantiation or reseeding
	unsigned char V[CTRDRBG_NUM_BLOCKLEN];			 // value V
	unsigned char roundkey[CTRDRBG_NUM_ROUNDKEYLEN]; // AES-128 round keys

} CTRDRBG_CTX;

// update CTRDRBG internal state
int ctrdrbg_update(CTRDRBG_CTX *ctx, unsigned char *provided_data);

// instantiate CTRDRBG with seed
int ctrdrbg_instantiate(CTRDRBG_CTX *ctx, unsigned char *seed, int seedlen);

// reseed CTRDRBG with seed
int ctrdrbg_reseed(CTRDRBG_CTX *ctx, unsigned char *seed, int seedlen);

// generate random bytes in one request
int ctrdrbg_getrnd_req(CTRDRBG_CTX *ctx, unsigned char *rand, int randlen);

// generate random bytes
int ctrdrbg_getrnd(CTRDRBG_CTX *ctx, unsigned char *rand, int randlen);

// clear CTRDRBG internal state
int ctrdrbg_remove(CTRDRBG_CTX *ctx);

#endif // CTRDRBG_H
