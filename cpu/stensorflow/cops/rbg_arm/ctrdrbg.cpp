/**************************************************************************
Copyright © 2021, Ant Financial and/or its affiliates. All rights reserved.

Without express written authorization from Ant Financial, no one may conduct any of the following actions:
1) reproduce, spread, present, set up a mirror of, upload, download this source code;
2) modify, translate and adapt this source code, or develop derivative products, works, and services based on this source code; or
3) distribute, lease, rent, sub-license, demise or transfer any rights in relation to this source code, or authorize the reproduction of this source code on other's computers.
**************************************************************************/

#include <stdlib.h>
#include <string.h>
#include "ctrdrbg.h"

// refer to NIST SP 800-90A Rev.1 10.2 DRBG Mechanism Based on Block Ciphers
// https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-90Ar1.pdf

// update CTRDRBG internal state
// ctx, input & output, CTRDRBG internal state
// provided_data, input, data provided for update
int ctrdrbg_update(CTRDRBG_CTX *ctx, unsigned char *provided_data)
{
	int i, j;
	unsigned char cipher[CTRDRBG_NUM_SEEDLEN];

	// check parameters
	if (ctx == NULL)
		return ERROR_CTRDRBG_CTX;

	// prepare CTRDRBG_NUM_SEEDLEN cipher
	for (i = 0; i < 2; i++)
	{
		// V = (V+1) mod 2^CTRDRBG_BLOCKLEN
		for (j = 0; j < CTRDRBG_NUM; j++)
		{
			*(unsigned long long *)(ctx->V + j * CTRDRBG_BLOCKLEN) += 1;
			if (*(unsigned long long *)(ctx->V + j * CTRDRBG_BLOCKLEN) == 0)
				*(unsigned long long *)(ctx->V + 8 + j * CTRDRBG_BLOCKLEN) += 1;
		}

		// cipher = encrypt(key, V)
		if (CTRDRBG_NUM == 4) // 4 rk 4 blocks
			aes128_enc4(cipher + i * CTRDRBG_NUM_BLOCKLEN, ctx->V, ctx->roundkey);
		else
		{
			for (j = 0; j < CTRDRBG_NUM; j++)
				aes128_enc(cipher + j * CTRDRBG_BLOCKLEN + i * CTRDRBG_NUM_BLOCKLEN, ctx->V + j * CTRDRBG_BLOCKLEN, ctx->roundkey + j * CTRDRBG_ROUNDKEYLEN);
		}
	}

	// cipher = cipher ^ provided_data
	if (provided_data != NULL)
	{
		for (i = 0; i < 2 * CTRDRBG_NUM_BLOCKLEN; i++)
			cipher[i] = cipher[i] ^ provided_data[i];
	}

	// round key = key_schedule(leftmost(cipher, CTRDRBG_KEYLEN))
	for (i = 0; i < CTRDRBG_NUM; i++)
		aes128_set_enckey(ctx->roundkey + i * CTRDRBG_ROUNDKEYLEN, cipher + i * CTRDRBG_KEYLEN);

	// V = rightmost(cipher, CTRDRBG_BLOCKLEN)
	memcpy(ctx->V, cipher + CTRDRBG_NUM_KEYLEN, CTRDRBG_NUM_BLOCKLEN);

	// clear cipher
	memset(cipher, 0, CTRDRBG_NUM_SEEDLEN);

	return SUCCESS_CTRDRBG;
}

// instantiate CTRDRBG with seed
// ctx, input & output, CTRDRBG internal state
// seed, input, seed buffer
// seedlen, input, number of bytes
int ctrdrbg_instantiate(CTRDRBG_CTX *ctx, unsigned char *seed, int seedlen)
{
	int i;
	unsigned char key[CTRDRBG_NUM_KEYLEN];

	// check parameters
	if (ctx == NULL)
		return ERROR_CTRDRBG_CTX;

	if (seed == NULL)
		return ERROR_CTRDRBG_SEEDBUF;

	if (seedlen < CTRDRBG_NUM_SEEDLEN)
		return ERROR_CTRDRBG_SEEDLEN;

	// set key to 0
	memset(key, 0, CTRDRBG_NUM_KEYLEN);
	for (i = 0; i < CTRDRBG_NUM; i++)
		aes128_set_enckey(ctx->roundkey + i * CTRDRBG_ROUNDKEYLEN, key + i * CTRDRBG_KEYLEN);

	// set V to 0
	memset(ctx->V, 0, CTRDRBG_NUM_BLOCKLEN);

	// update ctx with seed
	ctrdrbg_update(ctx, seed);

	// set counter to 1
	ctx->counter = 1;

	// set state to CTRDRBG_SEEDED
	ctx->state = CTRDRBG_SEEDED;

	return SUCCESS_CTRDRBG;
}

// reseed CTRDRBG with seed
// ctx, input & output, CTRDRBG internal state
// seed, input, seed buffer
// seedlen, input, number of bytes
int ctrdrbg_reseed(CTRDRBG_CTX *ctx, unsigned char *seed, int seedlen)
{
	// check parameters
	if (ctx == NULL)
		return ERROR_CTRDRBG_CTX;

	if (seed == NULL)
		return ERROR_CTRDRBG_SEEDBUF;

	if (seedlen < CTRDRBG_NUM_SEEDLEN)
		return ERROR_CTRDRBG_SEEDLEN;

	// update ctx with seed
	ctrdrbg_update(ctx, seed);

	// set counter to 1
	ctx->counter = 1;

	// set state to CTRDRBG_SEEDED
	ctx->state = CTRDRBG_SEEDED;

	return SUCCESS_CTRDRBG;
}

// generate random bytes in one request
// ctx, input & output, CTRDRBG internal state
// rand, output, rand buffer
// randlen, input, number of bytes
int ctrdrbg_getrnd_req(CTRDRBG_CTX *ctx, unsigned char *rand, int randlen)
{
	int i;
	int count, left;
	unsigned char *provided_data;
	unsigned char cipher[CTRDRBG_NUM_BLOCKLEN];

	// check parameters
	if (ctx == NULL)
		return ERROR_CTRDRBG_CTX;

	if (rand == NULL)
		return ERROR_CTRDRBG_RANDBUF;

	if (randlen < 1)
		return ERROR_CTRDRBG_RANDLEN;

	// number of random bytes in one request must be no bigger than CTRDRBG_REQUEST_MAXLEN
	if (randlen > CTRDRBG_REQUEST_MAXLEN)
		return ERROR_CTRDRBG_RANDLENBIGGER;

	// check CTRDRBG state
	if (ctx->state == CTRDRBG_UNSEEDED)
		return ERROR_CTRDRBG_UNSEEDED;

	if (ctx->state == CTRDRBG_RESEEDREQUIRED)
		return ERROR_CTRDRBG_RESEEDREQUIRED;

	// generate random bytes
	count = 0;
	left = randlen;
	while (left > 0)
	{
		// V = (V+1) mod 2^CTRDRBG_BLOCKLEN
		for (i = 0; i < CTRDRBG_NUM; i++)
		{
			*(unsigned long long *)(ctx->V + i * CTRDRBG_BLOCKLEN) += 1;
			if (*(unsigned long long *)(ctx->V + i * CTRDRBG_BLOCKLEN) == 0)
				*(unsigned long long *)(ctx->V + 8 + i * CTRDRBG_BLOCKLEN) += 1;
		}

		// generate CTRDRBG_NUM_BLOCKLEN random bytes
		if (left > CTRDRBG_NUM_BLOCKLEN)
		{
			// cipher = encrypt(key, V)
			if (CTRDRBG_NUM == 4) // 4 rk 4 blocks
				aes128_enc4(rand + count * CTRDRBG_NUM_BLOCKLEN, ctx->V, ctx->roundkey);
			else
			{
				for (i = 0; i < CTRDRBG_NUM; i++)
					aes128_enc(rand + i * CTRDRBG_BLOCKLEN + count * CTRDRBG_NUM_BLOCKLEN, ctx->V + i * CTRDRBG_BLOCKLEN, ctx->roundkey + i * CTRDRBG_ROUNDKEYLEN);
			}
			left = left - CTRDRBG_NUM_BLOCKLEN;
			count++;
		}
		else
		{ // generate left random bytes
			// cipher = encrypt(key, V)
			if (CTRDRBG_NUM == 4) // 4 rk 4 blocks
				aes128_enc4(cipher, ctx->V, ctx->roundkey);
			else
			{
				for (i = 0; i < CTRDRBG_NUM; i++)
					aes128_enc(cipher + i * CTRDRBG_BLOCKLEN, ctx->V + i * CTRDRBG_BLOCKLEN, ctx->roundkey + i * CTRDRBG_ROUNDKEYLEN);
			}
			memcpy(rand + count * CTRDRBG_NUM_BLOCKLEN, cipher, left);
			left = 0;
		}
	}

	// update ctx with NULL
	provided_data = NULL;
	ctrdrbg_update(ctx, provided_data);

	// counter = counter + 1
	ctx->counter = ctx->counter + 1;

	// check counter
	if (ctx->counter > CTRDRBG_SEED_INTERVAL)
		ctx->state = CTRDRBG_RESEEDREQUIRED;

	return SUCCESS_CTRDRBG;
}

// generate random bytes
// ctx, input & output, CTRDRBG internal state
// rand, output, rand buffer
// randlen, input, number of bytes
int ctrdrbg_getrnd(CTRDRBG_CTX *ctx, unsigned char *rand, int randlen)
{
	int i;
	int ret;
	int count, left;

	// check parameters
	if (ctx == NULL)
		return ERROR_CTRDRBG_CTX;

	if (rand == NULL)
		return ERROR_CTRDRBG_RANDBUF;

	if (randlen < 1)
		return ERROR_CTRDRBG_RANDLEN;

	// check CTRDRBG state
	if (ctx->state == CTRDRBG_UNSEEDED)
		return ERROR_CTRDRBG_UNSEEDED;

	if (ctx->state == CTRDRBG_RESEEDREQUIRED)
		return ERROR_CTRDRBG_RESEEDREQUIRED;

	count = randlen / CTRDRBG_REQUEST_MAXLEN;
	left = randlen % CTRDRBG_REQUEST_MAXLEN;

	// get count * CTRDRBG_REQUEST_MAXLEN random bytes
	for (i = 0; i < count; i++)
	{
		ret = ctrdrbg_getrnd_req(ctx, rand + i * CTRDRBG_REQUEST_MAXLEN, CTRDRBG_REQUEST_MAXLEN);
		if (ret != SUCCESS_CTRDRBG)
			return ret;
	}

	// get left random bytes
	if (left)
	{
		ret = ctrdrbg_getrnd_req(ctx, rand + count * CTRDRBG_REQUEST_MAXLEN, left);
		if (ret != SUCCESS_CTRDRBG)
			return ret;
	}

	return SUCCESS_CTRDRBG;
}

// clear CTRDRBG internal state
// ctx, input & output, CTRDRBG internal state
int ctrdrbg_remove(CTRDRBG_CTX *ctx)
{
	// check parameters
	if (ctx == NULL)
		return ERROR_CTRDRBG_CTX;

	memset(ctx, 0, sizeof(CTRDRBG_CTX));

	return SUCCESS_CTRDRBG;
}
