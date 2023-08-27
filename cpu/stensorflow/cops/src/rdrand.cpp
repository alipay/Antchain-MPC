/**************************************************************************
Copyright © 2021, Ant Financial and/or its affiliates. All rights reserved.

Without express written authorization from Ant Financial, no one may conduct any of the following actions:
1) reproduce, spread, present, set up a mirror of, upload, download this source code;
2) modify, translate and adapt this source code, or develop derivative products, works, and services based on this source code; or
3) distribute, lease, rent, sub-license, demise or transfer any rights in relation to this source code, or authorize the reproduction of this source code on other's computers.
**************************************************************************/

#include <string.h>
#include "rdrand.h"

//get exact multiples of 8 random bytes on calls to RDRAND
//rand, output, random buffer
//multiple, input, multiples
int get_rdrand_8bytes(unsigned char *rand, int multiple)
{
	__asm__(
		"movq %0, %%rcx         \n\t"
		"mov %1, %%edx          \n\t"

		"1:             		\n\t"
		"rdrand %%rax      		\n\t"
		"jnc 1b         		\n\t" //retry

		"movq %%rax, (%%rcx)	\n\t"
		"addq $8, %%rcx			\n\t"
		"sub $1, %%edx			\n\t"
		"jne 1b					\n\t" //next 8 bytes

		:
		: "r"(rand), "r"(multiple)
		: "memory", "cc", "%rax", "%rcx", "%edx");

	return SUCCESS_RDRAND;
}

//get random bytes on calls to RDRAND
//rand, output, random buffer
//randlen, input, number of bytes
int get_rdrand(unsigned char *rand, int randlen)
{
	int multiple, left;
	unsigned char buffer[8];

	//check parameters
	if (rand == NULL)
		return ERROR_RDRAND_RANDBUF;

	if (randlen < 1)
		return ERROR_RDRAND_RANDLEN;

	left = randlen % 8;
	multiple = (randlen - left) / 8;

	//exact multiples of 8
	if (multiple)
		get_rdrand_8bytes(rand, multiple);

	//less than 8
	if (left) {
		get_rdrand_8bytes(buffer, 1);
		memcpy(rand + 8 * multiple, buffer, left);
	}

	return SUCCESS_RDRAND;
}
