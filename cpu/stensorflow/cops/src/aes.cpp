/**************************************************************************
Copyright © 2021, Ant Financial and/or its affiliates. All rights reserved.

Without express written authorization from Ant Financial, no one may conduct any of the following actions:
1) reproduce, spread, present, set up a mirror of, upload, download this source code;
2) modify, translate and adapt this source code, or develop derivative products, works, and services based on this source code; or
3) distribute, lease, rent, sub-license, demise or transfer any rights in relation to this source code, or authorize the reproduction of this source code on other's computers.
**************************************************************************/

#include "aes.h"

//AES-128 key schedule for encryption round keys
//rk, output, encryption round keys
//key, input, key
int aes128_set_enckey(unsigned char *rk, unsigned char *key)
{
	__asm__(
		"vmovdqu (%1), %%xmm0					\n\t"
		"vmovdqu %%xmm0, (%0)					\n\t" //rk0
		"jmp 2f									\n\t"

		"1:										\n\t"
		"vpshufd $0xff, %%xmm1, %%xmm1			\n\t"
		"vpxor %%xmm0, %%xmm1, %%xmm1			\n\t"
		"vpslldq $4, %%xmm0, %%xmm0				\n\t"
		"vpxor %%xmm0, %%xmm1, %%xmm1			\n\t"
		"vpslldq $4, %%xmm0, %%xmm0				\n\t"
		"vpxor %%xmm0, %%xmm1, %%xmm1			\n\t"
		"vpslldq $4, %%xmm0, %%xmm0				\n\t"
		"vpxor %%xmm1, %%xmm0, %%xmm0			\n\t"
		"add $16, %0							\n\t"
		"vmovdqu %%xmm0, (%0)					\n\t"
		"ret									\n\t"

		"2:										\n\t"
		"vaeskeygenassist $0x1, %%xmm0, %%xmm1  \n\t"
		"call 1b								\n\t" //rk1
		"vaeskeygenassist $0x2, %%xmm0, %%xmm1	\n\t"
		"call 1b								\n\t" //rk2
		"vaeskeygenassist $0x4, %%xmm0, %%xmm1	\n\t"
		"call 1b								\n\t" //rk3
		"vaeskeygenassist $0x8, %%xmm0, %%xmm1	\n\t"
		"call 1b								\n\t" //rk4
		"vaeskeygenassist $0x10, %%xmm0, %%xmm1	\n\t"
		"call 1b								\n\t" //rk5
		"vaeskeygenassist $0x20, %%xmm0, %%xmm1	\n\t"
		"call 1b								\n\t" //rk6
		"vaeskeygenassist $0x40, %%xmm0, %%xmm1	\n\t"
		"call 1b								\n\t" //rk7
		"vaeskeygenassist $0x80, %%xmm0, %%xmm1	\n\t"
		"call 1b								\n\t" //rk8
		"vaeskeygenassist $0x1b, %%xmm0, %%xmm1	\n\t"
		"call 1b								\n\t" //rk9
		"vaeskeygenassist $0x36, %%xmm0, %%xmm1	\n\t"
		"call 1b								\n\t" //rk10

		:
		: "r"(rk), "r"(key)
		: "memory", "cc", "xmm0", "xmm1");

	return SUCCESS_AES;
}

//AES-128 key schedule for decryption round keys
//rk, output, decryption round keys
//key, input, key
int aes128_set_deckey(unsigned char *rk, unsigned char *key)
{
	__asm__(
		"vmovdqu (%1), %%xmm0					\n\t"
		"add $160, %0							\n\t"
		"vmovdqu %%xmm0, (%0)					\n\t" //rk10
		"jmp 2f									\n\t"

		"1:										\n\t"
		"vpshufd $0xff, %%xmm1, %%xmm1			\n\t"
		"vpxor %%xmm0, %%xmm1, %%xmm1			\n\t"
		"vpslldq $4, %%xmm0, %%xmm0				\n\t"
		"vpxor %%xmm0, %%xmm1, %%xmm1			\n\t"
		"vpslldq $4, %%xmm0, %%xmm0				\n\t"
		"vpxor %%xmm0, %%xmm1, %%xmm1			\n\t"
		"vpslldq $4, %%xmm0, %%xmm0				\n\t"
		"vpxor %%xmm1, %%xmm0, %%xmm0			\n\t"
		"vaesimc %%xmm0, %%xmm2					\n\t"
		"sub $16, %0							\n\t"
		"vmovdqu %%xmm2, (%0)					\n\t"
		"ret									\n\t"

		"2:										\n\t"
		"vaeskeygenassist $0x1, %%xmm0, %%xmm1  \n\t"
		"call 1b								\n\t" //rk9
		"vaeskeygenassist $0x2, %%xmm0, %%xmm1	\n\t"
		"call 1b								\n\t" //rk8
		"vaeskeygenassist $0x4, %%xmm0, %%xmm1	\n\t"
		"call 1b								\n\t" //rk7
		"vaeskeygenassist $0x8, %%xmm0, %%xmm1	\n\t"
		"call 1b								\n\t" //rk6
		"vaeskeygenassist $0x10, %%xmm0, %%xmm1	\n\t"
		"call 1b								\n\t" //rk5
		"vaeskeygenassist $0x20, %%xmm0, %%xmm1	\n\t"
		"call 1b								\n\t" //rk4
		"vaeskeygenassist $0x40, %%xmm0, %%xmm1	\n\t"
		"call 1b								\n\t" //rk3
		"vaeskeygenassist $0x80, %%xmm0, %%xmm1	\n\t"
		"call 1b								\n\t" //rk2
		"vaeskeygenassist $0x1b, %%xmm0, %%xmm1	\n\t"
		"call 1b								\n\t" //rk1
		"vaeskeygenassist $0x36, %%xmm0, %%xmm1	\n\t"

		"vpshufd $0xff, %%xmm1, %%xmm1			\n\t"
		"vpxor %%xmm0, %%xmm1, %%xmm1			\n\t"
		"vpslldq $4, %%xmm0, %%xmm0				\n\t"
		"vpxor %%xmm0, %%xmm1, %%xmm1			\n\t"
		"vpslldq $4, %%xmm0, %%xmm0				\n\t"
		"vpxor %%xmm0, %%xmm1, %%xmm1			\n\t"
		"vpslldq $4, %%xmm0, %%xmm0				\n\t"
		"vpxor %%xmm1, %%xmm0, %%xmm0			\n\t"
		"sub $16, %0							\n\t"
		"vmovdqu %%xmm0, (%0)					\n\t" //rk0

		:
		: "r"(rk), "r"(key)
		: "memory", "cc", "xmm0", "xmm1", "xmm2");

	return SUCCESS_AES;
}

//AES-128 1 key 1 block encryption
//cipher, output, ciphertext
//plain, input, plaintext
//rk, input, encryption round keys
int aes128_enc(unsigned char *cipher, unsigned char *plain, unsigned char *rk)
{
	__asm__(
		"vmovdqu (%1), %%xmm0					\n\t"
		"vmovdqu (%2), %%xmm1					\n\t" //rk0
		"vpxor %%xmm1, %%xmm0, %%xmm0			\n\t"
		"vmovdqu 16(%2), %%xmm1					\n\t" //rk1
		"vaesenc %%xmm1, %%xmm0, %%xmm0			\n\t"
		"vmovdqu 32(%2), %%xmm1					\n\t" //rk2
		"vaesenc %%xmm1, %%xmm0, %%xmm0			\n\t"
		"vmovdqu 48(%2), %%xmm1					\n\t" //rk3
		"vaesenc %%xmm1, %%xmm0, %%xmm0			\n\t"
		"vmovdqu 64(%2), %%xmm1					\n\t" //rk4
		"vaesenc %%xmm1, %%xmm0, %%xmm0			\n\t"
		"vmovdqu 80(%2), %%xmm1					\n\t" //rk5
		"vaesenc %%xmm1, %%xmm0, %%xmm0			\n\t"
		"vmovdqu 96(%2), %%xmm1					\n\t" //rk6
		"vaesenc %%xmm1, %%xmm0, %%xmm0			\n\t"
		"vmovdqu 112(%2), %%xmm1				\n\t" //rk7
		"vaesenc %%xmm1, %%xmm0, %%xmm0			\n\t"
		"vmovdqu 128(%2), %%xmm1				\n\t" //rk8
		"vaesenc %%xmm1, %%xmm0, %%xmm0			\n\t"
		"vmovdqu 144(%2), %%xmm1				\n\t" //rk9
		"vaesenc %%xmm1, %%xmm0, %%xmm0			\n\t"
		"vmovdqu 160(%2), %%xmm1				\n\t" //rk10
		"vaesenclast %%xmm1, %%xmm0, %%xmm0		\n\t"
		"vmovdqu %%xmm0, (%0)					\n\t"

		:
		: "r"(cipher), "r"(plain), "r"(rk)
		: "memory", "cc", "xmm0", "xmm1");

	return SUCCESS_AES;
}

//AES-128 1 key 1 block decryption
//plain, output, plaintext
//cipher, input, ciphertext
//rk, input, decryption round keys
int aes128_dec(unsigned char *plain, unsigned char *cipher, unsigned char *rk)
{
	__asm__(
		"vmovdqu (%1), %%xmm0					\n\t"
		"vmovdqu (%2), %%xmm1					\n\t" //rk0
		"vpxor %%xmm1, %%xmm0, %%xmm0			\n\t"
		"vmovdqu 16(%2), %%xmm1					\n\t" //rk1
		"vaesdec %%xmm1, %%xmm0, %%xmm0			\n\t"
		"vmovdqu 32(%2), %%xmm1					\n\t" //rk2
		"vaesdec %%xmm1, %%xmm0, %%xmm0			\n\t"
		"vmovdqu 48(%2), %%xmm1					\n\t" //rk3
		"vaesdec %%xmm1, %%xmm0, %%xmm0			\n\t"
		"vmovdqu 64(%2), %%xmm1					\n\t" //rk4
		"vaesdec %%xmm1, %%xmm0, %%xmm0			\n\t"
		"vmovdqu 80(%2), %%xmm1					\n\t" //rk5
		"vaesdec %%xmm1, %%xmm0, %%xmm0			\n\t"
		"vmovdqu 96(%2), %%xmm1					\n\t" //rk6
		"vaesdec %%xmm1, %%xmm0, %%xmm0			\n\t"
		"vmovdqu 112(%2), %%xmm1				\n\t" //rk7
		"vaesdec %%xmm1, %%xmm0, %%xmm0			\n\t"
		"vmovdqu 128(%2), %%xmm1				\n\t" //rk8
		"vaesdec %%xmm1, %%xmm0, %%xmm0			\n\t"
		"vmovdqu 144(%2), %%xmm1				\n\t" //rk9
		"vaesdec %%xmm1, %%xmm0, %%xmm0			\n\t"
		"vmovdqu 160(%2), %%xmm1				\n\t" //rk10
		"vaesdeclast %%xmm1, %%xmm0, %%xmm0		\n\t"
		"vmovdqu %%xmm0, (%0)					\n\t"

		:
		: "r"(plain), "r"(cipher), "r"(rk)
		: "memory", "cc", "xmm0", "xmm1");

	return SUCCESS_AES;
}

//AES-128 4 keys 4 blocks encryption
//cipher, output, ciphertext
//plain, input, plaintext
//rk, input, encryption round keys
int aes128_enc4(unsigned char *cipher, unsigned char *plain, unsigned char *rk)
{
	__asm__(
		"vmovdqu (%1), %%xmm0					\n\t"
		"vmovdqu 16(%1), %%xmm2					\n\t"
		"vmovdqu 32(%1), %%xmm4					\n\t"
		"vmovdqu 48(%1), %%xmm6					\n\t"

		"vmovdqu (%2), %%xmm1					\n\t" //rk0
		"vmovdqu 176(%2), %%xmm3				\n\t"
		"vmovdqu 352(%2), %%xmm5				\n\t"
		"vmovdqu 528(%2), %%xmm7				\n\t"
		"vpxor %%xmm1, %%xmm0, %%xmm0			\n\t"
		"vpxor %%xmm3, %%xmm2, %%xmm2			\n\t"
		"vpxor %%xmm5, %%xmm4, %%xmm4			\n\t"
		"vpxor %%xmm7, %%xmm6, %%xmm6			\n\t"

		"vmovdqu 16(%2), %%xmm1					\n\t" //rk1
		"vmovdqu 16+176(%2), %%xmm3				\n\t"
		"vmovdqu 16+352(%2), %%xmm5				\n\t"
		"vmovdqu 16+528(%2), %%xmm7				\n\t"
		"vaesenc %%xmm1, %%xmm0, %%xmm0			\n\t"
		"vaesenc %%xmm3, %%xmm2, %%xmm2			\n\t"
		"vaesenc %%xmm5, %%xmm4, %%xmm4			\n\t"
		"vaesenc %%xmm7, %%xmm6, %%xmm6			\n\t"

		"vmovdqu 32(%2), %%xmm1					\n\t" //rk2
		"vmovdqu 32+176(%2), %%xmm3				\n\t"
		"vmovdqu 32+352(%2), %%xmm5				\n\t"
		"vmovdqu 32+528(%2), %%xmm7				\n\t"
		"vaesenc %%xmm1, %%xmm0, %%xmm0			\n\t"
		"vaesenc %%xmm3, %%xmm2, %%xmm2			\n\t"
		"vaesenc %%xmm5, %%xmm4, %%xmm4			\n\t"
		"vaesenc %%xmm7, %%xmm6, %%xmm6			\n\t"

		"vmovdqu 48(%2), %%xmm1					\n\t" //rk3
		"vmovdqu 48+176(%2), %%xmm3				\n\t"
		"vmovdqu 48+352(%2), %%xmm5				\n\t"
		"vmovdqu 48+528(%2), %%xmm7				\n\t"
		"vaesenc %%xmm1, %%xmm0, %%xmm0			\n\t"
		"vaesenc %%xmm3, %%xmm2, %%xmm2			\n\t"
		"vaesenc %%xmm5, %%xmm4, %%xmm4			\n\t"
		"vaesenc %%xmm7, %%xmm6, %%xmm6			\n\t"

		"vmovdqu 64(%2), %%xmm1					\n\t" //rk4
		"vmovdqu 64+176(%2), %%xmm3				\n\t"
		"vmovdqu 64+352(%2), %%xmm5				\n\t"
		"vmovdqu 64+528(%2), %%xmm7				\n\t"
		"vaesenc %%xmm1, %%xmm0, %%xmm0			\n\t"
		"vaesenc %%xmm3, %%xmm2, %%xmm2			\n\t"
		"vaesenc %%xmm5, %%xmm4, %%xmm4			\n\t"
		"vaesenc %%xmm7, %%xmm6, %%xmm6			\n\t"

		"vmovdqu 80(%2), %%xmm1					\n\t" //rk5
		"vmovdqu 80+176(%2), %%xmm3				\n\t"
		"vmovdqu 80+352(%2), %%xmm5				\n\t"
		"vmovdqu 80+528(%2), %%xmm7				\n\t"
		"vaesenc %%xmm1, %%xmm0, %%xmm0			\n\t"
		"vaesenc %%xmm3, %%xmm2, %%xmm2			\n\t"
		"vaesenc %%xmm5, %%xmm4, %%xmm4			\n\t"
		"vaesenc %%xmm7, %%xmm6, %%xmm6			\n\t"

		"vmovdqu 96(%2), %%xmm1					\n\t" //rk6
		"vmovdqu 96+176(%2), %%xmm3				\n\t"
		"vmovdqu 96+352(%2), %%xmm5				\n\t"
		"vmovdqu 96+528(%2), %%xmm7				\n\t"
		"vaesenc %%xmm1, %%xmm0, %%xmm0			\n\t"
		"vaesenc %%xmm3, %%xmm2, %%xmm2			\n\t"
		"vaesenc %%xmm5, %%xmm4, %%xmm4			\n\t"
		"vaesenc %%xmm7, %%xmm6, %%xmm6			\n\t"

		"vmovdqu 112(%2), %%xmm1				\n\t" //rk7
		"vmovdqu 112+176(%2), %%xmm3			\n\t"
		"vmovdqu 112+352(%2), %%xmm5			\n\t"
		"vmovdqu 112+528(%2), %%xmm7			\n\t"
		"vaesenc %%xmm1, %%xmm0, %%xmm0			\n\t"
		"vaesenc %%xmm3, %%xmm2, %%xmm2			\n\t"
		"vaesenc %%xmm5, %%xmm4, %%xmm4			\n\t"
		"vaesenc %%xmm7, %%xmm6, %%xmm6			\n\t"

		"vmovdqu 128(%2), %%xmm1				\n\t" //rk8
		"vmovdqu 128+176(%2), %%xmm3			\n\t"
		"vmovdqu 128+352(%2), %%xmm5			\n\t"
		"vmovdqu 128+528(%2), %%xmm7			\n\t"
		"vaesenc %%xmm1, %%xmm0, %%xmm0			\n\t"
		"vaesenc %%xmm3, %%xmm2, %%xmm2			\n\t"
		"vaesenc %%xmm5, %%xmm4, %%xmm4			\n\t"
		"vaesenc %%xmm7, %%xmm6, %%xmm6			\n\t"

		"vmovdqu 144(%2), %%xmm1				\n\t" //rk9
		"vmovdqu 144+176(%2), %%xmm3			\n\t"
		"vmovdqu 144+352(%2), %%xmm5			\n\t"
		"vmovdqu 144+528(%2), %%xmm7			\n\t"
		"vaesenc %%xmm1, %%xmm0, %%xmm0			\n\t"
		"vaesenc %%xmm3, %%xmm2, %%xmm2			\n\t"
		"vaesenc %%xmm5, %%xmm4, %%xmm4			\n\t"
		"vaesenc %%xmm7, %%xmm6, %%xmm6			\n\t"

		"vmovdqu 160(%2), %%xmm1				\n\t" //rk10
		"vmovdqu 160+176(%2), %%xmm3			\n\t"
		"vmovdqu 160+352(%2), %%xmm5			\n\t"
		"vmovdqu 160+528(%2), %%xmm7			\n\t"
		"vaesenclast %%xmm1, %%xmm0, %%xmm0		\n\t"
		"vaesenclast %%xmm3, %%xmm2, %%xmm2		\n\t"
		"vaesenclast %%xmm5, %%xmm4, %%xmm4		\n\t"
		"vaesenclast %%xmm7, %%xmm6, %%xmm6		\n\t"
		"vmovdqu %%xmm0, (%0)            		\n\t"
		"vmovdqu %%xmm2, 16(%0)            		\n\t"
		"vmovdqu %%xmm4, 32(%0)            		\n\t"
		"vmovdqu %%xmm6, 48(%0)            		\n\t"

		:
		: "r"(cipher), "r"(plain), "r"(rk)
		: "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7");

	return SUCCESS_AES;
}

//AES-128 4 keys 4 blocks decryption
//plain, output, plaintext
//cipher, input, ciphertext
//rk, input, decryption round keys
int aes128_dec4(unsigned char *plain, unsigned char *cipher, unsigned char *rk)
{
	__asm__(
		"vmovdqu (%1), %%xmm0					\n\t"
		"vmovdqu 16(%1), %%xmm2					\n\t"
		"vmovdqu 32(%1), %%xmm4					\n\t"
		"vmovdqu 48(%1), %%xmm6					\n\t"

		"vmovdqu (%2), %%xmm1					\n\t" //rk0
		"vmovdqu 176(%2), %%xmm3				\n\t"
		"vmovdqu 352(%2), %%xmm5				\n\t"
		"vmovdqu 528(%2), %%xmm7				\n\t"
		"vpxor %%xmm1, %%xmm0, %%xmm0			\n\t"
		"vpxor %%xmm3, %%xmm2, %%xmm2			\n\t"
		"vpxor %%xmm5, %%xmm4, %%xmm4			\n\t"
		"vpxor %%xmm7, %%xmm6, %%xmm6			\n\t"

		"vmovdqu 16(%2), %%xmm1					\n\t" //rk1
		"vmovdqu 16+176(%2), %%xmm3				\n\t"
		"vmovdqu 16+352(%2), %%xmm5				\n\t"
		"vmovdqu 16+528(%2), %%xmm7				\n\t"
		"vaesdec %%xmm1, %%xmm0, %%xmm0			\n\t"
		"vaesdec %%xmm3, %%xmm2, %%xmm2			\n\t"
		"vaesdec %%xmm5, %%xmm4, %%xmm4			\n\t"
		"vaesdec %%xmm7, %%xmm6, %%xmm6			\n\t"

		"vmovdqu 32(%2), %%xmm1					\n\t" //rk2
		"vmovdqu 32+176(%2), %%xmm3				\n\t"
		"vmovdqu 32+352(%2), %%xmm5				\n\t"
		"vmovdqu 32+528(%2), %%xmm7				\n\t"
		"vaesdec %%xmm1, %%xmm0, %%xmm0			\n\t"
		"vaesdec %%xmm3, %%xmm2, %%xmm2			\n\t"
		"vaesdec %%xmm5, %%xmm4, %%xmm4			\n\t"
		"vaesdec %%xmm7, %%xmm6, %%xmm6			\n\t"

		"vmovdqu 48(%2), %%xmm1					\n\t" //rk3
		"vmovdqu 48+176(%2), %%xmm3				\n\t"
		"vmovdqu 48+352(%2), %%xmm5				\n\t"
		"vmovdqu 48+528(%2), %%xmm7				\n\t"
		"vaesdec %%xmm1, %%xmm0, %%xmm0			\n\t"
		"vaesdec %%xmm3, %%xmm2, %%xmm2			\n\t"
		"vaesdec %%xmm5, %%xmm4, %%xmm4			\n\t"
		"vaesdec %%xmm7, %%xmm6, %%xmm6			\n\t"

		"vmovdqu 64(%2), %%xmm1					\n\t" //rk4
		"vmovdqu 64+176(%2), %%xmm3				\n\t"
		"vmovdqu 64+352(%2), %%xmm5				\n\t"
		"vmovdqu 64+528(%2), %%xmm7				\n\t"
		"vaesdec %%xmm1, %%xmm0, %%xmm0			\n\t"
		"vaesdec %%xmm3, %%xmm2, %%xmm2			\n\t"
		"vaesdec %%xmm5, %%xmm4, %%xmm4			\n\t"
		"vaesdec %%xmm7, %%xmm6, %%xmm6			\n\t"

		"vmovdqu 80(%2), %%xmm1					\n\t" //rk5
		"vmovdqu 80+176(%2), %%xmm3				\n\t"
		"vmovdqu 80+352(%2), %%xmm5				\n\t"
		"vmovdqu 80+528(%2), %%xmm7				\n\t"
		"vaesdec %%xmm1, %%xmm0, %%xmm0			\n\t"
		"vaesdec %%xmm3, %%xmm2, %%xmm2			\n\t"
		"vaesdec %%xmm5, %%xmm4, %%xmm4			\n\t"
		"vaesdec %%xmm7, %%xmm6, %%xmm6			\n\t"

		"vmovdqu 96(%2), %%xmm1					\n\t" //rk6
		"vmovdqu 96+176(%2), %%xmm3				\n\t"
		"vmovdqu 96+352(%2), %%xmm5				\n\t"
		"vmovdqu 96+528(%2), %%xmm7				\n\t"
		"vaesdec %%xmm1, %%xmm0, %%xmm0			\n\t"
		"vaesdec %%xmm3, %%xmm2, %%xmm2			\n\t"
		"vaesdec %%xmm5, %%xmm4, %%xmm4			\n\t"
		"vaesdec %%xmm7, %%xmm6, %%xmm6			\n\t"

		"vmovdqu 112(%2), %%xmm1				\n\t" //rk7
		"vmovdqu 112+176(%2), %%xmm3			\n\t"
		"vmovdqu 112+352(%2), %%xmm5			\n\t"
		"vmovdqu 112+528(%2), %%xmm7			\n\t"
		"vaesdec %%xmm1, %%xmm0, %%xmm0			\n\t"
		"vaesdec %%xmm3, %%xmm2, %%xmm2			\n\t"
		"vaesdec %%xmm5, %%xmm4, %%xmm4			\n\t"
		"vaesdec %%xmm7, %%xmm6, %%xmm6			\n\t"

		"vmovdqu 128(%2), %%xmm1				\n\t" //rk8
		"vmovdqu 128+176(%2), %%xmm3			\n\t"
		"vmovdqu 128+352(%2), %%xmm5			\n\t"
		"vmovdqu 128+528(%2), %%xmm7			\n\t"
		"vaesdec %%xmm1, %%xmm0, %%xmm0			\n\t"
		"vaesdec %%xmm3, %%xmm2, %%xmm2			\n\t"
		"vaesdec %%xmm5, %%xmm4, %%xmm4			\n\t"
		"vaesdec %%xmm7, %%xmm6, %%xmm6			\n\t"

		"vmovdqu 144(%2), %%xmm1				\n\t" //rk9
		"vmovdqu 144+176(%2), %%xmm3			\n\t"
		"vmovdqu 144+352(%2), %%xmm5			\n\t"
		"vmovdqu 144+528(%2), %%xmm7			\n\t"
		"vaesdec %%xmm1, %%xmm0, %%xmm0			\n\t"
		"vaesdec %%xmm3, %%xmm2, %%xmm2			\n\t"
		"vaesdec %%xmm5, %%xmm4, %%xmm4			\n\t"
		"vaesdec %%xmm7, %%xmm6, %%xmm6			\n\t"

		"vmovdqu 160(%2), %%xmm1				\n\t" //rk10
		"vmovdqu 160+176(%2), %%xmm3			\n\t"
		"vmovdqu 160+352(%2), %%xmm5			\n\t"
		"vmovdqu 160+528(%2), %%xmm7			\n\t"
		"vaesdeclast %%xmm1, %%xmm0, %%xmm0		\n\t"
		"vaesdeclast %%xmm3, %%xmm2, %%xmm2		\n\t"
		"vaesdeclast %%xmm5, %%xmm4, %%xmm4		\n\t"
		"vaesdeclast %%xmm7, %%xmm6, %%xmm6		\n\t"
		"vmovdqu %%xmm0, (%0)            		\n\t"
		"vmovdqu %%xmm2, 16(%0)            		\n\t"
		"vmovdqu %%xmm4, 32(%0)            		\n\t"
		"vmovdqu %%xmm6, 48(%0)            		\n\t"

		:
		: "r"(plain), "r"(cipher), "r"(rk)
		: "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7");

	return SUCCESS_AES;
}
