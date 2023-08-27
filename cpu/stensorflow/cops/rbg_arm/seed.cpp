/**************************************************************************
Copyright © 2021, Ant Financial and/or its affiliates. All rights reserved.

Without express written authorization from Ant Financial, no one may conduct any of the following actions:
1) reproduce, spread, present, set up a mirror of, upload, download this source code;
2) modify, translate and adapt this source code, or develop derivative products, works, and services based on this source code; or
3) distribute, lease, rent, sub-license, demise or transfer any rights in relation to this source code, or authorize the reproduction of this source code on other's computers.
**************************************************************************/

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "seed.h"

// get random bytes from device random file
// randbuf, output, random buffer
// randnum, input, number of bytes
int read_devrandom(unsigned char *randbuf, int randlen)
{
	int fd = -2;
	int num = -2;

	// set device random file
#ifndef USE_URANDOM
	char dev_file[] = "/dev/random";
#else
	char dev_file[] = "/dev/urandom";
#endif

	// open file
	fd = open(dev_file, O_RDONLY | O_NONBLOCK);
	if (fd == -1)
		return ERROR_SEED_OPENFILE;

	// get bytes
	num = read(fd, randbuf, randlen);
	if (num == -1)
		return ERROR_SEED_READFILE;

	// close file
	if (close(fd) == -1)
		return ERROR_SEED_CLOSEFILE;

	return num;
}

// get seed
// seed, output, seed buffer
// seedlen, input, number of bytes
int get_seed(unsigned char *seed, int seedlen)
{
	int i;
	int num = 0, left = 0;
	unsigned char *seedbuf = NULL;

	// check parameters
	if (seed == NULL)
		return ERROR_SEED_SEEDBUF;

	if (seedlen < 1)
		return ERROR_SEED_SEEDLEN;

	seedbuf = seed;
	left = seedlen;

	// try MAX_LOOP
	for (i = 0; i < MAX_LOOP; i++)
	{
		// get enough bytes
		if (left == 0)
			break;

		// get bytes
		num = read_devrandom(seedbuf, left);

		// read_devrandom fail
		if (num < 0)
			return ERROR_SEED_DEVRANDOM;

		seedbuf = seedbuf + num;
		left = left - num;

		// sleep SLEEP_TIME for next calling
		usleep(SLEEP_TIME);
	}

	return SUCCESS_SEED;
}
