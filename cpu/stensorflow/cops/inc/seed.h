/**************************************************************************
Copyright © 2021, Ant Financial and/or its affiliates. All rights reserved.

Without express written authorization from Ant Financial, no one may conduct any of the following actions:
1) reproduce, spread, present, set up a mirror of, upload, download this source code;
2) modify, translate and adapt this source code, or develop derivative products, works, and services based on this source code; or
3) distribute, lease, rent, sub-license, demise or transfer any rights in relation to this source code, or authorize the reproduction of this source code on other's computers.
**************************************************************************/

#ifndef SEED_H
#define SEED_H

// setting
#define USE_URANDOM // use /dev/urandom

// fixed parameter
#define MAX_LOOP 10000  // maximum number of attempts in get_seed
#define SLEEP_TIME 1000 // sleep time for calling read_devrandom again

// return value
#define SUCCESS_SEED 0
#define ERROR_SEED_SEEDBUF 1
#define ERROR_SEED_SEEDLEN 2
#define ERROR_SEED_DEVRANDOM 3

// return value
#define SUCCESS_SEED 0
#define ERROR_SEED_OPENFILE -1
#define ERROR_SEED_READFILE -2
#define ERROR_SEED_CLOSEFILE -3

// get random bytes from device random file
int read_devrandom(unsigned char *randbuf, int randnum);

// get seed
int get_seed(unsigned char *seed, int seedlen);

#endif // SEED_H
