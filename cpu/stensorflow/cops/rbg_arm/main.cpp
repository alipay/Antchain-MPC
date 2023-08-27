/**************************************************************************
Copyright © 2021, Ant Financial and/or its affiliates. All rights reserved.

Without express written authorization from Ant Financial, no one may conduct any of the following actions:
1) reproduce, spread, present, set up a mirror of, upload, download this source code;
2) modify, translate and adapt this source code, or develop derivative products, works, and services based on this source code; or
3) distribute, lease, rent, sub-license, demise or transfer any rights in relation to this source code, or authorize the reproduction of this source code on other's computers.
**************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include "rbg.h"

void RBGTest_correctness()
{
    int i;
    int ret;
    int randnum;
    int seednum;
    long long *randbuf;
    long long *seed;
    long long *unifbuf;
    long long min, max;
    unsigned char *ctx;

    seednum = RBG_SEEDED_SEEDLEN / 8;
    seed = (long long *)malloc(RBG_SEEDED_SEEDLEN);
    memset(seed, 0, RBG_SEEDED_SEEDLEN);

    // rbg_seed
    ret = rbg_getseed(seed, seednum);
    if(ret!=0) printf("##### ret: %d #####\n", ret);

    printf("rbg_seed seednum: %d\n", seednum);
    for (i = 0; i < seednum; i++)
        printf("%08llx ", seed[i]);
    printf("\n");

    char lock_filepath[] = "Makefile";
    ret = rbg_getseed_lock(seed, seednum, lock_filepath);
    if(ret!=0) printf("##### ret: %d #####\n", ret);

    printf("rbg_getseed_lock seednum: %d\n", seednum);
    for (i = 0; i < seednum; i++)
        printf("%08llx ", seed[i]);
    printf("\n");

    randnum = 23;
    randbuf = (long long *)malloc(randnum * 8);
    memset(randbuf, 0, randnum * 8);

    unifbuf = (long long *)malloc(randnum * 8);
    memset(unifbuf, 0, randnum * 8);

    // rbg_nonseeded
    ret = rbg_nonseeded(randbuf, randnum);
    if(ret!=0) printf("##### ret: %d #####\n", ret);
    printf("rbg_nonseeded randnum: %d\n", randnum);
    for (i = 0; i < randnum; i++)
        printf("%08llx ", randbuf[i]);
    printf("\n");

    memset(randbuf, 0, randnum * 8);
    // rbg_seeded
    ret = rbg_seeded(randbuf, seed, randnum);
    if(ret!=0) printf("##### ret: %d #####\n", ret);
    printf("rbg_seeded randnum: %d\n", randnum);
    for (i = 0; i < randnum; i++)
        printf("%08llx ", randbuf[i]);
    printf("\n");

    ctx = (unsigned char *)malloc(RBG_SEEDED_CONTEXTLEN);
    memset(ctx, 0, RBG_SEEDED_CONTEXTLEN);

    // seeded RBG, instantiate RBG
    ret = rbg_seeded_instantiate(ctx, seed);
    if(ret!=0) printf("##### ret: %d #####\n", ret);

    memset(randbuf, 0, randnum * 8);
    // seeded RBG, generate long long random number array
    ret = rbg_seeded_random(ctx, randbuf, randnum);
    if(ret!=0) printf("##### ret: %d #####\n", ret);
    printf("rbg_seeded_random1 randnum: %d\n", randnum);
    for (i = 0; i < randnum; i++)
        printf("%08llx ", randbuf[i]);
    printf("\n");

    memset(randbuf, 0, randnum * 8);
    // seeded RBG, generate long long random number array
    ret = rbg_seeded_random(ctx, randbuf, randnum);
    if(ret!=0) printf("##### ret: %d #####\n", ret);
    printf("rbg_seeded_random2 randnum: %d\n", randnum);
    for (i = 0; i < randnum; i++)
        printf("%08llx ", randbuf[i]);
    printf("\n");

    // rbg_seed
    ret = rbg_getseed(seed, seednum);
    if(ret!=0) printf("##### ret: %d #####\n", ret);

    // seeded RBG, reseed RBG
    ret = rbg_seeded_reseed(ctx, seed);
    if(ret!=0) printf("##### ret: %d #####\n", ret);

    memset(randbuf, 0, randnum * 8);
    // seeded RBG, generate long long random number array
    ret = rbg_seeded_random(ctx, randbuf, randnum);
    if(ret!=0) printf("##### ret: %d #####\n", ret);
    printf("rbg_seeded_random3 randnum: %d\n", randnum);
    for (i = 0; i < randnum; i++)
        printf("%08lld ", randbuf[i]);
    printf("\n");

    // rbg_uniform
    min = -0x7FFFFFFFFFFFFFFFLL - 1LL;
    max = 0x7FFFFFFFFFFFFFFFLL - 1;
    // max = -3987LL;
    // min = -1223376272107128242LL;
    // max = 5LL;
    ret = rbg_nonseeded_uniform(unifbuf, randbuf, randnum, min, max);
    if(ret!=0) printf("##### ret: %d #####\n", ret);
    printf("rbg_nonseeded_uniform randnum: %d\n", randnum);
    for (i = 0; i < randnum; i++)
        printf("%08lld ", unifbuf[i]);
    printf("\n");

    memset(randbuf, 0, randnum * 8);
    // seeded RBG, generate long long random number array
    ret = rbg_seeded_random(ctx, randbuf, randnum);
    if(ret!=0) printf("##### ret: %d #####\n", ret);
    printf("rbg_seeded_random4 randnum: %d\n", randnum);
    for (i = 0; i < randnum; i++)
        printf("%08lld ", randbuf[i]);
    printf("\n");

    // rbg_seeded_uniform
    min = -1;
    // min = -0x7FFFFFFFFFFFFFFFLL-1LL;
    max = 0x7FFFFFFFFFFFFFFFLL - 1;
    // max = -3987LL;
    // min = -1223376272107128242LL;
    // max = 5LL;
    // max = -0x6FFFFFFFFFFFFFFFLL;
    ret = rbg_seeded_uniform(ctx, unifbuf, randbuf, randnum, min, max);
    if(ret!=0) printf("##### ret: %d #####\n", ret);
    printf("rbg_seeded_uniform randnum: %d\n", randnum);
    for (i = 0; i < randnum; i++)
        printf("%08lld ", unifbuf[i]);
    printf("\n");

    // seeded RBG, remove RBG
    ret = rbg_seeded_remove(ctx);
    if(ret!=0) printf("##### ret: %d #####\n", ret);

    free(unifbuf);
    free(randbuf);
    free(seed);
    free(ctx);
}

void RBGTest_performance()
{
    int i;
    int ret;
    int loop;
    int randlen, randnum;
    int seedlen, seednum;
    long long *randbuf;
    long long *seed;
    long long *unifbuf;
    unsigned char *ctx;
    long long min, max;

    struct timeval start, end;
    double time_pass;

    seedlen = 1024;
    seed = (long long *)malloc(seedlen);
    memset(seed, 0, seedlen);

    randlen = 64 * 1024 * 8;
    randbuf = (long long *)malloc(randlen);
    memset(randbuf, 0, randlen);

    unifbuf = (long long *)malloc(randlen);
    memset(unifbuf, 0, randlen);

    ctx = (unsigned char *)malloc(RBG_SEEDED_CONTEXTLEN);
    memset(ctx, 0, RBG_SEEDED_CONTEXTLEN);

    // rbg_getseed
    loop = 600;
    seednum = 1024 / 8;
    gettimeofday(&start, NULL);

    for (i = 0; i < loop; i++)
    {
        ret = rbg_getseed(seed, seednum);
        if(ret!=0) if(ret!=0) printf("##### ret: %d #####\n", ret);
    }

    gettimeofday(&end, NULL);
    time_pass = end.tv_sec - start.tv_sec + (float)(end.tv_usec - start.tv_usec) / 1000000;
    printf("rbg_getseed  time=%f s  tps=%f MB/s\n", time_pass, (double)loop / 1000 / 1000 * seednum * 8 / time_pass);

    // rbg_nonseeded
    loop = 100;
    randnum = 102400 / 8;
    gettimeofday(&start, NULL);

    for (i = 0; i < loop; i++)
    {
        ret = rbg_nonseeded(randbuf, randnum);
        if(ret!=0) printf("##### ret: %d #####\n", ret);
    }

    gettimeofday(&end, NULL);
    time_pass = end.tv_sec - start.tv_sec + (float)(end.tv_usec - start.tv_usec) / 1000000;
    printf("rbg_nonseeded  time=%f s  tps=%f MB/s\n", time_pass, (double)loop / 1000 / 1000 * randnum * 8 / time_pass);

    // rbg_nonseeded_uniform
    loop = 100;
    randnum = 102400 / 8;
    min = -0x7FFFFFFFFFFFFFFFLL - 1LL;
    // max = 0x7FFFFFFFFFFFFFFFLL-1;
    // max = 0x3FFFFFFFFFFFFFFFLL;
    // max = 0;
    // max = -0x6FFFFFFFFFFFFFFFLL;
    max = 0x7FFFFFFFFFFFFFFFLL;
    // max = 67;
    // min = -1;
    gettimeofday(&start, NULL);

    for (i = 0; i < loop; i++)
    {
        ret = rbg_nonseeded(randbuf, randnum);
        if(ret!=0) printf("##### ret: %d #####\n", ret);
        ret = rbg_nonseeded_uniform(unifbuf, randbuf, randnum, min, max);
        if(ret!=0) printf("##### ret: %d #####\n", ret);
    }

    gettimeofday(&end, NULL);
    time_pass = end.tv_sec - start.tv_sec + (float)(end.tv_usec - start.tv_usec) / 1000000;
    printf("rbg_noneseeded_uniform  time=%f s  tps=%f MB/s\n", time_pass, (double)loop / 1000 / 1000 * randnum * 8 / time_pass);

    // rbg_seeded
    loop = 40;
    randnum = 64 * 1024 * 8 / 8;
    gettimeofday(&start, NULL);

    for (i = 0; i < loop; i++)
    {
        ret = rbg_seeded(randbuf, seed, randnum);
        if(ret!=0) printf("##### ret: %d #####\n", ret);
    }

    gettimeofday(&end, NULL);
    time_pass = end.tv_sec - start.tv_sec + (float)(end.tv_usec - start.tv_usec) / 1000000;
    printf("rbg_seeded  time=%f s  tps=%f MB/s\n", time_pass, (double)loop / 1000 / 1000 * randnum * 8 / time_pass);

    // seeded RBG, instantiate RBG
    ret = rbg_seeded_instantiate(ctx, seed);
    if(ret!=0) printf("##### ret: %d #####\n", ret);

    // rbg_seeded_random
    loop = 40;
    randnum = 64 * 1024 * 8 / 8;
    // loop = 4000 * 64 * 2;
    // randnum = 512;
    gettimeofday(&start, NULL);

    for (i = 0; i < loop; i++)
    {
        ret = rbg_seeded_random(ctx, randbuf, randnum);
        if(ret!=0) printf("##### ret: %d #####\n", ret);
    }

    gettimeofday(&end, NULL);
    time_pass = end.tv_sec - start.tv_sec + (float)(end.tv_usec - start.tv_usec) / 1000000;
    printf("rbg_seeded_random  time=%f s  tps=%f MB/s\n", time_pass, (double)loop / 1000 / 1000 * randnum * 8 / time_pass);

    // rbg_seeded_uniform
    loop = 40;
    randnum = 64 * 1024 * 8 / 8;
    // min = -0x7FFFFFFFFFFFFFFFLL-1LL;
    // max = 0x7FFFFFFFFFFFFFFFLL-231;
    // max = 0x71FFFFFFFFFFFFFFLL;
    // max = 0x3FFFFFFFFFFFFFFFLL;
    // max = 0;
    // max = -0x6FFFFFFFFFFFFFFFLL;
    // max = 0x7FFFFFFFFFFFFFFFLL;
    max = 67;
    min = -1;
    gettimeofday(&start, NULL);

    for (i = 0; i < loop; i++)
    {
        ret = rbg_seeded_random(ctx, randbuf, randnum);
        if(ret!=0) printf("##### ret: %d #####\n", ret);
        ret = rbg_seeded_uniform(ctx, unifbuf, randbuf, randnum, min, max);
        if(ret!=0) printf("##### ret: %d #####\n", ret);
    }

    gettimeofday(&end, NULL);
    time_pass = end.tv_sec - start.tv_sec + (float)(end.tv_usec - start.tv_usec) / 1000000;
    printf("rbg_seeded_uniform  time=%f s  tps=%f MB/s\n", time_pass, (double)loop / 1000 / 1000 * randnum * 8 / time_pass);

    // seeded RBG, remove RBG
    ret = rbg_seeded_remove(ctx);
    if(ret!=0) printf("##### ret: %d #####\n", ret);

    free(unifbuf);
    free(randbuf);
    free(seed);
    free(ctx);
}

int main()
{

    RBGTest_correctness();

    RBGTest_performance();

    return 0;
}
