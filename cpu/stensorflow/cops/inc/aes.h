/**************************************************************************
Copyright © 2021, Ant Financial and/or its affiliates. All rights reserved.

Without express written authorization from Ant Financial, no one may conduct any of the following actions:
1) reproduce, spread, present, set up a mirror of, upload, download this source code;
2) modify, translate and adapt this source code, or develop derivative products, works, and services based on this source code; or
3) distribute, lease, rent, sub-license, demise or transfer any rights in relation to this source code, or authorize the reproduction of this source code on other's computers.
**************************************************************************/

#ifndef AES_H
#define AES_H

// returned value
#define SUCCESS_AES 0

// AES-128 key schedule
int aes128_set_enckey(unsigned char *encrk, unsigned char *key);
int aes128_set_deckey(unsigned char *decrk, unsigned char *key);

// AES-128 1 key 1 block encryption/decryption
int aes128_enc(unsigned char *cipher, unsigned char *plain, unsigned char *encrk);
int aes128_dec(unsigned char *plain, unsigned char *cipher, unsigned char *decrk);

// AES-128 4 keys 4 blocks encryption/decryption
int aes128_enc4(unsigned char *cipher, unsigned char *plain, unsigned char *encrk);
int aes128_dec4(unsigned char *plain, unsigned char *cipher, unsigned char *decrk);

#endif // AES_H
