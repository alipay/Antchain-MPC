#!/usr/bin/env python
# coding:utf-8
import compileall
import re

# compileall.compile_dir('Lib/', force=True)


def comple_all():
    for dir in ['commonutils', 'stensorflow']:
        compileall.compile_dir(dir, rx=re.compile(r'[/\\][.]git'), force=True)


def main():
    comple_all()


if __name__ == "__main__":
    main()
