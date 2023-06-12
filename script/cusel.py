#!/usr/bin/python3
'''
Blocking CUDA Device Selector

Usage: cusel [-m mem -n num]

Without cusel:

    $ gpustat  # (then manually decide the card to use)
    $ CUDA_VISISBLE_DEVICES=5 python3 train.py

With cusel:

    1. wait for a card with 11000MB idle memory and automatically select it
    $ CUDA_VISIBLE_DEVICES=$(cusel) python3 train.py

    2. wait for a card with 4000MB idle memory and automatically select it
    $ CUDA_VISIBLE_DEVICES=$(cusel -m 4000) python3 train.py

    3. wait for 2 cards with 4000MB idle memory on each card and automatically select them
    $ CUDA_VISIBLE_DEVICES=$(cusel -m 4000 -n 2) python3 train.py

Copyright (C) 2020 Mo Zhou <lumin@debian.org>
License: MIT/Expat
'''
from typing import *
import os
import time
import random
import argparse
import re
import fcntl
import subprocess as sp


def __getCards(exclude: list = []) -> dict:
    '''
    Get a dictionary of GPU status. GPU ID: int -> attributes: dict.
    '''
    gpustat = sp.Popen(['gpustat'], stdout=sp.PIPE, stderr=sp.PIPE).communicate()[0].decode().strip()
    lines = [line for line in gpustat.split('\n') if re.match(r'^\[\d\]', line)]
    ret = dict()
    for line in lines:
        idx, usage, muse, mall = re.match(r'^\[(\d)\].*,\s+(\d+)\s+%\s+\|\s+(\d+)\s+/\s+(\d+) MB', line).groups()
        if int(idx) not in exclude:
            ret[int(idx)] = (int(usage), int(muse), int(mall))
    return ret


def __selectCard(cards: dict, mem: int) -> Union[int, None]:
    '''
    Select a card. This function does not block.
    '''
    
    clist = []
    selected_list = []
    for (k, (us, mu, ma)) in cards.items():
        clist.append((k, us, mu, ma))
    clist = sorted(clist, key=lambda x: (x[1] + 1) * x[2])
    for (k, us, mu, ma) in clist:
        if ma - mu >= mem:
            selected_list.append(k)
    return selected_list

def find(ag):
    if ag.seq:
        cards = __getCards(ag.exclude)
        num_cards = len(cards)
        print(list(cards.keys())[ag.seq_i % num_cards])
    else:
        # Acquire lock (concurrent cusel)
        lock = open(ag.lock, 'w+')
        fcntl.lockf(lock, fcntl.LOCK_EX)

        # Polling cards
        while True:
            cards = __getCards(ag.exclude)
            sel = __selectCard(cards, ag.m)
            if len(sel) >= ag.n:
                sel = ','.join([str(idx) for idx in sel[:ag.n]])
                return sel
                print(sel)
                break
            else:
                time.sleep(ag.i)

        # Release lock (concurrent cusel)
        fcntl.lockf(lock, fcntl.LOCK_UN)
        lock.close()


class AG:
    pass
def cusel(m=11000, n=3, i=30, exclude=[], lock=f'/tmp/yhzhai-cusel.lock', seq=False, seq_i=0):
    paras = locals()
    ag = AG()
    for k,v in paras.items():
        ag.__setattr__(k, v)
    return find(ag)
    


if __name__ == '__main__':

    os.putenv('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')

    ag = argparse.ArgumentParser()
    ag.add_argument('-m', type=int, default=11000, help='how much memory (MB) on each card')
    ag.add_argument('-n', type=int, default=3, help='how many card')
    ag.add_argument('-i', type=int, default=30, help='polling interval')
    ag.add_argument('--exclude', type=int, default=[], nargs='+',
                    help='exclude gpu indices')
    ag.add_argument('--lock', type=str, default=f'/tmp/yhzhai-cusel.lock')
    ag.add_argument('--seq', action='store_true', default=False)
    ag.add_argument('--seq_i', type=int, default=0)
    ag = ag.parse_args()

    res = find(ag)
    print(res)


