# Box Of Hats (https://github.com/Box-Of-Hats)

import win32api as wapi

w = [1, 0, 0, 0, 0, 0, 0, 0, 0]
s = [0, 1, 0, 0, 0, 0, 0, 0, 0]
a = [0, 0, 1, 0, 0, 0, 0, 0, 0]
d = [0, 0, 0, 1, 0, 0, 0, 0, 0]
wa = [0, 0, 0, 0, 1, 0, 0, 0, 0]
wd = [0, 0, 0, 0, 0, 1, 0, 0, 0]
sa = [0, 0, 0, 0, 0, 0, 1, 0, 0]
sd = [0, 0, 0, 0, 0, 0, 0, 1, 0]
nk = [0, 0, 0, 0, 0, 0, 0, 0, 1]


key_list = ["\b"]
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'APS$/\\":
    key_list.append(char)


def key_check():
    keys = []
    for key in key_list:
        if wapi.GetAsyncKeyState(ord(key)):
            keys.append(key)
    return keys


def keys_to_output_AWD(keys):
    #        [A, W, D]
    output = [0, 0, 0]

    if 'A' in keys:
        output[0] = 1
    elif 'D' in keys:
        output[2] = 1
    else:
        output[1] = 1
    return output


def keys_to_output_complex(keys):
    """
    sentex
    Convert keys to a ...multi-hot... array
     0  1  2  3  4   5   6   7    8
    [W, S, A, D, WA, WD, SA, SD, NOKEY] boolean values.
    """


    output = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    if "W" in keys and "A" in keys:
        output = wa
    elif "W" in keys and "D" in keys:
        output = wd
    elif "S" in keys and "A" in keys:
        output = sa
    elif "S" in keys and "D" in keys:
        output = sd
    elif "W" in keys:
        output = w
    elif "S" in keys:
        output = s
    elif "A" in keys:
        output = a
    elif "D" in keys:
        output = d
    else:
        output = nk
    return output
