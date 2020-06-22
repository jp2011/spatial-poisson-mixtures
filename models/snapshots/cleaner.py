import os
import re


files = os.listdir(os.getcwd())

matched_files = [re.match(r'(logalpha-|log-alpha-|alpha-|beta-|hmc-|Z-)samples--(.*)--([0-9]*)\.npy', fname) for fname in files]
matched_files = [x for x in matched_files if x is not None]

decomposed_files = [(match.group(1), match.group(2), match.group(3)) for match in matched_files]

uids = []
for _, uid, _ in decomposed_files:
    uids.append(uid)
unique_uids = set(uids)

for u_uid in unique_uids:
    max_samples = 0
    for _, _uid, _n in decomposed_files:
        if _uid == u_uid and int(_n) > max_samples:
            max_samples = int(_n)

    for ftype, _uid, _n in decomposed_files:
        if _uid == u_uid and int(_n) < max_samples:
            try:
                os.remove("{}/hmc-samples--{}--{}.npy".format(os.getcwd(), _uid, _n))
            except Exception as e:
                print(e)

            try:
                os.remove("{}/Z-samples--{}--{}.npy".format(os.getcwd(), _uid, _n))
            except Exception as e:
                print(e)

            try:
                os.remove("{}/beta-samples--{}--{}.npy".format(os.getcwd(), _uid, _n))
            except Exception as e:
                print(e)

            try:
                os.remove("{}/alpha-samples--{}--{}.npy".format(os.getcwd(), _uid, _n))
            except Exception as e:
                print(e)

            try:
                os.remove("{}/F-samples--{}--{}.npy".format(os.getcwd(), _uid, _n))
            except Exception as e:
                print(e)

            try:
                os.remove("{}/logtheta-samples--{}--{}.npy".format(os.getcwd(), _uid, _n))
            except Exception as e:
                print(e)

            try:
                os.remove("{}/logalpha-samples--{}--{}.npy".format(os.getcwd(), _uid, _n))
            except Exception as e:
                print(e)
            try:
                os.remove("{}/log-alpha-samples--{}--{}.npy".format(os.getcwd(), _uid, _n))
            except Exception as e:
                print(e)
