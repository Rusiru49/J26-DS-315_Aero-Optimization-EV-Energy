p = "make_urban.py"; s = open(p).read()
if "dm = 100" in s: print("already patched"); raise SystemExit
a1 = '    if not low[d] and abs(dd) > 1.0: msgs.append("Cd_sum differs %.2f%%" % dd)\n'
b1 = '    ref = float(O[(d, 40)]["Cd_mean"]); dm = 100*(cd_sum-ref)/ref\n    if not low[d] and abs(dm) > 1.0: msgs.append("Cd_sum differs from Cd_mean by %.2f%%" % dm)\n'
a2 = '    print("%s faces %d Cd(file) %.4f Cd(sum) %.4f (%.2f%%)  %s" % (d, n, cd, cd_sum, dd, "FAIL: " + "; ".join(msgs) if msgs else "OK"))\n'
b2 = '    print("%s faces %d Cd(file) %.4f Cd_mean %.4f Cd(sum) %.4f (vs Cd_mean %.2f%%)  %s" % (d, n, cd, ref, cd_sum, dm, "FAIL: " + "; ".join(msgs) if msgs else "OK"))\n'
for a, b in ((a1, b1), (a2, b2)):
    if a not in s: print("PATTERN NOT FOUND:", a[:60]); raise SystemExit(1)
s = s.replace(a1, b1)
s = s.replace(a2, "")
s = s.replace(b1, b1 + b2)
open(p, "w").write(s); print("patched", p)
