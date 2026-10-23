p = "map_remesh.py"; s = open(p).read()
if "--pkg=" in s: print("already patched"); raise SystemExit
a = 'pkg = os.path.expanduser("~/Documents/EV_Aero/for_component3/npz")'
b = 'pkg = os.path.expanduser("~/Documents/EV_Aero/for_component3/npz")\n        for _a in sys.argv[1:]:\n            if _a.startswith("--pkg="): pkg = os.path.expanduser(_a.split("=", 1)[1])\n        os.makedirs(pkg, exist_ok=True)'
if a not in s: print("PATTERN NOT FOUND"); raise SystemExit(1)
open(p, "w").write(s.replace(a, b)); print("patched", p)
