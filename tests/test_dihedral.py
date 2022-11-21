import sys
if '..' not in sys.path:
    sys.path.append('..')

from dihedral import DihGroup, Dih_irreps

dih = DihGroup(4)
irreps = Dih_irreps(dih.N)

print("Elements:")
print(dih.elements)
print("-" * 80 + "\n")

print("Multiplication table:")
for r in dih.mul_table():
    print(r)
print("-" * 80 + "\n")

print("Conjugacy classes:")
for cc in dih.conj_classes():
    print(cc)
print("-" * 80 + "\n")

print("Character tables for the generators (r and s)")
for irr in irreps.chars:
    print([irr(dih.elem(1, 0)), irr(dih.elem(0, 1))])
print()

print("Character tables for conjugacy class")
for irr in irreps.chars:
    print([irr(c[0]) for c in dih.conj_classes()])
print()
print("-" * 80 + "\n")

print(dih)
print(irreps)
print()
print("Matrix elements")
for g in dih:
    print(repr(g))
    print(irreps.mels_dict(g))
    # print("Left repr:", irreps.mels(g), "\nRight repr:", irreps.mels(~g))
    print()
