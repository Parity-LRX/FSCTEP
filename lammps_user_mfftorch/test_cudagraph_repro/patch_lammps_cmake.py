#!/usr/bin/env python3
"""Patch LAMMPS cmake/CMakeLists.txt to recognize USER-MFFTORCH:
  (1) add USER-MFFTORCH to set(STANDARD_PACKAGES ...) so PKG_USER-MFFTORCH exists;
  (2) add USER-MFFTORCH to foreach(PKG_WITH_INCL ...) so the package cmake module
      (which links LibTorch) gets include()d when enabled.
Both edits are paren-matched (robust to formatting/version) and idempotent.
If foreach(PKG_WITH_INCL ...) is absent, falls back to appending an explicit
include guard right after the lammps target is defined.
Usage: python3 patch_lammps_cmake.py /path/to/lammps/cmake/CMakeLists.txt
"""
import sys


def insert_before_matching_paren(s, call_token, item):
    idx = s.find(call_token)
    if idx < 0:
        return s, False, False  # not found
    popen = s.find('(', idx)
    if popen < 0:
        return s, False, False
    depth = 0
    i = popen
    while i < len(s):
        if s[i] == '(':
            depth += 1
        elif s[i] == ')':
            depth -= 1
            if depth == 0:
                break
        i += 1
    if i >= len(s):
        return s, False, False
    if item in s[popen:i]:
        return s, False, True  # already present
    return s[:i] + '\n  ' + item + '\n' + s[i:], True, True


def main():
    path = sys.argv[1]
    with open(path) as f:
        s = f.read()
    orig = s

    s, c1, _ = insert_before_matching_paren(s, 'set(STANDARD_PACKAGES', 'USER-MFFTORCH')
    s, c2, found2 = insert_before_matching_paren(s, 'foreach(PKG_WITH_INCL', 'USER-MFFTORCH')

    # Fallback: if there is no PKG_WITH_INCL foreach to join, add an explicit include
    # guard right after the lammps library target is created (so target_sources works).
    if not found2 and 'include(Packages/USER-MFFTORCH)' not in s:
        anchor = 'add_library(lammps'
        ai = s.find(anchor)
        if ai >= 0:
            # insert after the end of that statement's line
            eol = s.find('\n', ai)
            inject = ('\n# USER-MFFTORCH: link LibTorch when enabled\n'
                      'if(PKG_USER-MFFTORCH)\n'
                      '  include(Packages/USER-MFFTORCH)\n'
                      'endif()\n')
            s = s[:eol + 1] + inject + s[eol + 1:]
            c2 = True

    if s != orig:
        with open(path, 'w') as f:
            f.write(s)
    print(f"patch_lammps_cmake: STANDARD_PACKAGES_changed={c1} PKG_WITH_INCL_changed={c2}")


if __name__ == '__main__':
    main()
