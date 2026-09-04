#!/bin/bash
# Generate the joint I x delta datasets for the SMW favourable-regime study.
#
# Two families are produced:
#
#   data/joint/I<n>_d<delta>_dm10   the *favourable* regime: the contrast mesh
#       is pinned at delta_m = 10 while the field mesh is refined, so P stays
#       at 394 and P/(I*L) falls from 0.88 (I=2, delta=10) to 0.0036
#       (I=32, delta=40). This is the regime P << I*L that Sec. 5.8 says the
#       SMW route is built for, and that no measurement so far has entered.
#
#   data/joint/I<n>_d40_dm40        the *unfavourable* companion at large P
#       (P = 6720 > L = 3461), kept so the two regimes can be compared at the
#       same field mesh, and so the CG tolerance-schedule question can be
#       re-asked where the capacitance is large.
#
# Usage:  bash scripts/generate_joint_sweep.sh [outdir]
set -euo pipefail

ROOT="${1:-data/joint}"
EDP="scripts/GenerateMatrixSweep.edp"

gen () {   # gen <I> <delta> <delta_m>
  local i="$1" d="$2" dm="$3"
  local out="${ROOT}/I${i}_d${d}_dm${dm}"
  if [ -f "${out}/dims.txt" ]; then
    echo "[skip] ${out} already generated"
    return
  fi
  mkdir -p "${out}"
  echo "[gen ] I=${i} delta=${d} delta_m=${dm} -> ${out}"
  FreeFem++ -nw "${EDP}" -I "${i}" -delta "${d}" -delta_m "${dm}" -out "${out}/" \
    > "${out}/generate.log" 2>&1 \
    || { echo "FAILED, see ${out}/generate.log"; tail -20 "${out}/generate.log"; exit 1; }
  grep -E '^(I|delta|delta_m|L|P|J) ' "${out}/dims.txt" | tr '\n' ' '; echo
}

# Favourable regime: coarse contrast mesh held fixed.
for d in 10 20 40; do
  for i in 2 4 8 16 32; do
    gen "$i" "$d" 10
  done
done

# Unfavourable companion at the finest field mesh (P = 6720).
for i in 2 8 16; do
  gen "$i" 40 40
done

echo
echo "Datasets under ${ROOT}:"
for f in "${ROOT}"/*/dims.txt; do
  printf '  %-24s ' "$(basename "$(dirname "$f")")"
  tr '\n' ' ' < "$f"; echo
done
