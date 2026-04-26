# Overflow-safety harness for R3D.IntExact.
#
# Why this script exists: IntExact's per-clip GCD reduction keeps the
# numerator/denominator bit-width in check empirically, but it is a
# heuristic — there is no closed-form bound on bit growth across
# arbitrary chains of oblique-integer-coefficient cuts. Downstream
# consumers picking a storage type T need an empirical answer to
# "how many such clips can I survive before T overflows?"
#
# What it measures: for each integer storage type T and each of three
# fixed oblique-cut sequences, applies up to N clips to a unit
# 10x10x10 D=3 cube, scanning every surviving vertex after each clip
# for the maximum bit-width of |positions_num[k, v]| and
# |positions_den[v]|. Reports the depth at which the max bit-width
# saturates the type (or N, the budget, if the type survived).
#
# How to read the table: rows = T. Columns:
#   - depth_reached  - how many clips completed before halt (== N
#                      means "T survived"; less means overflow or
#                      capacity exhaustion stopped the loop)
#   - max_num_bits   - largest |numerator| bit-width observed
#   - max_den_bits   - largest |denominator| bit-width observed
#   - status         - ok | clip_returned_false |
#                      bitwidth_near_capacity
#
# Decision rule: if your consumer's expected clip depth (per polytope)
# is <= depth_reached for the named T at numerator+denominator
# bit-budget within max_*_bits, that T is safe. Otherwise, step up
# (Int64 -> Int128 -> BigInt).
#
# Run:  julia --project=/tmp/r3djl_ci_env R3D.jl/bench/intexact_overflow.jl

using R3D
using Random

const N_CLIPS = 8
# Three fixed seeds so the table is reproducible across runs / machines.
const SEEDS = (UInt(0xC0FFEE01), UInt(0xC0FFEE02), UInt(0xC0FFEE03))

"""
Build an oblique integer-coefficient plane that grazes the cube
volume rather than slicing off most of it. Coefficients are drawn
from a small range to keep the planes genuinely oblique (no
axis parallels) without inflating per-clip linear combinations
beyond what GCD reduction can recover.

Strategy: pick a normal n in [-3, 3]^3, then pick a target offset
inside the (-c, +c) span where c = sum(|n[k]|) * 10 / 2 — this
guarantees the half-space {n.x + d >= 0} keeps a substantial
chunk of the cube. Without this trick, three random "all-positive
n" planes in a row trivially empty the polytope and the bench
stops measuring anything useful.
"""
function random_oblique_plane(rng, ::Type{T}) where {T}
    while true
        n1 = T(rand(rng, -3:3))
        n2 = T(rand(rng, -3:3))
        n3 = T(rand(rng, -3:3))
        if abs(n1) + abs(n2) + abs(n3) >= 2 &&
                count(!iszero, (n1, n2, n3)) >= 2
            # n . v ranges over [n_lo, n_hi] for v in cube = [0, 10]^3.
            n_lo = T(0)
            n_hi = T(0)
            for n in (n1, n2, n3)
                if n > 0
                    n_hi += T(10) * n
                else
                    n_lo += T(10) * n
                end
            end
            # Pick d so the cut sits in the middle 60% of the cube extent.
            span = n_hi - n_lo
            mid  = div(n_lo + n_hi, 2)
            jitter = div(span, 5)
            d = -(mid + T(rand(rng, -jitter:jitter)))
            return R3D.Plane{3, T}(R3D.Vec{3, T}(n1, n2, n3), d)
        end
    end
end

"""
Scan all live vertices, return (max |num| bits, max |den| bits).
For BigInt this is a true bit-count via Base.ndigits(x; base = 2).
"""
function scan_bitwidth(poly)
    mx_num = 0
    mx_den = 0
    @inbounds for v in 1:poly.nverts
        d = abs(poly.positions_den[v])
        bw = d == 0 ? 0 : Base.ndigits(d; base = 2)
        mx_den = max(mx_den, bw)
        for k in 1:size(poly.positions_num, 1)
            x = abs(poly.positions_num[k, v])
            bw = x == 0 ? 0 : Base.ndigits(x; base = 2)
            mx_num = max(mx_num, bw)
        end
    end
    return mx_num, mx_den
end

"""
Run one (T, seed) trial. Returns
(depth_reached, max_num_bits, max_den_bits, status::Symbol).

Status semantics:
  :ok                       - completed all N_CLIPS without overflow.
  :clip_returned_false      - capacity exhaustion / overflow inside
                              the kernel.
  :bitwidth_near_capacity   - bit-width reached the type's safety
                              threshold; further clips would risk
                              silent wrap-around.
  :polytope_emptied         - the random plane sequence wholly
                              clipped the polytope away; not a T
                              fault, just an unlucky cut.
"""
function run_trial(::Type{T}, seed::UInt) where {T}
    rng = Random.Xoshiro(seed)
    poly = R3D.IntExact.box((T(0), T(0), T(0)),
                            (T(10), T(10), T(10));
                            capacity = 256)

    # For fixed-width Ts, halt before the type's signed-max is reached.
    # A safety margin of 3 bits keeps us clear of the per-clip linear
    # combination's intermediate products.
    cap_bits = T <: BigInt ? typemax(Int) : 8 * sizeof(T) - 1
    margin   = 3
    threshold = T <: BigInt ? typemax(Int) : cap_bits - margin

    mx_num = 0
    mx_den = 0
    status = :ok
    depth = 0

    for step in 1:N_CLIPS
        plane = random_oblique_plane(rng, T)
        ok = R3D.IntExact.clip!(poly, plane)
        if !ok
            status = :clip_returned_false
            break
        end
        depth = step
        nb, db = scan_bitwidth(poly)
        mx_num = max(mx_num, nb)
        mx_den = max(mx_den, db)
        if !(T <: BigInt) && (mx_num >= threshold || mx_den >= threshold)
            status = :bitwidth_near_capacity
            break
        end
        if poly.nverts == 0
            status = :polytope_emptied
            break
        end
    end
    return depth, mx_num, mx_den, status
end

function format_row(label, depth, mx_num, mx_den, status)
    return "| $(rpad(label, 7)) | $(lpad(depth, 13)) | $(lpad(mx_num, 12)) | $(lpad(mx_den, 12)) | $(rpad(String(status), 22)) |"
end

function main()
    types = (Int32, Int64, Int128, BigInt)
    println("# IntExact overflow-safety harness")
    println()
    println("Cube: [0, 10]^3, oblique-plane clips with coefficients in [-3, 3].")
    println("N_CLIPS = $N_CLIPS per trial, $(length(SEEDS)) seeds per T (worst-case reported).")
    println()
    println("| T       | depth_reached | max_num_bits | max_den_bits | status                 |")
    println("|---------|---------------|--------------|--------------|------------------------|")
    for T in types
        # Worst-case across the seeds, but ignore :polytope_emptied
        # outcomes when picking the worst depth — those reflect the
        # random plane sequence, not a T fault. The shallowest
        # overflow / capacity trial wins; if no trial overflowed,
        # report the deepest depth among the survivors.
        worst_num   = 0
        worst_den   = 0
        fault_depth   = nothing
        fault_status  = :ok
        survivor_depth = 0
        for seed in SEEDS
            d, n, dn, s = run_trial(T, seed)
            worst_num = max(worst_num, n)
            worst_den = max(worst_den, dn)
            if s == :clip_returned_false || s == :bitwidth_near_capacity
                if fault_depth === nothing || d < fault_depth
                    fault_depth  = d
                    fault_status = s
                end
            else
                survivor_depth = max(survivor_depth, d)
            end
        end
        depth, status = fault_depth === nothing ?
            (survivor_depth, :ok) : (fault_depth, fault_status)
        println(format_row(string(T), depth, worst_num, worst_den, status))
    end
    println()
    println("Read this as: rows toward the bottom (Int128, BigInt) carry")
    println("more headroom; rows toward the top (Int32) saturate first.")
    println("If your consumer's clip depth is <= depth_reached and the")
    println("expected vertex bit-width is <= max_*_bits, the named T")
    println("is safe for that workload.")
end

main()
