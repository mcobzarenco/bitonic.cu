# CUDA bitonic sort in rust

Implements [bitonic parallel
sort](https://en.m.wikipedia.org/wiki/Bitonic_sorter) in CUDA, using
rust for the host side.

For now, it only sorts `f32`s and it only works on power-of-two arrays.

Benchmarks on a Ubuntu 23.10 desktop PC
- CPU: 24 core 12th Gen Intel(R) Core(TM) i9-12900KS
- GPU: NVIDIA GeForce RTX 3090 Ti

```
~ cargo bench

test bitonic_32768           ... bench:       100,796 ns/iter (+/- 2,366)
test bitonic_65536           ... bench:       152,260 ns/iter (+/- 2,395)
test bitonic_1048576         ... bench:     2,428,581 ns/iter (+/- 201,018)
test bitonic_33554432        ... bench:    85,287,101 ns/iter (+/- 6,548,327)
test rayon_stable_32768      ... bench:       231,308 ns/iter (+/- 5,226)
test rayon_stable_65536      ... bench:       345,339 ns/iter (+/- 8,769)
test rayon_stable_1048576    ... bench:     4,682,997 ns/iter (+/- 295,796)
test rayon_stable_33554432   ... bench:   155,202,823 ns/iter (+/- 7,103,368)
test rayon_unstable_32768    ... bench:       154,876 ns/iter (+/- 7,472)
test rayon_unstable_65536    ... bench:       283,243 ns/iter (+/- 21,623)
test rayon_unstable_1048576  ... bench:     4,112,566 ns/iter (+/- 839,600)
test rayon_unstable_33554432 ... bench:   111,669,328 ns/iter (+/- 7,156,464)
test std_stable_32768        ... bench:     1,278,452 ns/iter (+/- 75,445)
test std_stable_65536        ... bench:     2,732,487 ns/iter (+/- 90,811)
test std_stable_1048576      ... bench:    55,422,265 ns/iter (+/- 2,664,924)
test std_stable_33554432     ... bench: 2,247,221,910 ns/iter (+/- 11,668,572)
test std_unstable_32768      ... bench:       554,653 ns/iter (+/- 11,537)
test std_unstable_65536      ... bench:     1,152,406 ns/iter (+/- 43,448)
test std_unstable_1048576    ... bench:    20,855,117 ns/iter (+/- 190,036)
test std_unstable_33554432   ... bench:   754,761,743 ns/iter (+/- 21,142,560)
```
