#![feature(test)]
extern crate test;

use std::cmp::Ordering;

use bitonic_cuda::{generate_some_data, GpuSorter, Struct};
use rayon::slice::ParallelSliceMut;
use test::{black_box, Bencher};

fn bench_sorter(bencher: &mut Bencher, size: usize, sorter: impl Fn(&mut [Struct])) {
    let elements = generate_some_data(size);
    let mut sortable_elements = Vec::with_capacity(size);
    bencher.iter(|| {
        sortable_elements.clear();
        for element in &elements {
            sortable_elements.push(*element);
        }
        black_box(sorter(sortable_elements.as_mut_slice()));
    });
}

#[inline]
fn struct_cmp(a: &Struct, b: &Struct) -> Ordering {
    if a.value < b.value {
        Ordering::Less
    } else if a.value == b.value {
        Ordering::Equal
    } else {
        Ordering::Greater
    }
}

fn std_stable(slice: &mut [Struct]) {
    slice.sort_by(struct_cmp);
}

fn std_unstable(slice: &mut [Struct]) {
    slice.sort_unstable_by(struct_cmp);
}

fn rayon_stable(slice: &mut [Struct]) {
    slice.par_sort_by(struct_cmp);
}

fn rayon_unstable(slice: &mut [Struct]) {
    slice.par_sort_unstable_by(struct_cmp);
}

// 32,768

#[bench]
fn std_stable_32768(bencher: &mut Bencher) {
    bench_sorter(bencher, 32768, std_stable);
}

#[bench]
fn rayon_stable_32768(bencher: &mut Bencher) {
    bench_sorter(bencher, 32768, rayon_stable);
}

#[bench]
fn std_unstable_32768(bencher: &mut Bencher) {
    bench_sorter(bencher, 32768, std_unstable);
}

#[bench]
fn rayon_unstable_32768(bencher: &mut Bencher) {
    bench_sorter(bencher, 32768, rayon_unstable);
}

#[bench]
fn bitonic_32768(bencher: &mut Bencher) {
    let sorter = GpuSorter::new(0).unwrap();
    bench_sorter(bencher, 32768, |slice| sorter.sort_structs(slice).unwrap());
}

// 65,536

#[bench]
fn std_stable_65536(bencher: &mut Bencher) {
    bench_sorter(bencher, 65536, std_stable);
}

#[bench]
fn rayon_stable_65536(bencher: &mut Bencher) {
    bench_sorter(bencher, 65536, rayon_stable);
}

#[bench]
fn std_unstable_65536(bencher: &mut Bencher) {
    bench_sorter(bencher, 65536, std_unstable);
}

#[bench]
fn rayon_unstable_65536(bencher: &mut Bencher) {
    bench_sorter(bencher, 65536, rayon_unstable);
}

#[bench]
fn bitonic_65536(bencher: &mut Bencher) {
    let sorter = GpuSorter::new(0).unwrap();
    bench_sorter(bencher, 65536, |slice| sorter.sort_structs(slice).unwrap());
}

// 1,048,576

#[bench]
fn std_stable_1048576(bencher: &mut Bencher) {
    bench_sorter(bencher, 1048576, std_stable);
}

#[bench]
fn rayon_stable_1048576(bencher: &mut Bencher) {
    bench_sorter(bencher, 1048576, rayon_stable);
}

#[bench]
fn std_unstable_1048576(bencher: &mut Bencher) {
    bench_sorter(bencher, 1048576, std_unstable);
}

#[bench]
fn rayon_unstable_1048576(bencher: &mut Bencher) {
    bench_sorter(bencher, 1048576, rayon_unstable);
}

#[bench]
fn bitonic_1048576(bencher: &mut Bencher) {
    let sorter = GpuSorter::new(0).unwrap();
    bench_sorter(bencher, 1048576, |slice| sorter.sort_structs(slice).unwrap());
}

// 33,554,432

#[bench]
fn std_stable_33554432(bencher: &mut Bencher) {
    bench_sorter(bencher, 33554432, std_stable);
}

#[bench]
fn rayon_stable_33554432(bencher: &mut Bencher) {
    bench_sorter(bencher, 33554432, rayon_stable);
}

#[bench]
fn std_unstable_33554432(bencher: &mut Bencher) {
    bench_sorter(bencher, 33554432, std_unstable);
}

#[bench]
fn rayon_unstable_33554432(bencher: &mut Bencher) {
    bench_sorter(bencher, 33554432, rayon_unstable);
}

#[bench]
fn bitonic_33554432(bencher: &mut Bencher) {
    let sorter = GpuSorter::new(0).unwrap();
    bench_sorter(bencher, 33554432, |slice| sorter.sort_structs(slice).unwrap());
}

// 1,073,741,824

#[bench]
fn bitonic_1073741824(bencher: &mut Bencher) {
    let sorter = GpuSorter::new(0).unwrap();
    bench_sorter(bencher, 1073741824, |slice| sorter.sort_structs(slice).unwrap());
}

#[bench]
fn rayon_unstable_1073741824(bencher: &mut Bencher) {
    bench_sorter(bencher, 1073741824, rayon_unstable);
}
